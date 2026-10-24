"""Fine tuning, batched overlapping inference and reproducible evaluation."""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import BertForTokenClassification, BertTokenizer

from core import LABELS, encode_text, metrics, render, windows


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def read_rows(split, limit=0):
    rows = [json.loads(x) for x in Path(f"data/{split}.jsonl").read_text(encoding="utf-8").splitlines()]
    random.Random(42).shuffle(rows)
    return rows[:limit] if limit else rows


def device_for(name):
    if name != "auto":
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


class Engine:
    def __init__(self, model, device="auto", pretrained=False):
        self.device = device_for(device)
        self.tokenizer = BertTokenizer.from_pretrained(model)
        kwargs = dict(num_labels=len(LABELS), id2label=dict(enumerate(LABELS)),
                      label2id={x: i for i, x in enumerate(LABELS)}) if pretrained else {}
        # MPS SDPA does not implement attention dropout during training.
        if self.device.type == "mps":
            kwargs["attn_implementation"] = "eager"
        self.model = BertForTokenClassification.from_pretrained(model, **kwargs).to(self.device)
        if not pretrained and [self.model.config.id2label[i] for i in range(len(LABELS))] != LABELS:
            raise ValueError("Checkpoint label mapping does not match this task")
        self.calls = 0

    def pack(self, texts, labels=None):
        # Exactly one vocabulary lookup per Unicode character; no WordPiece drift.
        ids = [[self.tokenizer.cls_token_id] + self.tokenizer.convert_tokens_to_ids([c.lower() for c in t])
               + [self.tokenizer.sep_token_id] for t in texts]
        if any(len(x) > self.model.config.max_position_embeddings for x in ids):
            raise ValueError("Input exceeds BERT position limit")
        width = max(map(len, ids))
        result = {"input_ids": torch.tensor([x + [self.tokenizer.pad_token_id]*(width-len(x)) for x in ids]),
                  "attention_mask": torch.tensor([[1]*len(x) + [0]*(width-len(x)) for x in ids])}
        if labels is not None:
            result["labels"] = torch.tensor([[-100] + list(y) + [-100]*(width-len(y)-1) for y in labels])
        return {k: v.to(self.device) for k, v in result.items()}

    def sync(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        elif self.device.type == "mps":
            torch.mps.synchronize()

    @torch.inference_mode()
    def probabilities(self, texts):
        self.model.eval()
        self.calls += 1
        logits = self.model(**self.pack(texts)).logits
        probs = logits.float().softmax(-1).cpu().numpy()
        return [p[1:len(t)+1] for p, t in zip(probs, texts)]

    def predict(self, text, size=400, stride=300, batch=8, serial=False):
        if batch < 1:
            raise ValueError("Batch size must be positive")
        spans = list(windows(len(text), size, stride))
        if not text:
            return []
        if serial:
            result, start = [], 0
            while start < len(text):
                end = min(start+size, len(text))
                labels = self.probabilities([text[start:end]])[0].argmax(-1).tolist()
                take = len(labels) if end == len(text) else min(stride, len(labels))
                if end != len(text):
                    stops = [i+1 for i in range(take) if labels[i] == 2]
                    # Fallback guarantees progress even when no period is predicted.
                    take = stops[-1] if stops else take
                result.extend(labels[:take])
                start += take
            return result
        scores = np.zeros((len(text), len(LABELS)), dtype=np.float32)
        weight_sum = np.zeros(len(text), dtype=np.float32)
        for offset in range(0, len(spans), batch):
            group = spans[offset:offset+batch]
            probs = self.probabilities([text[s:e] for s, e in group])
            for (s, e), p in zip(group, probs):
                n = e-s
                weights = np.minimum(np.arange(1, n+1), np.arange(n, 0, -1)).astype(np.float32)
                scores[s:e] += p * weights[:, None]
                weight_sum[s:e] += weights
        return (scores / weight_sum[:, None]).argmax(-1).tolist()


def evaluate(engine, rows, args):
    gold, pred = [], []
    for row in rows:
        gold.extend(row["labels"])
        pred.extend(engine.predict(row["text"], args.window, args.stride, args.batch))
    return metrics(gold, pred)


def train(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    engine = Engine(args.model, args.device, pretrained=True)
    rows = read_rows("train", args.train_limit)
    samples = [(r["text"][s:e], r["labels"][s:e]) for r in rows
               for s, e in windows(len(r["text"]), args.train_window, args.train_window)]
    loader = DataLoader(samples, batch_size=args.batch, shuffle=True, collate_fn=lambda x: x)
    if args.freeze_layers:
        for p in engine.model.bert.embeddings.parameters():
            p.requires_grad = False
        for layer in engine.model.bert.encoder.layer[:args.freeze_layers]:
            for p in layer.parameters():
                p.requires_grad = False
    optimizer = torch.optim.AdamW([p for p in engine.model.parameters() if p.requires_grad], lr=args.lr)
    dev = read_rows("dev", args.eval_limit)
    history, best = [], -1.0
    started = time.perf_counter()
    for epoch in range(args.epochs):
        engine.model.train()
        loss_sum, tokens = 0.0, 0
        for step, items in enumerate(loader, 1):
            texts, labels = zip(*items)
            optimizer.zero_grad(set_to_none=True)
            loss = engine.model(**engine.pack(texts, labels)).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(engine.model.parameters(), 1.0)
            optimizer.step()
            count = sum(map(len, texts))
            loss_sum += loss.item()*count
            tokens += count
            if step % 10 == 0:
                print(f"epoch={epoch+1} step={step}/{len(loader)} loss={loss_sum/tokens:.4f}", flush=True)
        result = evaluate(engine, dev, args)
        history.append({"epoch": epoch+1, "train_loss": loss_sum/tokens, "dev": result})
        print(json.dumps(history[-1], ensure_ascii=False), flush=True)
        if result["punctuation_micro"]["f1"] > best:
            best = result["punctuation_micro"]["f1"]
            engine.model.save_pretrained(args.output)
            engine.tokenizer.save_pretrained(args.output)
    engine.sync()
    save_json("results/training.json", {"arguments": vars(args), "device": str(engine.device),
              "train_paragraphs": len(rows), "train_windows": len(samples), "history": history,
              "seconds": time.perf_counter()-started})
    engine = Engine(args.output, args.device)
    save_json("results/test.json", evaluate(engine, read_rows("test", args.eval_limit), args))


def benchmark(args):
    engine = Engine(args.model, args.device)
    rows = read_rows("test", args.eval_limit)
    # Concatenate held-out paragraphs to exercise multi-thousand-character windows.
    text = "".join(r["text"] for r in rows)[:args.chars]
    gold = [y for r in rows for y in r["labels"]][:len(text)]
    engine.predict(text[:args.window], args.window, args.stride, args.batch)
    results, predictions = {}, {}
    for name, batch, serial in [("serial_period", 1, True), ("fixed_batch_1", 1, False),
                                 ("fixed_batch_n", args.batch, False)]:
        timings = []
        for _ in range(args.repeats):
            engine.calls = 0
            engine.sync()
            start = time.perf_counter()
            pred = engine.predict(text, args.window, args.stride, batch, serial)
            engine.sync()
            timings.append(time.perf_counter()-start)
        seconds = float(np.median(timings))
        predictions[name] = pred
        results[name] = {"seconds_median": seconds, "seconds_all": timings, "forward_calls": engine.calls,
                         "chars_per_second": len(text)/seconds, "metrics": metrics(gold, pred)}
    results["batch_speedup"] = results["fixed_batch_1"]["seconds_median"]/results["fixed_batch_n"]["seconds_median"]
    results["overall_speedup"] = results["serial_period"]["seconds_median"]/results["fixed_batch_n"]["seconds_median"]
    results["batch_prediction_disagreements"] = sum(a != b for a, b in zip(
        predictions["fixed_batch_1"], predictions["fixed_batch_n"]))
    save_json("results/benchmark.json", {"arguments": vars(args), "device": str(engine.device),
                                        "characters": len(text), "results": results})
    print(json.dumps(results, ensure_ascii=False, indent=2))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=["train", "predict", "benchmark", "evaluate"])
    p.add_argument("--model", help="Defaults to pretrained BERT for train, fine-tuned model otherwise")
    p.add_argument("--output", default="models/punctuation")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--window", type=int, default=400)
    p.add_argument("--stride", type=int, default=300)
    p.add_argument("--train-window", type=int, default=128)
    p.add_argument("--train-limit", type=int, default=0)
    p.add_argument("--eval-limit", type=int, default=0)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--freeze-layers", type=int, default=0)
    p.add_argument("--chars", type=int, default=4000)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--text", default="学而时习之不亦说乎有朋自远方来不亦乐乎")
    p.add_argument("--input", help="UTF-8 input file for long text")
    args = p.parse_args()
    if args.model is None:
        args.model = "models/bert-base-chinese" if args.command == "train" else "models/punctuation"
    if args.batch < 1 or args.epochs < 1 or args.repeats < 1 or args.chars < 1:
        p.error("batch, epochs, repeats and chars must be positive")
    if min(args.train_limit, args.eval_limit) < 0 or not 0 <= args.freeze_layers <= 12 or args.lr <= 0:
        p.error("Limits must be nonnegative, freeze-layers in [0, 12], lr positive")
    list(windows(1, args.window, args.stride))
    list(windows(1, args.train_window, args.train_window))
    if args.command == "train":
        train(args)
    elif args.command == "benchmark":
        benchmark(args)
    else:
        engine = Engine(args.model, args.device)
        if args.command == "evaluate":
            save_json("results/test.json", evaluate(engine, read_rows("test", args.eval_limit), args))
        else:
            raw = Path(args.input).read_text(encoding="utf-8") if args.input else args.text
            text, _ = encode_text(raw)
            output = render(text, engine.predict(text, args.window, args.stride, args.batch))
            Path("results").mkdir(exist_ok=True)
            Path("results/prediction.txt").write_text(output + "\n", encoding="utf-8")
            print(output)


if __name__ == "__main__":
    main()
