import unittest
import numpy as np
from bert_punct import Engine


class FakeEngine(Engine):
    def __init__(self, label=1):
        self.calls = 0
        self.label = label

    def probabilities(self, texts):
        self.calls += 1
        return [np.eye(8, dtype=np.float32)[[self.label]*len(t)] for t in texts]


class InferenceTests(unittest.TestCase):
    def test_real_bert_alignment_and_backward(self):
        import torch
        from transformers import BertConfig, BertForTokenClassification, BertTokenizer
        engine = FakeEngine()
        engine.device = torch.device("cpu")
        engine.tokenizer = BertTokenizer.from_pretrained("models/bert-base-chinese")
        # Random tiny BERT is exclusively an integration test, not the experiment model.
        engine.model = BertForTokenClassification(BertConfig(
            vocab_size=engine.tokenizer.vocab_size, hidden_size=16, num_hidden_layers=1,
            num_attention_heads=2, intermediate_size=32, num_labels=8))
        batch = engine.pack(["古文ABC𠮷İ", "甲"], [[0]*7, [2]])
        self.assertEqual(batch["input_ids"].shape, (2, 9))
        self.assertEqual(batch["labels"].tolist()[1], [-100, 2]+[-100]*7)
        output = engine.model(**batch)
        self.assertTrue(torch.isfinite(output.loss))
        output.loss.backward()
        self.assertIsNotNone(engine.model.classifier.weight.grad)

    def test_batch_equivalence_and_calls(self):
        engine = FakeEngine()
        text = "古"*5000
        a = engine.predict(text, batch=1)
        single_calls = engine.calls
        engine.calls = 0
        b = engine.predict(text, batch=8)
        self.assertEqual(a, b)
        self.assertEqual(len(b), len(text))
        self.assertLess(engine.calls, single_calls)

    def test_serial_progress(self):
        for label in (0, 2):
            engine = FakeEngine(label)
            self.assertEqual(engine.predict("文"*1201, serial=True), [label]*1201)

    def test_empty_and_short(self):
        engine = FakeEngine()
        self.assertEqual(engine.predict(""), [])
        self.assertEqual(engine.calls, 0)
        self.assertEqual(engine.predict("文"), [1])

    def test_overlapping_weights(self):
        engine = FakeEngine()
        def probs(texts):
            out = []
            for text in texts:
                p = np.zeros((len(text), 8), dtype=np.float32)
                p[:, 1 if text.startswith("甲") else 2] = 1
                out.append(p)
            return out
        engine.probabilities = probs
        result = engine.predict("甲乙丙丁戊己庚辛", size=6, stride=4)
        self.assertEqual(result, [1, 1, 1, 1, 1, 2, 2, 2])

    def test_real_corpus_split_disjoint(self):
        from bert_punct import read_rows
        rows = {k: read_rows(k) for k in ("train", "dev", "test")}
        articles = {k: {r["article"] for r in v} for k, v in rows.items()}
        for a, b in (("train", "dev"), ("train", "test"), ("dev", "test")):
            self.assertFalse(articles[a] & articles[b])
            self.assertFalse({r["text"] for r in rows[a]} & {r["text"] for r in rows[b]})
        for split in rows.values():
            self.assertTrue(all(len(r["text"]) == len(r["labels"]) for r in split))


if __name__ == "__main__":
    unittest.main()
