"""Build the submission report from the template and measured artifacts."""
import json
import random
import re
from collections import Counter
from pathlib import Path
from string import Template

from core import LABELS

ROOT = Path(__file__).resolve().parent


def load(name):
    return json.loads((ROOT / name).read_text(encoding="utf-8"))


def sample_counts(split, limit):
    rows = [json.loads(line) for line in (ROOT / f"data/{split}.jsonl").read_text(encoding="utf-8").splitlines()]
    random.Random(42).shuffle(rows)
    selected = rows[:limit] if limit else rows
    return Counter(y for row in selected for y in row["labels"])


def as_markdown(text):
    lines = text.splitlines()
    output = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if i == 0:
            output.append("# " + line)
        elif line.startswith("表 "):
            output.extend(["**" + line + "**", ""])
            rows = []
            i += 1
            while i < len(lines) and lines[i].strip():
                rows.append(re.split(r"\s{2,}", lines[i].strip()))
                i += 1
            if line.startswith("表 3"):
                rows.insert(0, ["参数", "设置"])
            width = len(rows[0])
            if any(len(row) != width for row in rows):
                raise ValueError(f"Inconsistent columns in {line}")
            output.append("| " + " | ".join(rows[0]) + " |")
            output.append("| " + " | ".join(["---"] * width) + " |")
            output.extend("| " + " | ".join(row) + " |" for row in rows[1:])
            output.append("")
            continue
        elif line == "摘　要" or re.match(r"^(\d+　|附录 [A-C]　)", line):
            output.append("## " + line)
        elif re.match(r"^(\d+\.\d+|B\.\d+)　", line):
            output.append("### " + line)
        elif re.match(r"例 \d+$", line):
            output.append("#### " + line)
        elif line.startswith(("python", "source .venv", "pip install", "mkdir -p", "bsdtar -xf")):
            output.append("```bash")
            while i < len(lines) and lines[i].strip():
                output.append(lines[i])
                i += 1
            output.extend(["```", ""])
            continue
        elif line.startswith("prepare.py "):
            output.append("```text")
            while i < len(lines) and lines[i].strip():
                output.append(lines[i])
                i += 1
            output.extend(["```", ""])
            continue
        elif re.match(r"^（[1-6]）", line):
            output.append(re.sub(r"^（([1-6])）", r"\1. ", line))
        elif line.startswith(("输入：", "参考：", "预测：")):
            output.extend(["**" + line[:3] + "**" + line[3:], ""])
        elif line.startswith(("hᵢ =", "pᵢ =", "L =", "w(i)=", "P(j,c)=", "ŷⱼ=", "Precision=", "F1=")):
            output.append("```text")
            while i < len(lines) and lines[i].strip():
                output.append(lines[i])
                i += 1
            output.extend(["```", ""])
            continue
        else:
            prefix = "- " if line.startswith(("操作系统：", "Python：", "PyTorch：", "Transformers：", "实际训练设备：")) else ""
            output.append(prefix + line)
        i += 1
    return "\n".join(output) + "\n"


def main():
    train = load("results/training.json")
    test = load("results/test.json")
    bench = load("results/benchmark.json")
    ancient = load("results/ancient_examples.json")
    stats = load("data/stats.json")
    env = load("results/environment.json")
    args = train["arguments"]
    counts = {s: sample_counts(s, args["train_limit"] if s == "train" else args["eval_limit"])
              for s in ("train", "dev", "test")}
    split_names = {"train": "训练集", "dev": "验证集", "test": "测试集"}
    dataset_table = "\n".join(
        f"{split_names[s]}    {v['articles']:,} 篇    {v['paragraphs']:,} 段    {v['characters']:,} 字"
        for s, v in stats.items())
    class_table = "\n".join(
        f"{label}    {counts['train'][i]:>6}    {counts['dev'][i]:>6}    {counts['test'][i]:>6}"
        for i, label in enumerate(LABELS))
    history_table = "\n".join(
        f"{h['epoch']}    {h['train_loss']:.4f}    "
        f"{h['dev']['punctuation_micro']['f1']:.4f}    {h['dev']['boundary']['f1']:.4f}"
        for h in train["history"])
    test_table = "\n".join(
        f"{title}    {test[k]['precision']:.4f}    {test[k]['recall']:.4f}    {test[k]['f1']:.4f}"
        for k, title in (("punctuation_micro", "标点分类"), ("boundary", "断点识别")))
    per_class_table = "\n".join(
        f"{label}    {counts['test'][i]:>4}    {test['per_label'][label]['precision']:.4f}    "
        f"{test['per_label'][label]['recall']:.4f}    {test['per_label'][label]['f1']:.4f}"
        for i, label in enumerate(LABELS[1:], 1))
    results = bench["results"]
    names = {"serial_period": "串行句号截断", "fixed_batch_1": "固定窗口 B=1", "fixed_batch_n": "固定窗口 B=8"}
    speed_table = "\n".join(
        f"{name}    {results[k]['forward_calls']:>2}    {results[k]['seconds_median']:.4f}    "
        f"{results[k]['chars_per_second']:.1f}    {results[k]['metrics']['punctuation_micro']['f1']:.4f}"
        for k, name in names.items())
    time_table = "\n".join(f"{name}    " + "    ".join(f"{t:.4f}" for t in results[k]["seconds_all"])
                           for k, name in names.items())
    examples = "\n\n".join(f"例 {i}\n输入：{r['input']}\n参考：{r['reference']}\n预测：{r['prediction']}"
                           for i, r in enumerate(ancient["examples"], 1))
    best = max(train["history"], key=lambda h: h["dev"]["punctuation_micro"]["f1"])
    values = dict(dataset_table=dataset_table, class_table=class_table, history_table=history_table,
                  test_table=test_table, per_class_table=per_class_table, speed_table=speed_table,
                  time_table=time_table, examples=examples, python=env["python"].split()[0],
                  torch=env["torch"], transformers=env["transformers"], platform=env["platform"],
                  device=train["device"], train_paragraphs=f"{train['train_paragraphs']:,}",
                  train_windows=f"{train['train_windows']:,}", train_chars=f"{sum(counts['train'].values()):,}",
                  dev_chars=f"{sum(counts['dev'].values()):,}", test_chars=f"{test['characters']:,}",
                  eval_limit=args["eval_limit"], epochs=args["epochs"], batch=args["batch"],
                  train_window=args["train_window"], window=args["window"], stride=args["stride"],
                  freeze=args["freeze_layers"], lr=args["lr"], seed=args["seed"],
                  seconds=f"{train['seconds']:.2f}", best_epoch=best["epoch"],
                  test_f1=f"{test['punctuation_micro']['f1']:.4f}",
                  boundary_f1=f"{test['boundary']['f1']:.4f}", accuracy=f"{test['accuracy']:.4f}",
                  ancient_f1=f"{ancient['metrics']['punctuation_micro']['f1']:.4f}",
                  batch_speedup=f"{results['batch_speedup']:.3f}", overall_speedup=f"{results['overall_speedup']:.3f}",
                  overall_reduction=f"{100*(1-1/results['overall_speedup']):.1f}",
                  batch_reduction=f"{100*(1-1/results['batch_speedup']):.1f}",
                  disagreements=results["batch_prediction_disagreements"], sha=env["pretrained_sha256"])
    template = Template((ROOT / "REPORT.template.txt").read_text(encoding="utf-8"))
    (ROOT / "REPORT.md").write_text(as_markdown(template.substitute(values)), encoding="utf-8")
    print("Generated REPORT.md from measured results and REPORT.template.txt.")


if __name__ == "__main__":
    main()
