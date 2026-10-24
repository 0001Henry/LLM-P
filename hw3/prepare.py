import argparse
import json
from pathlib import Path
from core import clean_line, encode_text, split_for


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("data/raw/1998-上半年-带音/1998-01-2003版-带音.txt"))
    parser.add_argument("--output", type=Path, default=Path("data"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = {k: [] for k in ("train", "dev", "test")}
    seen = set()
    for line in args.input.read_text(encoding="gb18030").splitlines():
        parsed = clean_line(line)
        if not parsed:
            continue
        article, original = parsed
        text, labels = encode_text(original)
        if len(text) < 10 or text in seen:
            continue
        seen.add(text)
        rows[split_for(article)].append({"article": article, "text": text, "labels": labels})
    stats = {}
    for split, examples in rows.items():
        with (args.output / f"{split}.jsonl").open("w", encoding="utf-8") as f:
            for row in examples:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        stats[split] = {"paragraphs": len(examples), "articles": len({r['article'] for r in examples}),
                        "characters": sum(len(r['text']) for r in examples)}
    (args.output / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
