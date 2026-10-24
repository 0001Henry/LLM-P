"""Prepare article-level page classification without metadata features."""
import hashlib
import json
import random
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

SEED = 42
PAGES = ["01", "02", "03", "04", "05", "06"]
ROOT = Path(__file__).resolve().parent


def save_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def tokenizer():
    import jieba
    ROOT.joinpath(".cache").mkdir(exist_ok=True)
    obj = jieba.Tokenizer()
    obj.tmp_dir = str(ROOT / ".cache")
    return obj


def tokenize(text, segmenter):
    text = unicodedata.normalize("NFKC", text).lower()
    return [w for w in segmenter.lcut(text, HMM=False)
            if re.fullmatch(r"[\u3400-\u9fff]+|[a-z]+", w)]


def parse_line(line):
    m = re.match(r"^(\d{8})-(\d{2})-(\d{3})-(\d+)\s+(.+)$", line.strip())
    if not m:
        return None
    date, page, article, paragraph, text = m.groups()
    text = "".join(re.sub(r"\{[^}]*\}", "", text).split())
    return {"id": f"{date}-{page}-{article}", "page": page, "paragraph": int(paragraph), "text": text}


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def split_articles(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row["page"]].append(row)
    splits = {s: [] for s in ("train", "dev", "test")}
    for page, group in sorted(groups.items()):
        random.Random(SEED).shuffle(group)
        a, b = int(.7*len(group)), int(.85*len(group))
        for split, subset in zip(splits, (group[:a], group[a:b], group[b:])):
            splits[split].extend(subset)
    return splits


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]


def write_jsonl(path, rows):
    with Path(path).open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def prepare(extra_limit=4000):
    data = ROOT / "data"
    raw = data / "raw/1998-上半年-带音"
    segmenter = tokenizer()
    grouped = defaultdict(list)
    january_texts = set()
    source_hashes = {}
    for month in ("01", "02"):
        path = raw / f"1998-{month}-2003版-带音.txt"
        source_hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    for line in (raw / "1998-01-2003版-带音.txt").read_text(encoding="gb18030").splitlines():
        row = parse_line(line)
        if row:
            january_texts.add(digest(row["text"]))
            if row["page"] in PAGES:
                grouped[row["id"]].append(row)
    articles, seen = [], set()
    for aid, paragraphs in sorted(grouped.items()):
        paragraphs.sort(key=lambda x: x["paragraph"])
        text = "".join(p["text"] for p in paragraphs)
        tokens = tokenize(text, segmenter)
        fingerprint = digest("".join(tokens))
        if len(tokens) < 30 or fingerprint in seen:
            continue
        seen.add(fingerprint)
        articles.append({"id": aid, "page": paragraphs[0]["page"], "tokens": tokens,
                         "fingerprint": fingerprint})
    splits = split_articles(articles)
    # Remove exact paragraphs shared with held-out articles, even inside training articles.
    heldout = {r["id"] for s in ("dev", "test") for r in splits[s]}
    heldout_paragraphs = {digest(p["text"]) for aid in heldout for p in grouped[aid]}
    sentences = []
    for row in splits["train"]:
        for para in grouped[row["id"]]:
            if digest(para["text"]) not in heldout_paragraphs:
                tokens = tokenize(para["text"], segmenter)
                sentences.extend(tokens[i:i+128] for i in range(0, len(tokens), 128) if len(tokens[i:i+128]) >= 3)
    extras, extra_seen = [], set()
    for line in (raw / "1998-02-2003版-带音.txt").read_text(encoding="gb18030").splitlines():
        row = parse_line(line)
        if not row:
            continue
        h = digest(row["text"])
        if h in january_texts or h in extra_seen:
            continue
        extra_seen.add(h)
        tokens = tokenize(row["text"], segmenter)
        if len(tokens) >= 10:
            extras.append(tokens)
    random.Random(SEED).shuffle(extras)
    extras = extras[:extra_limit]
    for tokens in extras:
        sentences.extend(tokens[i:i+128] for i in range(0, len(tokens), 128) if len(tokens[i:i+128]) >= 3)
    for split, rows in splits.items():
        write_jsonl(data / f"{split}.jsonl", rows)
    write_jsonl(data / "representation.jsonl", sentences)
    stats = {"seed": SEED, "labels": PAGES, "minimum_article_tokens": 30,
             "split_counts": {s: dict(sorted(Counter(r['page'] for r in rows).items())) for s, rows in splits.items()},
             "split_tokens": {s: sum(len(r['tokens']) for r in rows) for s, rows in splits.items()},
             "representation_sentences": len(sentences), "representation_tokens": sum(map(len, sentences)),
             "february_paragraphs": len(extras), "source_sha256": source_hashes}
    save_json(data / "stats.json", stats)
    print(json.dumps(stats, ensure_ascii=False, indent=2), flush=True)
    return stats


if __name__ == "__main__":
    prepare()
