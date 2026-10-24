"""Corpus preparation and window algorithms, independent of ML dependencies."""
import hashlib
import re
import unicodedata

LABELS = ["O", "，", "。", "？", "！", "；", "：", "、"]
PUNCT = {p: i for i, p in enumerate(LABELS) if i}
NORMALIZE = {",": "，", ".": "。", "?": "？", "!": "！", ";": "；", ":": "："}


def clean_line(line):
    match = re.match(r"^(\d{8}-\d{2}-\d{3})-\d+\s+(.*)$", line.strip())
    if not match:
        return None
    article, text = match.groups()
    text = re.sub(r"\{[^}]*\}", "", text)
    return article, "".join(text.split())


def encode_text(text):
    chars, labels = [], []
    for char in text:
        punct = NORMALIZE.get(char, char)
        if punct in PUNCT:
            if labels:
                labels[-1] = PUNCT[punct]
        elif char.isspace() or unicodedata.category(char).startswith("P"):
            continue
        else:
            chars.append(char)
            labels.append(0)
    return "".join(chars), labels


def split_for(article):
    value = int(hashlib.sha256(article.encode()).hexdigest()[:8], 16) % 100
    return "train" if value < 80 else "dev" if value < 90 else "test"


def windows(length, size=400, stride=300):
    if not 0 < stride <= size <= 510:
        raise ValueError("Require 0 < stride <= size <= 510")
    start = 0
    while start < length:
        end = min(start + size, length)
        yield start, end
        if end == length:
            break
        start += stride


def render(text, labels):
    if len(text) != len(labels):
        raise ValueError("Text and labels must align")
    return "".join(c + (LABELS[int(y)] if y else "") for c, y in zip(text, labels))


def metrics(gold, pred):
    if len(gold) != len(pred):
        raise ValueError("Gold and prediction must align")
    def prf(tp, fp, fn):
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        return {"precision": p, "recall": r, "f1": 2*p*r/(p+r) if p+r else 0.0}
    per = {}
    for i, name in enumerate(LABELS[1:], 1):
        per[name] = prf(sum(g == p == i for g, p in zip(gold, pred)),
                        sum(g != i and p == i for g, p in zip(gold, pred)),
                        sum(g == i and p != i for g, p in zip(gold, pred)))
    exact = prf(sum(g == p and g != 0 for g, p in zip(gold, pred)),
                sum(p != 0 and g != p for g, p in zip(gold, pred)),
                sum(g != 0 and g != p for g, p in zip(gold, pred)))
    boundary = prf(sum(g != 0 and p != 0 for g, p in zip(gold, pred)),
                   sum(g == 0 and p != 0 for g, p in zip(gold, pred)),
                   sum(g != 0 and p == 0 for g, p in zip(gold, pred)))
    return {"punctuation_micro": exact, "boundary": boundary, "per_label": per,
            "characters": len(gold), "accuracy": sum(g == p for g, p in zip(gold, pred))/max(1, len(gold))}
