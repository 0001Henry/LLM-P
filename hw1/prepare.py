"""Clean the supplied segmented corpus and split it by article."""
import argparse
import hashlib
import json
import random
import re
import unicodedata
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SEED = 42


def save_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def parse_line(line):
    match = re.match(r'^(\d{8}-\d{2}-\d{3})-\d+\s+(.+)$', line.strip())
    if not match:
        return None
    aid, text = match.groups()
    text = re.sub(r'\{[^}]*\}', '', text)
    tokens = []
    for token in text.split():
        # Also accept the POS/bracket notation used in other People's Daily editions.
        token = re.sub(r'\][A-Za-z]+$', '', token).lstrip('[')
        token = re.sub(r'/[A-Za-z]+$', '', token)
        token = unicodedata.normalize('NFKC', token)
        if token and token not in {'<s>', '</s>', '<unk>', '〓'}:
            tokens.append(token)
    return aid, tokens


def sentences(tokens):
    sentence = []
    for token in tokens:
        sentence.append(token)
        if token in {'。', '!', '?', '！', '？'}:
            if len(sentence) >= 3:
                yield sentence
            sentence = []
    if len(sentence) >= 3:
        yield sentence


def split_name(aid):
    bucket = int(hashlib.sha256(aid.encode()).hexdigest()[:8], 16) % 100
    return 'train' if bucket < 80 else 'dev' if bucket < 90 else 'test'


def prepare(source):
    groups = {s: [] for s in ('train', 'dev', 'test')}
    seen, duplicates, annotations, parsed, layout_marks = set(), 0, 0, 0, 0
    for line in source.read_text(encoding='gb18030').splitlines():
        row = parse_line(line)
        if row is None:
            continue
        parsed += 1
        annotations += len(re.findall(r'\{[^}]*\}', line))
        layout_marks += line.split().count('〓')
        aid, tokens = row
        for sentence in sentences(tokens):
            key = ''.join(sentence)
            if key in seen:
                duplicates += 1
                continue
            seen.add(key)
            groups[split_name(aid)].append({'article': aid, 'tokens': sentence})
    data = ROOT / 'data'
    data.mkdir(exist_ok=True)
    for name, rows in groups.items():
        random.Random(SEED).shuffle(rows)
        with (data / f'{name}.jsonl').open('w', encoding='utf-8') as out:
            for row in rows:
                out.write(json.dumps(row, ensure_ascii=False) + '\n')
        (data / f'{name}.txt').write_text(''.join(' '.join(r['tokens']) + '\n' for r in rows), encoding='utf-8')
    for percent in (25, 50):
        rows = groups['train'][:len(groups['train']) * percent // 100]
        (data / f'train{percent}.txt').write_text(''.join(' '.join(r['tokens']) + '\n' for r in rows), encoding='utf-8')
    vocab = Counter(w for r in groups['train'] for w in r['tokens'])
    stats = {'source': source.name, 'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
             'seed': SEED, 'parsed_paragraphs': parsed, 'removed_pronunciations': annotations,
             'removed_layout_marks': layout_marks,
             'duplicate_sentences_removed': duplicates, 'vocabulary_size': len(vocab),
             'splits': {s: {'articles': len({r['article'] for r in rows}), 'sentences': len(rows),
                            'tokens': sum(len(r['tokens']) for r in rows),
                            'oov_tokens': sum(w not in vocab for r in rows for w in r['tokens'])}
                        for s, rows in groups.items()}}
    save_json(data / 'stats.json', stats)
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return stats


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    prepare(parser.parse_args().source)
