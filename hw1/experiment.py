"""Train KenLM models, measure elapsed time, score held-out text and sample."""
import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import random
import re
import statistics
import subprocess
import time
from pathlib import Path

import kenlm

from generate import PREFIX, PREFIX_TOKENS, generate, vocabulary
from prepare import ROOT, save_json


def evaluate(model, path):
    total_log10, total, oov, sentences = 0.0, 0, 0, 0
    matched = {}
    for line in path.read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        sentences += 1
        for score, order, unknown in model.full_scores(line, bos=True, eos=True):
            total_log10 += score
            total += 1
            oov += int(unknown)
            matched[str(order)] = matched.get(str(order), 0) + 1
    return {'sentences': sentences, 'predicted_tokens_including_eos': total,
            'log10_probability': total_log10, 'perplexity': 10 ** (-total_log10 / total),
            'oov_tokens': oov, 'matched_order_counts': matched}


def run(lmplz):
    results, models = ROOT / 'results', ROOT / 'models'
    results.mkdir(exist_ok=True)
    models.mkdir(exist_ok=True)
    temp = ROOT / '.cache/tmp'
    temp.mkdir(parents=True, exist_ok=True)
    specs = [(f'{n}gram', n, 'train.txt') for n in (2, 3, 5)]
    specs += [(f'3gram-{p}', 3, f'train{p}.txt') for p in (25, 50)]
    plan = [(name, n, text, repeat) for name, n, text in specs for repeat in range(1, 4)]
    random.Random(42).shuffle(plan)
    rows, sequence = {}, []
    for name, order, text, repeat in plan:
        arpa = models / f'{name}.arpa'
        cmd = [str(lmplz), '-o', str(order), '-S', '256M', '-T', str(temp),
               '--text', str(ROOT / 'data' / text), '--arpa', str(arpa)]
        print(f'Train {name}, repeat {repeat}', flush=True)
        start = time.perf_counter()
        with (results / f'{name}-run{repeat}.log').open('w') as log:
            subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True)
        elapsed = time.perf_counter() - start
        counts = {}
        with arpa.open() as f:
            for line in f:
                if line.startswith('\\1-grams:'):
                    break
                match = re.match(r'ngram (\d+)=(\d+)', line)
                if match:
                    counts[match[1]] = int(match[2])
        row = rows.setdefault(name, {'name': name, 'order': order, 'training_file': text, 'runs': []})
        row['runs'].append({'repeat': repeat, 'seconds': elapsed})
        row.update({'ngram_counts': counts, 'arpa_bytes': arpa.stat().st_size,
                    'arpa_sha256': hashlib.sha256(arpa.read_bytes()).hexdigest()})
        sequence.append({'name': name, 'repeat': repeat})
        print(f'  {elapsed:.3f} s, {sum(counts.values()):,} n-grams', flush=True)
    for row in rows.values():
        row['runs'].sort(key=lambda r: r['repeat'])
        row['median_seconds'] = statistics.median(r['seconds'] for r in row['runs'])
        text = (ROOT / 'data' / row['training_file']).read_text()
        row['training_sentences'] = len(text.splitlines())
        row['training_tokens'] = len(text.split())
    outputs = []
    settings = [('greedy', 0, 0, 1), ('low-temperature', .7, 10, 1),
                ('top-k', 1., 50, 1), ('top-p', 1.3, 0, .95)]
    for order in (2, 3, 5):
        name = f'{order}gram'
        arpa = models / f'{name}.arpa'
        model = kenlm.Model(str(arpa))
        rows[name]['evaluation'] = {s: evaluate(model, ROOT / 'data' / f'{s}.txt') for s in ('dev', 'test')}
        words = vocabulary(arpa)
        for label, temperature, top_k, top_p in settings:
            print(f'Generate {name}: {label}', flush=True)
            output = generate(model, words, PREFIX_TOKENS, temperature, top_k, top_p)
            output.update({'order': order, 'setting': label, 'temperature': temperature,
                           'top_k': top_k, 'top_p': top_p, 'seed': 42, 'max_tokens': 40})
            outputs.append(output)
    metrics = {'environment': {'python': platform.python_version(), 'platform': platform.platform(),
                               'kenlm_python': importlib.metadata.version('kenlm'),
                               'numpy': importlib.metadata.version('numpy'),
                               'kenlm_source_commit': subprocess.check_output(
                                   ['git', '-C', str(ROOT / '.cache/kenlm'), 'rev-parse', 'HEAD'], text=True).strip(),
                               'lmplz_sha256': hashlib.sha256(lmplz.read_bytes()).hexdigest()},
               'memory_budget': '256M', 'repeats': 3, 'run_sequence': sequence,
               'models': [rows[name] for name, _, _ in specs],
               'prefix': PREFIX, 'prefix_tokens': PREFIX_TOKENS, 'generations': outputs}
    assert all(math.isfinite(r['median_seconds']) for r in rows.values())
    save_json(results / 'metrics.json', metrics)
    print('Saved results/metrics.json', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--lmplz', type=Path, default=ROOT / '.cache/kenlm/build/bin/lmplz')
    run(p.parse_args().lmplz.resolve())
