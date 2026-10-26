"""Sample words using KenLM scores, excluding control tokens from output."""
import argparse
import json
import time
from pathlib import Path

import kenlm
import numpy as np

ROOT = Path(__file__).resolve().parent
PREFIX = '在阳光明媚的五月，我们学校胜利召开了'
PREFIX_TOKENS = ['在', '阳光明媚', '的', '五月', ',', '我们', '学校', '胜利', '召开', '了']


def vocabulary(arpa):
    words, active = [], False
    with arpa.open(encoding='utf-8') as stream:
        for line in stream:
            if line.strip() == '\\1-grams:':
                active = True
                continue
            if active and line.startswith('\\'):
                break
            if active and line.strip():
                words.append(line.split()[1])
    return sorted(w for w in words if w not in {'<s>', '<unk>'})


def choose(log10_scores, temperature, top_k, top_p, rng):
    if not np.isfinite(temperature) or temperature < 0 or top_k < 0 or not 0 < top_p <= 1:
        raise ValueError('temperature >= 0, top_k >= 0, 0 < top_p <= 1 required')
    if temperature == 0:
        return int(np.argmax(log10_scores))
    order = np.argsort(-log10_scores, kind='stable')
    if top_k:
        order = order[:top_k]
    logits = log10_scores[order] * np.log(10) / temperature
    probs = np.exp(logits - logits.max())
    probs /= probs.sum()
    if top_p < 1:
        end = min(len(probs), int(np.searchsorted(np.cumsum(probs), top_p)) + 1)
        order, probs = order[:end], probs[:end]
        probs /= probs.sum()
    return int(rng.choice(order, p=probs))


def generate(model, words, prefix_tokens, temperature=1, top_k=50, top_p=1, seed=42, max_tokens=40):
    if not words or max_tokens < 1:
        raise ValueError('nonempty vocabulary and positive max_tokens required')
    state, scratch = kenlm.State(), kenlm.State()
    model.BeginSentenceWrite(state)
    for word in prefix_tokens:
        model.BaseScore(state, word, scratch)
        state, scratch = scratch, state
    rng, output, stopped = np.random.default_rng(seed), [], False
    start = time.perf_counter()
    for _ in range(max_tokens):
        scores = np.fromiter((model.BaseScore(state, w, scratch) for w in words), dtype=float, count=len(words))
        word = words[choose(scores, temperature, top_k, top_p, rng)]
        if word == '</s>':
            stopped = True
            break
        output.append(word)
        model.BaseScore(state, word, scratch)
        state, scratch = scratch, state
    return {'tokens': output, 'continuation': ''.join(output), 'stopped_at_eos': stopped,
            'seconds': time.perf_counter() - start,
            'prefix_oov': [w for w in prefix_tokens if w not in model]}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Supply a space-tokenized prefix for custom text.')
    p.add_argument('--model', type=Path, default=ROOT / 'models/3gram.arpa')
    p.add_argument('--prefix-tokens', default=' '.join(PREFIX_TOKENS))
    p.add_argument('--temperature', type=float, default=1)
    p.add_argument('--top-k', type=int, default=50)
    p.add_argument('--top-p', type=float, default=1)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--max-tokens', type=int, default=40)
    a = p.parse_args()
    result = generate(kenlm.Model(str(a.model)), vocabulary(a.model), a.prefix_tokens.split(),
                      a.temperature, a.top_k, a.top_p, a.seed, a.max_tokens)
    print(json.dumps(result, ensure_ascii=False, indent=2))
