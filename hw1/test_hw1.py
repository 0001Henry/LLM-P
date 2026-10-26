import json
import unittest

import kenlm
import numpy as np

from generate import choose, generate, vocabulary
from prepare import ROOT, parse_line, sentences, split_name


class PreparationTests(unittest.TestCase):
    def test_annotation_removal(self):
        aid, words = parse_line('19980101-01-001-002 在{zai4}/p [北京/ns 大学/n]nt 。/w')
        self.assertEqual(aid, '19980101-01-001')
        self.assertEqual(words, ['在', '北京', '大学', '。'])

    def test_invalid_line(self):
        self.assertIsNone(parse_line('metadata'))

    def test_normalization(self):
        self.assertEqual(parse_line('19980101-01-001-002 １９９８ ， 〓 <s> <unk>')[1], ['1998', ','])

    def test_sentences(self):
        self.assertEqual(list(sentences(['甲', '乙', '。', '丙', '丁', '!'])),
                         [['甲', '乙', '。'], ['丙', '丁', '!']])

    def test_split_stable(self):
        self.assertEqual(split_name('19980101-01-001'), split_name('19980101-01-001'))

    def test_corpus_isolation(self):
        paths = [ROOT / 'data' / f'{s}.jsonl' for s in ('train', 'dev', 'test')]
        if not all(p.exists() for p in paths):
            self.skipTest('Prepare corpus first')
        seen_ids, seen_text = set(), set()
        for path in paths:
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            ids = {r['article'] for r in rows}
            texts = {''.join(r['tokens']) for r in rows}
            self.assertFalse(ids & seen_ids)
            self.assertFalse(texts & seen_text)
            self.assertEqual(len(texts), len(rows))
            self.assertTrue(all('{' not in w and '}' not in w for r in rows for w in r['tokens']))
            seen_ids |= ids
            seen_text |= texts


class SamplingTests(unittest.TestCase):
    def test_greedy(self):
        self.assertEqual(choose(np.array([-2., -1., -3.]), 0, 0, 1, np.random.default_rng(42)), 1)

    def test_top_k(self):
        for seed in range(20):
            self.assertEqual(choose(np.array([-2., -1.]), 1, 1, 1, np.random.default_rng(seed)), 1)

    def test_top_p(self):
        self.assertEqual(choose(np.array([-5., 0]), 1, 0, .5, np.random.default_rng(42)), 1)

    def test_invalid_parameters(self):
        for t, k, p in [(-1, 0, 1), (1, -1, 1), (1, 0, 0), (1, 0, 2)]:
            with self.assertRaises(ValueError):
                choose(np.array([0.]), t, k, p, np.random.default_rng())

    def test_model_reload_and_states(self):
        path = ROOT / 'models/3gram.arpa'
        if not path.exists():
            self.skipTest('Train model first')
        model = kenlm.Model(str(path))
        tokens = ['我们', '学校', '召开', '了']
        state, out = kenlm.State(), kenlm.State()
        model.BeginSentenceWrite(state)
        score = 0
        for w in tokens + ['</s>']:
            score += model.BaseScore(state, w, out)
            state, out = out, state
        self.assertAlmostEqual(score, model.score(' '.join(tokens)), places=4)
        words = vocabulary(path)
        self.assertNotIn('<unk>', words)
        self.assertIn('</s>', words)
        total = sum(10 ** model.BaseScore(state, w, out) for w in words + ['<unk>'])
        self.assertAlmostEqual(total, 1.0, places=4)
        first = generate(model, words, tokens, seed=42, max_tokens=4)
        second = generate(model, words, tokens, seed=42, max_tokens=4)
        self.assertEqual(first['tokens'], second['tokens'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
