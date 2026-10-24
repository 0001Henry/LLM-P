import unittest
from collections import Counter

import numpy as np
from gensim.models import FastText, Word2Vec

from data_utils import ROOT, digest, parse_line, read_jsonl, split_articles, tokenize, tokenizer
from experiment import average_vectors, evaluate


class DataTests(unittest.TestCase):
    def test_metadata_removed(self):
        r = parse_line("19980101-06-001-002 中国 在 上{shang5} 海 。")
        self.assertEqual(r['text'], "中国在上海。")
        self.assertEqual(r['id'], "19980101-06-001")
        self.assertEqual(r['page'], "06")

    def test_invalid_line(self):
        self.assertIsNone(parse_line("这不是语料记录"))

    def test_split_reproducible(self):
        rows = [{"id": str(i), "page": str(i % 3)} for i in range(60)]
        a, b = split_articles(rows), split_articles(rows)
        self.assertEqual(a, b)
        self.assertEqual(sum(len(x) for x in a.values()), len(rows))
        self.assertFalse({r['id'] for r in a['train']} & {r['id'] for r in a['test']})

    def test_tokenization(self):
        self.assertEqual(tokenize("１２３！ ABC", tokenizer()), ["abc"])
        self.assertEqual(tokenize("！？１２３", tokenizer()), [])

    def test_real_split_integrity(self):
        splits = {s: read_jsonl(ROOT / f"data/{s}.jsonl") for s in ('train', 'dev', 'test')}
        for a, b in (('train', 'dev'), ('train', 'test'), ('dev', 'test')):
            self.assertFalse({r['id'] for r in splits[a]} & {r['id'] for r in splits[b]})
            self.assertFalse({r['fingerprint'] for r in splits[a]} & {r['fingerprint'] for r in splits[b]})
        for rows in splits.values():
            self.assertEqual(len(Counter(r['page'] for r in rows)), 6)
            self.assertTrue(all(len(r['tokens']) >= 30 for r in rows))
            self.assertTrue(all(r['id'] not in r['tokens'] for r in rows))


class ModelTests(unittest.TestCase):
    def test_negative_one_dot_regression(self):
        from fix_gensim_blas import dot_regression
        self.assertEqual(dot_regression(), -1.0)

    @classmethod
    def setUpClass(cls):
        corpus = [["中国", "经济", "发展"], ["中国", "教育", "发展"]]*10
        cls.ft = FastText(corpus, vector_size=8, min_count=1, min_n=2, max_n=4,
                          bucket=100, workers=1, epochs=2, seed=42)
        cls.w2v = Word2Vec(corpus, vector_size=8, min_count=1, workers=1, epochs=2, seed=42)

    def test_fasttext_oov(self):
        self.assertNotIn("中国经济", self.ft.wv.key_to_index)
        v = self.ft.wv['中国经济']
        self.assertEqual(v.shape, (8,))
        self.assertTrue(np.isfinite(v).all())

    def test_word2vec_oov_and_empty(self):
        self.assertNotIn("中国经济", self.w2v.wv)
        np.testing.assert_array_equal(average_vectors([], self.w2v.wv), np.zeros(8))
        np.testing.assert_array_equal(average_vectors(['未知'], self.w2v.wv), np.zeros(8))

    def test_mean_vector(self):
        np.testing.assert_allclose(average_vectors(['中国', '经济'], self.ft.wv),
                                   (self.ft.wv['中国']+self.ft.wv['经济'])/2)

    def test_metrics(self):
        m = evaluate(['01', '02'], ['01', '01'], ['01', '02'])
        self.assertEqual(m['accuracy'], .5)
        self.assertEqual(m['confusion_matrix'], [[1, 0], [1, 0]])


if __name__ == '__main__':
    unittest.main()
