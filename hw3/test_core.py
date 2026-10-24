import unittest
from core import clean_line, encode_text, metrics, render, split_for, windows


class CoreTests(unittest.TestCase):
    def test_corpus(self):
        self.assertEqual(clean_line("19980101-01-001-007 他 在 上{shang5} 面 。"),
                         ("19980101-01-001", "他在上面。"))

    def test_labels(self):
        text, labels = encode_text("“学而时习之，不亦说乎？”")
        self.assertEqual(text, "学而时习之不亦说乎")
        self.assertEqual(render(text, labels), "学而时习之，不亦说乎？")

    def test_empty(self):
        self.assertEqual(encode_text("？！ \n"), ("", []))
        self.assertEqual(list(windows(0)), [])

    def test_window_coverage(self):
        for n in (1, 299, 300, 399, 400, 401, 510, 700, 701, 5000):
            spans = list(windows(n))
            covered = set()
            for s, e in spans:
                self.assertLessEqual(e-s, 400)
                covered.update(range(s, e))
            self.assertEqual(covered, set(range(n)))
            self.assertEqual(spans[-1][1], n)

    def test_invalid_windows(self):
        for size, stride in ((511, 300), (400, 0), (300, 400)):
            with self.assertRaises(ValueError):
                list(windows(1000, size, stride))

    def test_split(self):
        self.assertEqual(split_for("19980101-01-001"), split_for("19980101-01-001"))

    def test_metrics_ignore_true_negative(self):
        result = metrics([0]*100+[1, 2], [0]*100+[1, 1])
        self.assertEqual(result["punctuation_micro"]["f1"], 0.5)
        self.assertEqual(result["boundary"]["f1"], 1.0)


if __name__ == "__main__":
    unittest.main()
