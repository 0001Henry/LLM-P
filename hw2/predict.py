import argparse
import json

import joblib
from gensim.models import FastText
from sklearn.preprocessing import normalize

from data_utils import ROOT, tokenize, tokenizer
from experiment import average_vectors


def predict(text):
    selection = json.loads((ROOT / "models/selected.json").read_text())
    name = selection['method']
    model = FastText.load(str(ROOT / f"models/{name}.gensim"))
    classifier = joblib.load(ROOT / f"models/{name}-6.joblib")
    tokens = tokenize(text, tokenizer())
    if not tokens:
        raise ValueError("Input contains no usable Chinese or English words")
    vector = normalize(average_vectors(tokens, model.wv).reshape(1, -1))
    return {"page": str(classifier.predict(vector)[0]), "method": name,
            "probabilities": dict(zip(classifier.classes_.tolist(), classifier.predict_proba(vector)[0].tolist()))}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", required=True)
    args = parser.parse_args()
    print(json.dumps(predict(args.text), ensure_ascii=False, indent=2))
