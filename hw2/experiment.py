"""Gensim representations with identical downstream classification protocols."""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import argparse
import hashlib
import inspect
import json
import logging
import platform
import time
import warnings
from pathlib import Path

import gensim
import joblib
import numpy as np
import sklearn
import scipy
from threadpoolctl import threadpool_info
from gensim import corpora, matutils
from gensim.models import FastText, LsiModel, TfidfModel, Word2Vec
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import normalize

from data_utils import PAGES, ROOT, SEED, read_jsonl, save_json


def average_vectors(tokens, keyed):
    vectors = [keyed[w] for w in tokens if w in keyed]
    return np.mean(vectors, axis=0) if vectors else np.zeros(keyed.vector_size, dtype=np.float32)


def evaluate(y, pred, labels):
    return {"accuracy": float(accuracy_score(y, pred)), "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
            "weighted_f1": float(f1_score(y, pred, average="weighted", zero_division=0)),
            "per_class": classification_report(y, pred, labels=labels, output_dict=True, zero_division=0),
            "confusion_matrix": confusion_matrix(y, pred, labels=labels).tolist()}


def sparse(corpus, width):
    return matutils.corpus2csc(corpus, num_terms=width).T.tocsr()


def run():
    warnings.filterwarnings('error', category=RuntimeWarning)
    models = ROOT / "models"
    models.mkdir(exist_ok=True)
    (ROOT / "results").mkdir(exist_ok=True)
    logging.basicConfig(filename=ROOT / "results/gensim.log", level=logging.INFO, filemode="w")
    np.random.seed(SEED)
    splits = {s: read_jsonl(ROOT / f"data/{s}.jsonl") for s in ("train", "dev", "test")}
    sentences = read_jsonl(ROOT / "data/representation.jsonl")
    features, timings = {}, {}

    started = time.perf_counter()
    dictionary = corpora.Dictionary(sentences)
    dictionary.filter_extremes(no_below=2, no_above=.95, keep_n=20000)
    dictionary.save(str(models / "dictionary.gensim"))
    bows = {s: [dictionary.doc2bow(r["tokens"]) for r in rows] for s, rows in splits.items()}
    representation_bow = [dictionary.doc2bow(tokens) for tokens in sentences]
    tfidf = TfidfModel(representation_bow, dictionary=dictionary)
    tfidf.save(str(models / "tfidf.gensim"))
    tfidfs = {s: list(tfidf[b]) for s, b in bows.items()}
    features["BoW"] = {s: normalize(sparse(b, len(dictionary))) for s, b in bows.items()}
    features["TF-IDF"] = {s: sparse(b, len(dictionary)) for s, b in tfidfs.items()}
    timings["BoW/TF-IDF"] = time.perf_counter()-started
    started = time.perf_counter()
    lsi = LsiModel(tfidf[representation_bow], id2word=dictionary, num_topics=64, random_seed=SEED)
    lsi.save(str(models / "lsi.gensim"))
    features["LSI"] = {s: normalize(sparse(list(lsi[b]), lsi.num_topics)) for s, b in tfidfs.items()}
    timings["LSI"] = time.perf_counter()-started

    embedding_models, diagnostics = {}, {}
    common = dict(vector_size=64, window=5, min_count=3, sg=1, negative=5, hs=0,
                  epochs=5, workers=1, seed=SEED, sample=1e-3)
    variants = [("Word2Vec", Word2Vec, {}),
                ("FastText-2-4", FastText, dict(min_n=2, max_n=4, bucket=50000)),
                ("FastText-3-6", FastText, dict(min_n=3, max_n=6, bucket=50000))]
    for name, model_class, extra in variants:
        print(f"Training {name}", flush=True)
        started = time.perf_counter()
        model = model_class(**common, **extra)
        model.build_vocab(sentences)
        initial = model.wv.vectors[:256].copy()
        model.train(sentences, total_examples=len(sentences), epochs=common['epochs'])
        change = float(np.linalg.norm(model.wv.vectors[:256]-initial))
        if not np.isfinite(model.wv.vectors).all() or change == 0:
            raise RuntimeError(f"{name}: nonfinite or unchanged trained vectors")
        training_seconds = time.perf_counter()-started
        model.save(str(models / f"{name}.gensim"))
        embedding_models[name] = model
        started = time.perf_counter()
        features[name] = {s: normalize(np.stack([average_vectors(r["tokens"], model.wv) for r in rows]))
                          for s, rows in splits.items()}
        timings[name] = {"train_seconds": training_seconds, "vectorize_seconds": time.perf_counter()-started}
        test_tokens = [w for r in splits["test"] for w in r["tokens"]]
        oov = [w for w in test_tokens if w not in model.wv.key_to_index]
        queries = [w for w in ("中国", "经济", "科学", "教育", "市场", "足球") if w in model.wv.key_to_index]
        probes = [w for w in ("数字化教育平台", "跨区域经济协作", "智能新闻分类器") if w not in model.wv.key_to_index]
        diagnostics[name] = {"vocabulary": len(model.wv), "first_256_vector_change_norm": change, "test_token_count": len(test_tokens),
                             "test_oov_count": len(oov), "test_oov_rate": len(oov)/len(test_tokens),
                             "test_oov_representable": sum(w in model.wv for w in oov),
                             "neighbors": {q: model.wv.most_similar(q, topn=5) for q in queries},
                             "oov_probes": {w: {"available": w in model.wv,
                                 "norm": float(np.linalg.norm(model.wv[w])) if w in model.wv else None} for w in probes}}
        print(name, timings[name], flush=True)

    results = []
    for n in (2, 4, 6):
        labels = PAGES[:n]
        indexes = {s: np.array([i for i, r in enumerate(rows) if r["page"] in labels]) for s, rows in splits.items()}
        ys = {s: np.array([splits[s][i]["page"] for i in idx]) for s, idx in indexes.items()}
        majority = DummyClassifier(strategy="most_frequent").fit(np.zeros((len(ys['train']), 1)), ys['train'])
        results.append({"classes": n, "method": "Majority", "test": evaluate(ys['test'], majority.predict(np.zeros((len(ys['test']), 1))), labels)})
        for name, values in features.items():
            x = {s: values[s][idx] for s, idx in indexes.items()}
            tuning, best, best_score = [], None, -1
            started = time.perf_counter()
            for c in (.1, 1., 10.):
                classifier = LogisticRegression(C=c, class_weight="balanced", max_iter=1000, random_state=SEED)
                classifier.fit(x['train'], ys['train'])
                dev = evaluate(ys['dev'], classifier.predict(x['dev']), labels)
                tuning.append({"C": c, "dev_macro_f1": dev['macro_f1'], "iterations": classifier.n_iter_.tolist()})
                if dev['macro_f1'] > best_score:
                    best, best_score = classifier, dev['macro_f1']
            fit_seconds = time.perf_counter()-started
            started = time.perf_counter()
            predictions = best.predict(x['test'])
            predict_seconds = time.perf_counter()-started
            result = {"classes": n, "method": name, "selected_C": best.C, "dev_macro_f1": best_score,
                      "test": evaluate(ys['test'], predictions, labels), "tuning": tuning,
                      "fit_search_seconds": fit_seconds, "test_predict_seconds": predict_seconds,
                      "counts": {s: len(y) for s, y in ys.items()},
                      "errors": [{"id": splits['test'][int(i)]['id'], "gold": str(gold), "prediction": str(pred)}
                                 for i, gold, pred in zip(indexes['test'], ys['test'], predictions) if gold != pred][:20]}
            results.append(result)
            joblib.dump(best, models / f"{name}-{n}.joblib")
            print(f"classes={n} method={name} dev={best_score:.4f} test={result['test']['macro_f1']:.4f}", flush=True)
    selected = max((r for r in results if r['classes'] == 6 and r['method'].startswith("FastText")), key=lambda r: r['dev_macro_f1'])
    save_json(models / "selected.json", {"method": selected['method'], "classes": PAGES})
    source = {}
    for cls, methods in ((FastText, ["build_vocab", "_do_train_job"]),
                         (type(embedding_models['FastText-2-4'].wv), ["get_vector", "adjust_vectors"]),
                         (Word2Vec, ["prepare_vocab", "train"])):
        for method in methods:
            fn = getattr(cls, method)
            path = Path(inspect.getsourcefile(fn))
            source[f"{cls.__name__}.{method}"] = {"file": str(path.relative_to(Path(gensim.__file__).parent.parent)),
                "line": inspect.getsourcelines(fn)[1], "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    save_json(ROOT / "results/metrics.json", {"configuration": common, "variants": {n: e for n, _, e in variants},
        "representation_times": timings, "dictionary_size": len(dictionary), "experiments": results,
        "selected_fasttext": selected['method'], "diagnostics": diagnostics, "source_locations": source,
        "environment": {"python": platform.python_version(), "platform": platform.platform(),
                        "gensim": gensim.__version__, "sklearn": sklearn.__version__, "numpy": np.__version__, "scipy": scipy.__version__,
                        "blas": [{k: v for k, v in info.items() if k != 'filepath'} for info in threadpool_info()]}})


if __name__ == "__main__":
    run()
