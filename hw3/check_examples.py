"""Small, transparent out-of-domain sanity check, not a corpus benchmark."""
import json
from pathlib import Path
from bert_punct import Engine, save_json
from core import encode_text, metrics, render

EXAMPLES = [
    "学而时习之，不亦说乎？有朋自远方来，不亦乐乎？人不知而不愠，不亦君子乎？",
    "道可道，非常道。名可名，非常名。无名天地之始，有名万物之母。",
    "故天将降大任于是人也，必先苦其心志，劳其筋骨，饿其体肤，空乏其身，行拂乱其所为。",
    "先天下之忧而忧，后天下之乐而乐。",
    "山不在高，有仙则名。水不在深，有龙则灵。",
]


def main():
    engine = Engine("models/punctuation")
    records, gold, pred = [], [], []
    for reference in EXAMPLES:
        text, labels = encode_text(reference)
        output = engine.predict(text)
        records.append({"input": text, "reference": reference, "prediction": render(text, output)})
        gold.extend(labels)
        pred.extend(output)
    save_json("results/ancient_examples.json", {"note": "手工选取的五段古文及一种参考标点，非标准古文测试集，未参与训练和选模。",
                                               "examples": records, "metrics": metrics(gold, pred)})
    print(json.dumps(records, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
