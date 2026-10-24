"""Run and record the complete bounded homework experiment."""
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path


def run(name, *args):
    command = [sys.executable, "-u", *args]
    print("RUN:", " ".join(command), flush=True)
    with Path(f"results/{name}.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, encoding="utf-8")
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        code = process.wait()
    if code:
        raise SystemExit(f"{name} failed with exit code {code}; see results/{name}.log")


def main():
    Path("results").mkdir(exist_ok=True)
    run("unit_tests", "-m", "unittest", "discover", "-v")
    run("train", "bert_punct.py", "train", "--train-limit", "1000", "--eval-limit", "100",
        "--epochs", "3", "--freeze-layers", "10", "--batch", "8")
    run("benchmark", "bert_punct.py", "benchmark", "--model", "models/punctuation",
        "--eval-limit", "100", "--chars", "4000", "--batch", "8", "--repeats", "3")
    run("ancient_examples", "check_examples.py")
    run("prediction", "bert_punct.py", "predict", "--model", "models/punctuation")
    import torch
    import transformers
    weights = Path("models/bert-base-chinese/model.safetensors")
    with weights.open("rb") as f:
        digest = hashlib.file_digest(f, "sha256").hexdigest()
    env = {"platform": platform.platform(), "python": sys.version, "torch": torch.__version__,
           "transformers": transformers.__version__, "pretrained_sha256": digest}
    Path("results/environment.json").write_text(json.dumps(env, indent=2), encoding="utf-8")
    run("report", "make_report.py")


if __name__ == "__main__":
    main()
