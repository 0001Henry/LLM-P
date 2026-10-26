"""Run all stages with captured logs and stop if a stage fails."""
import argparse
import subprocess
import sys
from pathlib import Path

from prepare import ROOT


def stage(name, args):
    with (ROOT / 'results' / f'{name}.log').open('w') as log:
        process = subprocess.Popen([sys.executable, '-u', *args], cwd=ROOT,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in process.stdout:
            print(line, end='', flush=True)
            log.write(line)
        if process.wait():
            raise RuntimeError(f'{name} failed; see results/{name}.log')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path, required=True)
    args = p.parse_args()
    (ROOT / 'results').mkdir(exist_ok=True)
    stage('prepare', ['prepare.py', '--source', str(args.source.resolve())])
    stage('experiment', ['experiment.py'])
    stage('tests', ['-m', 'unittest', '-v', 'test_hw1'])
    stage('report', ['make_report.py'])
