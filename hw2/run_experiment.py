"""Run all stages with recorded logs and a deterministic hash seed."""
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    (ROOT / 'results').mkdir(exist_ok=True)
    env = dict(os.environ, PYTHONHASHSEED='42', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
               MPLCONFIGDIR=str(ROOT / '.cache/matplotlib'), XDG_CACHE_HOME=str(ROOT / '.cache'))
    stages = [('dependency_fix', ['fix_gensim_blas.py']),
              ('prepare', ['data_utils.py']), ('tests', ['-m', 'unittest', 'discover', '-v']),
              ('experiment', ['experiment.py']), ('report', ['make_report.py']),
              ('prediction', ['predict.py', '--text', '国际社会关注各国经济合作与贸易发展'])]
    for name, args in stages:
        print(f'Running {name}', flush=True)
        numerical_issue = False
        with (ROOT / f'results/{name}.log').open('w', encoding='utf-8') as log:
            proc = subprocess.Popen([sys.executable, '-u', *args], cwd=ROOT, env=env,
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            for line in proc.stdout:
                print(line, end='', flush=True)
                log.write(line)
                log.flush()
                numerical_issue |= any(x in line for x in ('Exception ignored in:', 'RuntimeWarning:', 'ConvergenceWarning:'))
            if proc.wait() or numerical_issue:
                raise SystemExit(f'{name} failed; see results/{name}.log')
    print('All stages completed.', flush=True)


if __name__ == '__main__':
    main()
