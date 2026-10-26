# HW1：n-gram 建模训练

对应 L1 课件第 125 页。使用 KenLM 训练 2、3、5-gram，比较训练时间、留出文本困惑度和指定开头的续写。

- [实验报告](REPORT.md)
- [原始实验结果](results/metrics.json)
- [GitHub 目录](https://github.com/0001Henry/LLM-P/tree/main/hw1)

## 环境准备

下面命令在 `hw1/` 下执行，需要 Python 3.12、Conda、Git 和 C++ 编译器。macOS 使用 Command Line Tools。本次 Python 3.13 安装 KenLM 0.3.0 失败，使用 3.12 成功；不要直接使用不确定版本的 `python3` 创建环境。

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
conda create --prefix "$PWD/.cache/toolchain" --override-channels -c conda-forge 'libboost-devel=1.85.0' 'cmake=3.31.8' 'eigen=5.0.1' -y
git clone https://github.com/kpu/kenlm.git .cache/kenlm
git -C .cache/kenlm checkout 4cb443e60b7bf2c0ddf3c745378f76cb59e254e5
.cache/toolchain/bin/cmake -S .cache/kenlm -B .cache/kenlm/build -DCMAKE_PREFIX_PATH="$PWD/.cache/toolchain" -DCMAKE_BUILD_TYPE=Release -DENABLE_INTERPOLATE=OFF
.cache/toolchain/bin/cmake --build .cache/kenlm/build --target lmplz -j 4
```

`ENABLE_INTERPOLATE=OFF` 关闭独立的多模型插值工具，不关闭 `lmplz` 内部的 Kneser–Ney 插值平滑。Python 绑定为 PyPI 的 KenLM 0.3.0，训练程序来自上述固定源码提交，两者通过 ARPA 文件连接。系统包与 Python 环境都与其他作业隔离。编译依赖、源码、模型、环境及临时文件均不提交 GitHub。

## 准备数据并运行

原始数据来自课程提供的压缩包，不从网上另找语料。如果数据已解压，也可直接把文本路径传给 `--source`，无需复制。

```bash
mkdir -p data/raw
bsdtar -xf '../1998-上半年-带音(1).rar' -C data/raw '1998-上半年-带音/1998-01-2003版-带音.txt'
python run_experiment.py --source 'data/raw/1998-上半年-带音/1998-01-2003版-带音.txt'
```

完整入口依次清理语料、训练 15 次、评价模型、生成 12 组续写、运行测试并生成 Markdown 报告。任一阶段失败会停止。重新运行会覆盖模型和同名实验记录；需要比较多次实验时，先保留现有结果。

也可以分步执行：

```bash
python prepare.py --source 'data/raw/1998-上半年-带音/1998-01-2003版-带音.txt'
python experiment.py
python -m unittest -v test_hw1
python make_report.py
```

## 用模型续写

```bash
python generate.py --model models/3gram.arpa --temperature 0.7 --top-k 10
python generate.py --model models/5gram.arpa --temperature 1.3 --top-k 0 --top-p 0.95
python generate.py --model models/3gram.arpa --prefix-tokens '我们 学校 召开 了' --seed 42 --max-tokens 40
```

自定义前缀需按词用空格分开，与训练语料的分词方式尽量一致。默认使用课件指定开头。输出包含续写、实际词序列、是否遇到句尾标记、耗时及前缀未登录词。`temperature=0` 为贪心选择，`top-k=0` 为不做 top-k 截断。

## 文件说明

| 文件 | 用途 |
| --- | --- |
| `prepare.py` | 删除编号、注音和标签；保留词边界；切句、去重、划分 |
| `experiment.py` | 调用 KenLM，重复计时、评估和生成 |
| `generate.py` | 用状态接口逐词打分并采样 |
| `test_hw1.py` | 数据隔离、清理、采样和模型接口测试 |
| `run_experiment.py` | 完整入口与日志保存 |
| `make_report.py` | 从实测 JSON 生成报告 |
| `data/stats.json` | 数据规模、未登录词数量、原文件 SHA256 |
| `results/metrics.json` | 环境、工具校验值、训练时间、困惑度和全部生成 |
| `results/*-run*.log` | KenLM 各次训练的原始阶段日志 |
| `models/*.arpa` | 本地训练的模型 |

仅运行 `make_report.py` 不会训练模型。报告没有“关键词”栏；修改文字应编辑生成脚本，避免重新生成时丢失修改。
