# HW2：Gensim 表示学习与 FastText 文本分类

对应 L2.part1 第 137 页。正式提交文档见 [REPORT.md](REPORT.md)。

## 实验内容

- 表示方法：Gensim BoW、TF-IDF、LSI、Word2Vec、FastText（字符 n-gram 2–4 和 3–6）。
- 分类：人民日报一月文章的版面号 01–06，依次完成 2、4、6 类实验。不是人工主题标签。
- 下游分类器：逻辑回归，验证集选择 C；另有多数类基线。
- FastText 是无监督子词表示，均值池化后接分类器，不冒充原生监督 FastText 实现。
- 额外表示语料：二月 4,000 个新闻段落。所有表示方法共享无标签语料，不在验证/测试集拟合表示。

## 运行

在 `hw2` 目录执行：

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
mkdir -p data/raw
bsdtar -xf '../1998-上半年-带音(1).rar' -C data/raw '1998-上半年-带音/1998-01-2003版-带音.txt' '1998-上半年-带音/1998-02-2003版-带音.txt'
```

`requirements-lock.txt` 记录实跑依赖。完整入口先准备数据再执行测试，随后训练表示及分类器、评估、生成报告并重载模型预测。它会设置 `PYTHONHASHSEED=42` 与单线程环境。只需更新报告时执行 `python make_report.py`。

### Apple Silicon 数值库

本次机器默认的 Accelerate 构建出现 BLAS 提示，因此使用同版本的 OpenBLAS wheel。Apple Silicon 上在运行实验前执行以下命令；其他平台无需使用这些平台选项：

```bash
python -m pip download --only-binary=:all: --platform macosx_12_0_arm64 --dest .cache/wheels 'numpy==2.2.6' 'scipy==1.15.3'
python -m pip install --force-reinstall --no-index --find-links .cache/wheels 'numpy==2.2.6' 'scipy==1.15.3'
```

入口会检查日志中的数值异常及收敛警告，遇到这些问题时不会继续生成成功报告。

### Gensim 4.4.0 点积兼容修复

另已定位到 Gensim 的 Cython BLAS 包装将合法的点积 `-1` 误判为异常并返回 `0`。`fix_gensim_blas.py` 只移除生成 C 文件中的两处错误哨兵检查，重新编译当前虚拟环境内的扩展，再验证 `dot(-1, 1) == -1`。完整入口会自动执行并缓存此修复，需要本地 C 编译器（macOS 的 Command Line Tools）。修复证据与源码/二进制摘要保存在 `results/dependency_patch.json`。这是明确记录的本地依赖修复，不声称使用了未经修改的原始 wheel。

可单独验证：`python fix_gensim_blas.py --check`。重新安装 Gensim 后需重新执行完整入口以恢复修复。

完成环境准备后执行：

```bash
python run_experiment.py
python predict.py --text '国际社会关注各国经济合作与贸易发展'
```

## 输出

`results/metrics.json` 保存真实指标、验证参数搜索、各类统计、错误文章编号、训练耗时、OOV 与近邻、环境信息和源码位置。`results/*.log` 保存各阶段记录，`results/figures/` 保存图表。`models/` 为本地模型，`data/*.jsonl` 为本地处理后的语料，二者均不上传 Git。

代码与论文对应、标签定义及局限详见正式报告。报告中的未登录词可构造向量不等于语义准确；版面分类结果也不能替代主题分类评估。
