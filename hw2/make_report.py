"""Generate the Markdown submission report and figures from measured results."""
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.cache/matplotlib'))
os.environ.setdefault('XDG_CACHE_HOME', str(ROOT / '.cache'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def table(headers, rows):
    return '\n'.join(['| '+' | '.join(headers)+' |', '| '+' | '.join(['---']*len(headers))+' |']+
                     ['| '+' | '.join(str(v) for v in row)+' |' for row in rows])


def main():
    stats = json.loads((ROOT / 'data/stats.json').read_text())
    data = json.loads((ROOT / 'results/metrics.json').read_text())
    rows = data['experiments']
    selected = next(r for r in rows if r['classes'] == 6 and r['method'] == data['selected_fasttext'])
    methods = list(dict.fromkeys(r['method'] for r in rows))
    assets = ROOT / 'results/figures'
    assets.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, ax = plt.subplots(figsize=(9, 5), layout='constrained')
    for name in methods:
        scores = [next(r for r in rows if r['method'] == name and r['classes'] == n)['test']['macro_f1'] for n in (2,4,6)]
        ax.plot([2,4,6], scores, marker='o', label=name)
    ax.set(xlabel='Number of page labels', ylabel='Test macro-F1', xticks=[2,4,6], ylim=(0,1),
           title='Representation comparison on article page classification')
    ax.grid(alpha=.2)
    ax.legend(loc='upper left', bbox_to_anchor=(1,1))
    fig.savefig(assets / 'classification.png', dpi=160)
    plt.close(fig)
    matrix = np.array(selected['test']['confusion_matrix'])
    fig, ax = plt.subplots(figsize=(6,5), layout='constrained')
    im = ax.imshow(matrix, cmap='Blues')
    for i in range(6):
        for j in range(6):
            ax.text(j,i,str(matrix[i,j]),ha='center',va='center',color='white' if matrix[i,j]>matrix.max()/2 else 'black')
    ax.set(xticks=range(6), yticks=range(6), xticklabels=stats['labels'], yticklabels=stats['labels'],
           xlabel='Predicted page', ylabel='True page', title=data['selected_fasttext']+' / six classes')
    fig.colorbar(im, ax=ax)
    fig.savefig(assets / 'confusion.png', dpi=160)
    plt.close(fig)
    count_table = table(['版面标签','训练文章','验证文章','测试文章'],
        [[p]+[stats['split_counts'][s][p] for s in ('train','dev','test')] for p in stats['labels']])
    result_table = table(['类别数','表示方法','验证 Macro-F1','所选 C','测试 Accuracy','测试 Macro-F1'],
        [[r['classes'],r['method'],f"{r['dev_macro_f1']:.4f}" if 'dev_macro_f1' in r else '不调参',r.get('selected_C','—'),
          f"{r['test']['accuracy']:.4f}",f"{r['test']['macro_f1']:.4f}"] for r in rows])
    time_table = table(['词向量方法','表示训练时间/s','全部文章向量化时间/s','词表规模','测试词次 OOV 比例','可构造向量的 OOV 词次'],
        [[name,f"{t['train_seconds']:.3f}",f"{t['vectorize_seconds']:.3f}",data['diagnostics'][name]['vocabulary'],
          f"{data['diagnostics'][name]['test_oov_rate']:.2%}",data['diagnostics'][name]['test_oov_representable']]
          for name,t in data['representation_times'].items() if isinstance(t,dict)])
    class_table = table(['版面','Precision','Recall','F1','支持数'],
        [[p]+[f"{selected['test']['per_class'][p][k]:.4f}" for k in ('precision','recall','f1-score')]+[int(selected['test']['per_class'][p]['support'])] for p in stats['labels']])
    nearest = table(['模型','查询词','最近邻（余弦相似度）'],
        [[name,q,'；'.join(f'{w} ({v:.3f})' for w,v in neighbors)] for name,diag in data['diagnostics'].items()
         for q,neighbors in diag['neighbors'].items() if q in ('经济','科学','足球')])
    probes = table(['模型','未登录词','可以计算向量','向量范数'],
        [[name,w,v['available'],f"{v['norm']:.4f}" if v['norm'] is not None else '不支持']
         for name,diag in data['diagnostics'].items() for w,v in diag['oov_probes'].items()])
    locations = table(['函数','安装包源码位置'],
        [[name,f"`{info['file']}:{info['line']}`"] for name,info in data['source_locations'].items()])
    best_dev = max((r for r in rows if r['classes']==6 and 'dev_macro_f1' in r),key=lambda r:r['dev_macro_f1'])
    ft24 = next(r for r in rows if r['classes']==6 and r['method']=='FastText-2-4')
    ft36 = next(r for r in rows if r['classes']==6 and r['method']=='FastText-3-6')
    env = data['environment']
    errors = table(['文章 ID（仅用于定位）','真实版面','预测版面'],
                   [[e['id'],e['gold'],e['prediction']] for e in selected['errors'][:8]])
    report = f'''# Gensim 表示学习与 FastText 文本分类

《大语言模型与应用实践》HW2 实验报告

**GitHub：** [项目仓库](https://github.com/0001Henry/LLM-P) · [HW2 源代码](https://github.com/0001Henry/LLM-P/tree/main/hw2) · [在线实验报告](https://github.com/0001Henry/LLM-P/blob/main/hw2/REPORT.md) · [实验结果](https://github.com/0001Henry/LLM-P/tree/main/hw2/results)

## 摘要

本实验依据课程课件 L2.part1 第 137 页，完成新闻语料处理、多种 Gensim 表示学习、基于 FastText 向量的文本分类及论文与源码对照。使用人民日报一月文章进行版面分类，以二月新闻和一月训练部分构建无监督表示语料，比较 BoW、TF-IDF、LSI、Word2Vec 及两种字符 n-gram 配置的 FastText，并按 2、4、6 类逐步展开。分类器统一采用逻辑回归，仅通过验证集选择正则化参数。

按六分类验证集选择的 FastText 配置为 **{data['selected_fasttext']}**，独立测试 Accuracy 为 **{selected['test']['accuracy']:.4f}**，Macro-F1 为 **{selected['test']['macro_f1']:.4f}**。本文明确区分 Gensim 的无监督 FastText 词向量与原生 FastText 监督分类器，不将两阶段实验表述为原生监督实现的复现。

**关键词：** Gensim；FastText；词向量；文本分类；未登录词

## 1. 实验目标与完成范围

1. 了解 Gensim 的词典、稀疏语料、向量模型和模型持久化接口，完成不同表示方法的训练与应用。
2. 训练 FastText，将文章映射为固定维度向量，完成从少类别到多类别的分类实验。
3. 对照子词表示论文和高效率分类论文阅读源码，说明核心计算、数据管线及超参数作用。

本实验真实执行了数据准备、无监督模型训练、监督分类训练、验证集调参、独立测试、模型保存和重新加载预测。全部实验使用本地 CPU，未加载预训练词向量。

## 2. 数据、标签与划分

### 2.1 数据来源与标签定义

数据来自课程随附的 `1998-上半年-带音(1).rar`。一月语料用于分类，另选二月语料作为新闻风格一致的无标签表示学习文本。二月数据属于独立月份，不是额外人工标注的分类样本。

原始编号形如 `19980101-06-001-002`，依次对应日期、版面、文章和段落。本文选择 **01 至 06 版的版面号**作为六个类别。二分类使用 01、02 版，四分类增加 03、04 版，六分类再增加 05、06 版。

这是一项依据文章正文预测报纸版面的分类任务。版面与主题可能相关，但不等于稳定的人工主题标签，尤其不同日期的版面内容可能变化。因此不将标签擅自命名为经济、体育等主题，也不把该实验的分数解读为通用主题分类性能。编号仅用于聚合、划分和错误定位，不进入模型特征。

### 2.2 清洗与分词

以 GB18030 读取原文，去除编号、分词空格及花括号读音注释，按文章 ID 聚合段落并按段落序号排序。采用 NFKC 规范化并转小写，再用 jieba（`HMM=False`）统一重新分词。仅保留汉字词或英文字母词，去除纯数字、标点及其他符号；不使用外部停用词表，保留单字词。采用统一分词规则是为了保证训练与新文本推理一致。

按分词后拼接文本的摘要去除完全重复文章，过滤不足 30 个 token 的文章。对每个类别分别固定种子 42 打乱，按约 70%/15%/15% 划分训练、验证和测试文章。文章的各段落始终属于同一集合。

{count_table}

### 2.3 表示学习语料与数据隔离

表示语料由一月训练文章中的段落和随机抽取的 **{stats['february_paragraphs']:,} 个二月段落**组成。二月段落排除与一月任何段落完全相同的文本；一月训练段落若与留出文章的段落完全相同，也不用于表示训练。段落按最多 128 个 token 切成训练单元，丢弃不足 3 个 token 的尾片段。最终得到 **{stats['representation_sentences']:,} 个单元、{stats['representation_tokens']:,} 个 token**。

所有表示方法使用同一份无标签语料建立词典或训练向量，验证集和测试集不参与 IDF、LSI 或词向量拟合。去重过程仅比较文本是否重复，不利用留出标签优化特征。分类器只使用一月训练标签。

为控制表示差异，2、4、6 类实验共享六类训练部分建立的表示模型；较少类别实验仍使用其余类别的无标签训练文本。这不是“只用两类文本”的完全独立实验。各任务的类别集合和样本规模同时变化，因此分数变化不能仅归因于类别数量。没有进行跨月未来预测：二月无标签文本被用于一月分类表示学习。

完全重复过滤不能排除改写报道或近重复内容。表示训练排除了与留出文章完全相同的段落，但分类输入仍保留整篇文章，不同文章可能共享部分措辞或段落，不能据此声称消除了所有内容重叠。本文没有额外的时间外推测试，该限制应在结果解释时保留。

## 3. 表示学习方法

### 3.1 BoW、TF-IDF 与 LSI

使用 `gensim.corpora.Dictionary` 建立词典，过滤少于两个训练单元出现的词、超过 95% 单元出现的词，最多保留 20,000 个词。本次词典实际包含 **{data['dictionary_size']:,} 个词**。`doc2bow` 将文章映射为词频向量，BoW 在分类前进行 L2 归一化。

`TfidfModel` 在表示训练语料上估计 IDF，将高频但区分度低的词降权，并使用默认归一化。其文档频率的统计单位是上述训练单元，而不是测试文章。LSI 使用 `LsiModel` 对训练 TF-IDF 语料进行低秩投影，主题数为 64；分类文章投影到相同空间后再做 L2 归一化。

### 3.2 Word2Vec 与 FastText

Word2Vec 使用 Skip-gram 和负采样学习词级共现表示。FastText 在此基础上引入字符 n-gram，使词形内部片段能够共享参数，并允许为未进入整词词表的词构造向量。该设计来源于 [Bojanowski 等人的子词表示论文](https://aclanthology.org/Q17-1010/)。

两个 FastText 配置分别使用字符 n-gram 长度 2–4 和 3–6，其余显式参数相同。它们是字符片段，不是跨词的 bigram。词边界标记也参与子词构造。中文大量词长度较短，设置 2–4 用于观察较短字符片段的效果，但本文不预设其一定优于 3–6。

| 参数 | 本次设置 | 含义 |
| --- | --- | --- |
| vector_size | 64 | 稠密词向量维度 |
| window | 5 | 最大上下文窗口范围 |
| sg | 1 | Skip-gram，非 CBOW |
| negative / hs | 5 / 0 | 每个正例使用 5 个负样本，不启用层次 softmax |
| min_count | 3 | 整词最小出现次数 |
| sample | 0.001 | 高频词下采样阈值 |
| epochs | 5 | 表示语料训练轮数 |
| min_n / max_n | 2/4 或 3/6 | FastText 字符片段长度范围 |
| bucket | 50,000 | 子词哈希桶数量，控制内存并引入碰撞可能 |
| workers / seed | 1 / 42 | 单工作线程与固定随机种子 |
| PYTHONHASHSEED | 42 | 由完整实验入口在子进程启动前设置 |

文章表示为其可用词向量的算术平均，再进行 L2 归一化。Word2Vec 跳过词表外词；FastText 通过子词构造它们的向量。空的可用词集合返回零向量，交互预测则拒绝完全没有可用文字的输入。均值池化保留大致语义，但丢失文章词序和结构。

负采样训练的直观目标是提高中心词与真实上下文的内积分数，同时降低与负样本的分数；FastText 将中心词表示扩展为整词及字符片段的组合。记可用词向量为 v(w)，本文的文章表示为 `z = L2Normalize(mean(v(w)))`。该文章表示构造是本实验的下游选择，不是 FastText 词表示论文直接提供的文档分类标签。

### 3.3 分类器与选择协议

所有非多数类基线均使用 `LogisticRegression(class_weight='balanced', max_iter=1000)`。在每项任务的验证集上比较 `C ∈ {{0.1, 1, 10}}`，按 Macro-F1 选取 C，同分时保留较小 C。C 越大正则化越弱。验证集不合并回训练集，选定后直接评估独立测试集。

另设置始终预测训练集中最多类别的 Majority 基线。主要指标为 Macro-F1，同时保留 Accuracy、Weighted-F1、每类 Precision/Recall/F1 及混淆矩阵。测试结果只用于报告，不用于选择 C。六分类的演示预测模型在两个 FastText 变体中按验证 Macro-F1 选择，而不是按测试分数选择。

## 4. 实验结果

### 4.1 从二分类到六分类

{result_table}

![各方法随类别数量变化的测试 Macro-F1](results/figures/classification.png)

在六分类中，所有方法里验证 Macro-F1 最高的是 **{best_dev['method']}**，其测试 Macro-F1 为 **{best_dev['test']['macro_f1']:.4f}**。FastText-2-4 和 FastText-3-6 的六分类测试 Macro-F1 分别为 **{ft24['test']['macro_f1']:.4f}** 和 **{ft36['test']['macro_f1']:.4f}**。这是单次固定划分的实测比较，未进行多种随机种子下的显著性检验，不能仅据小幅分差判定某个配置普遍更好。

稀疏表示直接保留区分类别的关键词，而低维表示和均值池化可能平滑掉细粒度词汇信息。FastText 的子词能力主要解决词形共享和 OOV 构造问题，并不保证在所有分类任务上优于 TF-IDF 或 Word2Vec。

### 4.2 选定 FastText 的六分类表现

{class_table}

![六分类混淆矩阵](results/figures/confusion.png)

矩阵行为真实版面，列为预测版面。不同版面的内容分布可能重叠，同类内部也会因日期而变化，因此版面号本身不构成完美的语义类别。以下错误只展示文章编号和预测，不反向改变标签规则：

{errors}

### 4.3 训练开销与 OOV

{time_table}

时间来自 `perf_counter` 的单次本机 CPU 测量。表示训练时间包含模型建立与训练、不包含保存；向量化时间包含三个分类集合的文章均值表示与归一化。分类器的 C 搜索时间和测试预测时间另存于 `results/metrics.json`，不与表示训练混为一谈。未采用重复测速或峰值内存监测，因此不能把这些值当作跨设备性能结论。

OOV 按词是否存在于 `wv.key_to_index` 判定，而不是使用 FastText 的 `word in wv`：后者可能仅表示能够通过子词构造向量。以下人工拼接词只用于检验接口，不是语义质量测试：

{probes}

“可以构造向量”不等于这些字符片段都充分训练过，也不等于向量有正确语义。哈希碰撞和未充分训练的桶都可能影响输出。

### 4.4 词向量近邻观察

{nearest}

近邻由 `most_similar` 计算，采用余弦相似度。结果可以反映语料中的共现关联，但近邻不必是同义词，也可能包含同一报道事件的人名或地名。本文未使用人工标注的词相似度或类比评测集，因此该部分仅作定性观察。

## 5. 论文与源码对照

### 5.1 两种 FastText 用法的边界

[Enriching Word Vectors with Subword Information](https://aclanthology.org/Q17-1010/) 研究通过字符片段增强词表示；[Bag of Tricks for Efficient Text Classification](https://aclanthology.org/E17-2068/) 则将文本特征平均后用于标签预测，并讨论词级 n-gram 和高效率分类。二者共享浅层表示思想，但训练目标不同。

本实验用 Gensim 训练无监督 FastText，再冻结词向量、平均池化并交给 scikit-learn 逻辑回归。标签梯度不会回传到词向量，没有实现原生监督 FastText 的端到端词表示更新、词级 n-gram 分类特征或层次 softmax。因此这是课件要求的“基于向量完成分类”，不是对原生监督分类器的逐项复刻。[Gensim 模块说明](https://radimrehurek.com/gensim/models/fasttext.html)也将该模块定位为词表示学习接口。

### 5.2 本机源码定位

实验采用 Gensim {env['gensim']}，并应用下述已记录的本地 BLAS 包装修复。下表通过 `inspect` 从实际安装包定位，完整源文件 SHA256 记录在结果 JSON 中。路径相对于环境的 `site-packages`：

{locations}

阅读 [FastText 源码](https://github.com/piskvorky/gensim/blob/4.4.0/gensim/models/fasttext.py) 可见，`build_vocab` 建立词表及参数矩阵，`_do_train_job` 调用底层 `train_batch_any`。`adjust_vectors` 将整词与子词向量组合并平均；`get_vector` 对整词词表外输入按子词桶求向量，不能据此声称该整词出现过。

[Word2Vec 源码](https://github.com/piskvorky/gensim/blob/4.4.0/gensim/models/word2vec.py) 中的 `prepare_vocab` 涉及词频过滤、下采样及负采样分布准备，`train` 负责训练轮次、任务调度和学习率进度。这些周边逻辑决定真正参与训练的词次与复现条件，不能只关注内积或梯度公式。

本项目的数据管线、Gensim 模型调用、文章均值池化与验证集调参分别对应 `data_utils.py`、`experiment.py` 中的表示训练、`average_vectors` 和 C 搜索循环。模型通过 `.save()` 保存，分类器通过 joblib 保存，`predict.py` 重新加载相同组件完成预测。

### 5.3 底层点积包装的回归修复

实际运行中出现 `our_dot_float` 的异常提示。检查安装包的 `word2vec_inner.pxd` 和生成的 C 文件发现：BLAS 点积函数指针被标记为 `except -1`，生成代码据此把合法的返回值 `-1` 当作异常。直接调用已导出的 Gensim 点积包装，修复前 `dot([-1], [1])` 返回了 `0`，而数学上应为 `-1`。这与切换 Accelerate/OpenBLAS 是两个独立问题，单纯更换数值库不能消除该哨兵判断。

本项目提供 `fix_gensim_blas.py`：在校验 Gensim 版本为 4.4.0 且恰有两处对应检查后，仅移除生成 C 文件中 float/double 点积包装的错误哨兵判断，重新编译 `word2vec_inner` 扩展。修复后通过相同接口验证结果为 `-1`，并在自动测试中保留该回归用例。未修改 Skip-gram、负采样或 FastText 的训练目标。

源码摘要、修复后摘要及二进制摘要记录于 `results/dependency_patch.json`；修复只作用于本实验的独立环境。最终报告采用修复后重新执行的结果，不能将该环境称为未经修改的原始 Gensim wheel。若重新安装依赖，需要重新应用修复。

## 6. 验证、局限与结论

自动测试覆盖编号解析、读音清理、确定性划分、文章 ID 与文本摘要隔离、分词过滤、FastText OOV、Word2Vec 未登录词、空向量、均值池化及分类指标。测试中的微型随机模型仅用于接口验证，不代替正文中的真实语料训练。运行记录见 `results/tests.log`。

训练脚本还检查词向量中的数值是否有限，以及训练前后前 256 个词向量是否发生实际变化，变化范数记录于 `results/metrics.json`。最终使用 NumPy 2.2.6、SciPy 1.15.3 的 OpenBLAS 构建及上述点积修复重新执行完整实验。入口检查数值异常和收敛警告，发现异常则停止，不继续生成成功报告。本文只采用最终运行结果。

主要局限包括：版面标签只是内容分类的代理；未完全消除近重复新闻；表示语料额外使用了二月文本；仅一次划分与有限参数网格；均值池化忽略词序；子词哈希桶较小；没有标准语义评测、监督 FastText 原生实现或跨时间测试。后续可以人工整理稳定主题标签，进行多种随机划分和时间划分，比较 TF-IDF 加权池化及原生监督 FastText，但不得将未执行的方案作为本次结果。

综上，本实验跑通了 Gensim 的稀疏与稠密表示学习，并完成由 2 类扩展到 6 类的实际分类训练与测试。选定 FastText 模型在六分类测试集上的 Accuracy 为 {selected['test']['accuracy']:.4f}、Macro-F1 为 {selected['test']['macro_f1']:.4f}。实验同时验证了子词构造对未登录词的覆盖能力，以及“向量覆盖能力”和“下游分类效果”需要分别评价。

## 附录 A. 环境与复现

- Python：{env['python']}
- Gensim：{env['gensim']}
- scikit-learn：{env['sklearn']}
- NumPy：{env['numpy']}
- SciPy：{env['scipy']}
- 系统：{env['platform']}
- BLAS：{', '.join(str(b.get('internal_api'))+' '+str(b.get('version')) for b in env['blas'])}

在 `hw2/` 下执行：

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-lock.txt
mkdir -p data/raw
bsdtar -xf '../1998-上半年-带音(1).rar' -C data/raw '1998-上半年-带音/1998-01-2003版-带音.txt' '1998-上半年-带音/1998-02-2003版-带音.txt'
```

本次 Apple Silicon 环境还需选择 OpenBLAS 构建，避免默认 Accelerate 构建的数值提示；其他平台不执行以下平台专用命令：

```bash
python -m pip download --only-binary=:all: --platform macosx_12_0_arm64 --dest .cache/wheels 'numpy==2.2.6' 'scipy==1.15.3'
python -m pip install --force-reinstall --no-index --find-links .cache/wheels 'numpy==2.2.6' 'scipy==1.15.3'
```

随后执行完整实验。入口自动应用并验证本地 Gensim 点积修复，需要 C 编译器；macOS 可使用已安装的 Command Line Tools：

```bash
python run_experiment.py
```

其他平台若不能安装锁定依赖，可使用 `requirements.txt` 并重新记录版本。入口程序设置随机种子和单线程环境，仍不保证不同 BLAS、系统或依赖版本的浮点结果逐位一致。重新运行会更新同名模型与结果。

```bash
python predict.py --text '国际社会关注各国经济合作与贸易发展'
python make_report.py
```

第一个命令加载六分类验证选定的 FastText 与分类器，返回版面标签和概率；概率未额外校准。第二个命令只根据现有结果重建 Markdown 报告和图，不重新训练。

## 附录 B. 文件与证据

| 文件 | 内容 |
| --- | --- |
| `data_utils.py` | 数据清洗、文章聚合、划分与表示语料准备 |
| `experiment.py` | 全部表示训练、C 选择、分类评估及源码定位 |
| `predict.py` | 模型重载和新文本推理 |
| `test_hw2.py` | 数据与模型接口自动测试 |
| `fix_gensim_blas.py` | 本地依赖扩展的窄范围兼容修复及回归验证 |
| `results/dependency_patch.json` | 修复源码与二进制摘要、回归结果 |
| `run_experiment.py` | 完整执行入口及逐阶段日志 |
| `make_report.py` | 正式报告与图表生成 |
| `data/stats.json` | 数据规模与输入文件校验值 |
| `results/metrics.json` | 参数、全部指标、错误、耗时、OOV 与源码证据 |
| `results/figures/` | 分类比较图和混淆矩阵 |
| `results/*.log` | 数据准备、测试、训练与预测记录 |
| `models/` | Gensim 模型和各任务分类器，仅本地保存 |

代码、报告和结果可纳入版本控制；虚拟环境、语料与模型权重由 `.gitignore` 排除。原始数据文件 SHA256：

```text
{json.dumps(stats['source_sha256'],ensure_ascii=False,indent=2)}
```

## 参考资料

1. 课程课件《2026-秋季-大语言模型与应用实践-L2.part1.pptx》，第 137 页。
2. [Gensim 官方文档](https://radimrehurek.com/gensim/)。
3. Bojanowski et al. (2017). [Enriching Word Vectors with Subword Information](https://aclanthology.org/Q17-1010/).
4. Joulin et al. (2017). [Bag of Tricks for Efficient Text Classification](https://aclanthology.org/E17-2068/).
5. [Gensim 4.4.0 FastText 源码](https://github.com/piskvorky/gensim/blob/4.4.0/gensim/models/fasttext.py)及 [Word2Vec 源码](https://github.com/piskvorky/gensim/blob/4.4.0/gensim/models/word2vec.py)。
'''
    (ROOT / 'REPORT.md').write_text(report, encoding='utf-8')
    print('Generated REPORT.md and two result figures.')


if __name__ == '__main__':
    main()
