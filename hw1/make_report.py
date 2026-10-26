"""Build the report from recorded measurements, not estimated results."""
import json
from prepare import ROOT


def table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |',
                      '| ' + ' | '.join(['---'] * len(headers)) + ' |'] +
                     ['| ' + ' | '.join(map(str, row)) + ' |' for row in rows])


def main():
    stats = json.loads((ROOT / 'data/stats.json').read_text())
    data = json.loads((ROOT / 'results/metrics.json').read_text())
    models = data['models']
    full = [r for r in models if r['training_file'] == 'train.txt']
    split_table = table(['集合', '文章数', '句子数', '词数', '未登录词词次'],
                        [[s, *[r[k] for k in ('articles', 'sentences', 'tokens', 'oov_tokens')]]
                         for s, r in stats['splits'].items()])
    timing = table(['模型', '训练词数', '重复编号 1/s', '重复编号 2/s', '重复编号 3/s', '中位数/s', 'ARPA/MB'],
                   [[r['name'], r['training_tokens'], *[f"{t['seconds']:.3f}" for t in r['runs']],
                     f"{r['median_seconds']:.3f}", f"{r['arpa_bytes']/1e6:.2f}"] for r in models])
    evaluation = table(['阶数', '验证集困惑度', '测试集困惑度', '测试预测词次（含句尾）'],
                       [[r['order'], f"{r['evaluation']['dev']['perplexity']:.2f}",
                         f"{r['evaluation']['test']['perplexity']:.2f}",
                         r['evaluation']['test']['predicted_tokens_including_eos']] for r in full])
    counts = table(['模型', '1-gram', '2-gram', '3-gram', '4-gram', '5-gram'],
                   [[r['name'], *[r['ngram_counts'].get(str(n), '-') for n in range(1, 6)]] for r in full])
    examples = []
    for r in data['generations']:
        stop = '遇到句尾标记停止' if r['stopped_at_eos'] else '达到 40 词上限'
        examples.append(f"**{r['order']}-gram，{r['setting']}**（T={r['temperature']}，top-k={r['top_k']}，top-p={r['top_p']}；{stop}）\n\n> {data['prefix']}{r['continuation']}")
    best = min(full, key=lambda r: r['evaluation']['dev']['perplexity'])
    by_name = {r['name']: r for r in models}
    bigram, trigram, fivegram = [by_name[f'{n}gram'] for n in (2, 3, 5)]
    five_eval = fivegram['evaluation']['test']
    gain = 100 * (1 - five_eval['perplexity'] / trigram['evaluation']['test']['perplexity'])
    highest_share = 100 * five_eval['matched_order_counts'].get('5', 0) / five_eval['predicted_tokens_including_eos']
    report = f'''# n-gram 建模训练

《大语言模型与应用实践》HW1 实验报告

**GitHub：** [项目仓库](https://github.com/0001Henry/LLM-P) · [HW1 代码](https://github.com/0001Henry/LLM-P/tree/main/hw1) · [在线报告](https://github.com/0001Henry/LLM-P/blob/main/hw1/REPORT.md) · [实验记录](https://github.com/0001Henry/LLM-P/tree/main/hw1/results)

## 摘要

本实验清理人民日报切分语料，用 KenLM 训练 2、3、5-gram 词级语言模型，比较训练时间，并续写“在阳光明媚的五月，我们学校胜利召开了”。训练集包含 {stats['splits']['train']['tokens']:,} 个词，所有模型使用相同的完整训练集进行阶数比较。另用训练集的部分句子测试数据量对训练开销的影响。

验证集困惑度最低的是 {best['order']}-gram，其测试集困惑度为 {best['evaluation']['test']['perplexity']:.2f}。生成部分比较了贪心选择、温度、top-k 和 top-p 设置。困惑度衡量模型预测留出新闻的能力，续写样例则用于观察语句是否连贯，两者不能互相替代。

## 1. 任务与数据准备

课件 L1 第 125 页要求清理人民日报语料中的多余标签，用课上介绍的工具训练 n-gram，比较耗时，并尝试不同参数的文本生成。本实验使用 [KenLM](https://github.com/kpu/kenlm)，只使用课程压缩包中的一月文本 `{stats['source']}`，未使用其他月份或预训练模型。

原文件按 GB18030 编码读取，已用空格分词。程序保留这些词边界，删除行首段落编号和花括号里的注音，兼容去除 `/n` 等词性后缀及方括号组合标签，再做 NFKC 字符规范化。例如，全角数字变为半角数字，全角逗号变为半角逗号。标点是实际文本的一部分，因此保留。按句号、问号和叹号切句；段落末尾剩余的片段也作为一个序列，少于 3 个词的片段不使用。

本文件没有发现 `/词性` 或方括号组合标记，实际主要清理的是编号和注音，共处理 {stats['parsed_paragraphs']:,} 个有效段落，删除 {stats['removed_pronunciations']:,} 处注音和 {stats['removed_layout_marks']:,} 个排版占位符“〓”。去重后减少 {stats['duplicate_sentences_removed']} 个完全重复的句子。去重按去掉分词空格后的文本判断，保留首次出现的位置，因此划分前的去重会影响各集合规模，但不使用标签或预测分数。

这些标记为什么要删除？编号记录文章位置，注音帮助读者认字，都不是正文里应预测的词。保留它们会占用词表和 n-gram 计数，还可能让生成结果夹带编号或注音。清理时不能直接删除所有符号，否则也会丢掉句子中的正常标点。

文章 ID 的 SHA256 前八位转为整数后对 100 取余：小于 80 分入训练集，80 至 89 分入验证集，其余分入测试集。同一篇文章的句子不会跨集合。各集合再用种子 42 打乱，训练文本的前 25% 和 50% 句子用于数据量实验。百分比按句子数计算，所以词数不严格按比例变化。

{split_table}

完整训练集词表有 {stats['vocabulary_size']:,} 个正文词项。验证集和测试集不参与词表与 n-gram 计数，未见词由 KenLM 映射为 `<unk>`。完全重复过滤并不能排除改写新闻或近重复报道。

## 2. 模型怎样训练和生成

n-gram 用前 n−1 个词预测下一个词。例如，2-gram 只看前一个词，5-gram 最多看前四个词。训练不是反复做梯度更新，而是统计连续词组的出现次数，再计算条件概率。

本次使用 KenLM `lmplz` 的 modified Kneser–Ney 平滑，不剪枝，不启用折扣回退参数。平滑会为未见过的组合留下概率；高阶组合没有记录时，模型会退回较短的上下文。这也是为什么升高阶数不等于每一步都能使用更长的上下文。具体方法见 [KenLM 训练说明](https://kheafield.com/code/kenlm/estimation/)。

每个训练进程设置 `-S 256M` 作为排序内存预算，临时文件放在本地缓存目录。这不是整个进程的内存硬上限。训练输出为 ARPA 文本模型，包含概率和回退权重；程序随后用 Python KenLM 接口加载它。训练工具源码提交为 `{data['environment']['kenlm_source_commit']}`，Python 绑定版本为 {data['environment']['kenlm_python']}。

续写前缀按课程语料的分词方式手工切为：

```text
{' / '.join(data['prefix_tokens'])}
```

每次生成时，对词表中的候选词逐一计算条件概率。排除 `<s>` 和 `<unk>`，保留 `</s>` 作为停止条件。因为排除了控制词，采样时会在剩余候选中重新归一化；困惑度计算仍使用原模型概率。最长生成 40 个新词，也可能因选中 `</s>` 提前停止，展示时不强行补标点。

- 贪心选择：直接取概率最高的词，T=0。
- 温度采样：将自然对数概率除以 T 后归一化。T 越低越偏向高概率词，T 越高分布越平坦。
- top-k：只从概率最高的 k 个候选中抽样，k=0 表示不截断。
- top-p：保留累积概率达到 p 的最小候选集合，再抽样。

代码先按 top-k 截断，再按温度换算概率，最后按 top-p 截断。本次组合用于展示不同续写方式，并非逐个参数的严格消融实验。所有样例固定种子 42，不根据输出好坏挑选或重写。

## 3. 训练时间与预测结果

实验在本机 Apple M4 CPU 上运行，未使用 GPU。Python 为 {data['environment']['python']}，系统为 `{data['environment']['platform']}`。每种训练配置运行 3 次，共 15 次，顺序用种子 42 打乱；表中重复编号对应日志文件名，不表示执行先后，实际顺序记录在 `run_sequence` 中。计时从启动 `lmplz` 到进程退出，包含读取清洗后的文本、计数、平滑和写出 ARPA，不含数据清理、依赖安装、模型加载及文本生成。文件缓存和系统负载仍可能影响时间，未测量峰值内存。

{timing}

模型中的各阶记录数如下。表中的 1-gram 还包含 KenLM 的控制词，所以与正文词表规模不同。

{counts}

本次完整数据的训练中位耗时从 2-gram 的 {bigram['median_seconds']:.3f} 秒增加到 5-gram 的 {fivegram['median_seconds']:.3f} 秒，模型文件从 {bigram['arpa_bytes']/1e6:.2f} MB 增加到 {fivegram['arpa_bytes']/1e6:.2f} MB。3-gram 使用 25%、50% 和完整训练句子时，中位耗时分别为 {by_name['3gram-25']['median_seconds']:.3f}、{by_name['3gram-50']['median_seconds']:.3f}、{trigram['median_seconds']:.3f} 秒。数据越多、阶数越高，本次需要记录的词组和写盘量也越大。不过，这只是一个月语料和一台设备的结果，不能直接按比例推算更大规模的耗时。

训练日志保存在 `results/*-run*.log`。KenLM 依次进行初始计数、调整计数、计算初始概率、插值和写出 ARPA。这些阶段反映数据处理进度，不是神经网络的 epoch，也没有随轮次下降的训练 loss。

{evaluation}

困惑度按 `10 ** (-总 log10 概率 / 预测词次)` 计算，计入每句的 `</s>`，不计入 `<s>`。数值越低，说明模型对这份留出文本分配的平均概率越高。三种完整数据模型的词表和评估句子相同，因此可以直接比较。部分数据模型的词表不同，本报告不把它们的困惑度放在同一张排名表中。

结果文件还记录了每次预测实际匹配的 n-gram 阶数，可用于查看高阶模型有多少位置退回了低阶上下文。模型选择只参考验证集；测试集用于最后比较，不用于调整采样参数。未登录词被合并成同一个 `<unk>`，所以这里的困惑度不等于对每个未知词原始字面的预测能力。

从 3-gram 升到 5-gram，测试困惑度下降了 {gain:.2f}%，但 ARPA 文件大小约为原来的 {fivegram['arpa_bytes']/trigram['arpa_bytes']:.2f} 倍。5-gram 在测试时只有 {highest_share:.2f}% 的位置直接匹配到完整五词组合，其余位置使用较短上下文。这有助于理解为什么增加阶数的收益变小，但还不足以单独解释全部性能差异。

## 4. 指定开头的续写

以下给出全部 12 组输出。前缀原文保留全角逗号，模型内部采用规范化后的半角逗号。未登录前缀词为：{json.dumps(data['generations'][0]['prefix_oov'], ensure_ascii=False)}。

{chr(10).join(chr(10) + e + chr(10) for e in examples)}

这批输出中，3-gram 和 5-gram 的贪心方式都只接了一个句号，句子缺少“召开”的具体对象。2-gram 则接出了“在北京”“日电”等新闻常见写法，没有延续校园话题。T=0.7、top-k=10 时，3-gram 和 5-gram 能生成较长片段，但分别出现“一系列新的一条”和“中国第一次”等接法，后文也不断转移话题。局部词组接得上，不代表整句意思完整。

T=1.3、top-p=0.95 的输出出现更多零散词语，连贯性比低温样例更差。参数只改变从已有概率中怎样选词，不能补上模型缺少的主题记忆。前缀中的“阳光明媚”虽然不在词表，但到真正续写时，它已经超出这些模型的上下文窗口，因此也不能把续写失败简单归因于这个未登录词。

本次训练文本是新闻，指定前缀却要求校园场景。模型每步只看有限的几个词，无法始终记住“我们学校”这个话题。即使局部搭配通顺，整段也可能转向新闻中的机构、会议或事件。本文只做单个前缀、单个随机种子的展示，没有人工评分，不能据此给出一般性的生成质量排名。

## 5. 代码理解与复现

`prepare.py` 负责清洗、切句、去重和按文章划分。`experiment.py` 调用 KenLM，记录每次耗时，再计算留出集困惑度。模型训练的计数和平滑由 KenLM 完成，没有另写一套替代算法。

`generate.py` 中的 `State` 保存最近的上下文；`BaseScore` 给出接上下一个词的 log10 概率和新状态。枚举候选时始终使用同一个旧状态，选定词后才更新状态，避免候选之间互相影响。`choose` 将 log10 概率换为自然对数尺度，再做温度和截断采样。为便于理解，当前实现遍历整个词表，生成速度不是本实验的优化目标。

自动测试检查标记清理、字符规范化、句子切分、集合隔离、采样参数、模型重载、状态评分与整句评分的一致性，以及固定种子的重复生成。实际测试记录见 `results/tests.log`。

安装过程中，KenLM 0.3.0 的 Python 扩展在本机 Python 3.13 下编译失败，改用 Python 3.12 后安装成功。Homebrew 下载也遇到阻塞，训练工具改用项目缓存内的编译依赖构建。具体命令、文件说明和重新运行方法见 [README](README.md)。这些准备时间不计入训练耗时。

原始数据 SHA256：`{stats['source_sha256']}`。模型校验值、工具版本、训练顺序、原始时间和全部生成词序列保存在 `results/metrics.json`。模型、环境和语料仅留在本地，不纳入 Git。

## 6. 小结

本次 5-gram 的留出集预测效果最好，但相对 3-gram 的改善有限，文件明显更大；续写也没有因为阶数提高就变得可靠。对这份不到百万词的训练文本，KenLM 训练很快，主要困难反而是保持句意和话题。后续可以增加校园相关语料和更多前缀，再检查这些改动是否真正改善续写，而不是只看生成文本是否变长。

## 参考资料

1. 课程课件《2026-秋季-大语言模型与应用实践-L1.pptx》，第 125 页。
2. 课程随附人民日报切分语料：`1998-上半年-带音(1).rar`。
3. [KenLM 官方源码](https://github.com/kpu/kenlm)。
4. [KenLM 训练、平滑与参数说明](https://kheafield.com/code/kenlm/estimation/)。
5. Heafield et al. (2013). [Scalable Modified Kneser-Ney Language Model Estimation](https://aclanthology.org/P13-2121/)。
'''
    (ROOT / 'REPORT.md').write_text(report, encoding='utf-8')
    print('Generated REPORT.md')


if __name__ == '__main__':
    main()
