# HW3：高效率的 BERT 模型古文断句标点

对应课件第 68 页的两个任务：用现成语料微调预训练 BERT，以及对长文本进行高效率 Batch 滑窗推理。没有从零预训练 BERT。

## 文件

- `prepare.py`：读取随课提供的人民日报 1998 年 1 月带音语料，去编号、分词空格、读音注释，生成逐字标点标签。
- `bert_punct.py`：微调、验证集选择最佳模型、测试、长文推理、速度对比。
- `core.py`：清洗、标签、窗口、指标。
- `test_core.py`、`test_inference.py`：边界与融合算法测试。
- `results/`：实际运行的训练记录、测试指标、测速、示例。
- `models/punctuation/`：实际微调后保存的模型。
- `run_experiment.py`：一键重跑本次小规模训练、测试、测速和古文示例，记录日志。
- [REPORT.md](REPORT.md)：可独立阅读的正式 Markdown 实验报告，含两个任务的方法、结果分析、环境配置和复现步骤。
- `REPORT.template.txt`：报告正文模板；`make_report.py` 从实际结果生成数据表，避免重新生成时退回简略版报告。

## 环境和数据

在本目录运行以下命令。当前机器已单独建立 `.venv`，不改变系统 Python 环境。

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
mkdir -p data/raw
bsdtar -xf '../1998-上半年-带音(1).rar' -C data/raw '1998-上半年-带音/1998-01-2003版-带音.txt'
python prepare.py
python download_model.py
python -m unittest discover -v
```

预训练模型来自官方仓库：https://huggingface.co/google-bert/bert-base-chinese 。需要约 400 MB 权重下载空间。下载脚本仅在文件完整下载后改名，连接失败可以重跑。`requirements-lock.txt` 记录本次实际环境。

语料使用 GB18030 解码。文章编号的前三部分作为文章 ID，先按文章 ID 的 SHA256 固定划分约 80%/10%/10%，再切分窗口，防止同一文章的段落跨集合泄漏。相同无标点文本全局去重。短于 10 字的段落被过滤，其余段落保留，包括无目标标点的段落。

## 任务一：微调

每个字预测其后是否有标点。标签为 `O ， 。 ？ ！ ； ： 、`。交叉熵忽略 `[CLS]`、`[SEP]` 与 padding（标签 `-100`）。标签由原始带标点文本自动生成，因此推理输入不含目标标点。连续目标标点取最后一个；引号、括号、破折号、省略号等不在本作业标签集中，清洗时删除。空白也删除，因此输出保留的是清洗后的字序列，而非原始排版。

为严格保证字符与标签对齐，对每个 Unicode 字符直接查询 BERT 词表，每字一个 token，英文字符转小写查询，未知字符用 `[UNK]`，重建时仍使用原字符。这适合汉字主体文本，但不是通用英文 WordPiece 分词，罕见古字也可能损失信息。400 字窗口加两个特殊 token 为 402，低于 512 上限。

完整语料训练命令（默认更新全部参数）：

```bash
python bert_punct.py train --epochs 3 --batch 8
```

资源受限实验可以设 `--train-limit`、`--eval-limit` 和 `--freeze-layers`。冻结前若干编码层后仍更新上层 BERT 与分类头，是部分参数微调，不是从零训练。设为 0 表示不限制样本量或不冻结层。具体实跑参数以 `results/training.json` 为准，不能把子集结果称为全量训练结果。

本次可复现实验使用 1,000 个训练段落、各 100 个验证和测试段落、3 轮训练，冻结 embedding 和前 10 层 BERT。运行 `python run_experiment.py` 重现此配置。程序自动选择 CUDA、MPS 或 CPU；MPS 使用 eager attention，避开本机 PyTorch 的 SDPA 不支持训练 dropout 的限制。

按验证集的标点 micro-F1 保存最佳 epoch，最后仅用最佳 checkpoint 测试。主要指标排除 `O` 的真负例，避免“全预测无标点”取得虚高分数。另报告不区分标点类型的断点 F1、各类标点 P/R/F1 和字符准确率。

## 任务二：高效率长文本处理

串行基线每次输入 400 字，在前 300 字内找最后一个预测句号，以它后面的位置作为下一窗口起点。如果找不到句号，固定前进 300 字。尾窗输出全部结果。这种方法有前后依赖，不能预先批量生成所有窗口。

改进方案先按固定步长 300 建立长度 400 的重叠窗口，独立预测每一窗，不依赖上一窗的句号。把最多 B 个窗口一起送入 BERT，以 attention mask 屏蔽 padding。最后按原字符位置融合重叠区概率，整篇只解码一次，不在窗口边界强行插句号。

窗口中第 i 个字（从 0 开始）的权重为 `min(i+1, 窗口长度-i)`，中心权重大于边缘。全局字位置的概率为覆盖它的各窗口概率的加权平均，然后 argmax。每个字恰好输出一次，既不会重复，也不会遗漏。窗口任意一侧即使没有句号也不影响前进。

设窗口数为 K，则 B=1 需要 K 次 forward，Batch 方案只需 `ceil(K/B)` 次。它减少的是模型调用与调度开销、提高设备并行利用率，并没有消除重叠区的计算。总 token 计算量仍约为文本长度的 `400/300` 倍。模型输入显存约随 B 增大，融合缓存为 `N×8` 的 float32 概率。特别长的书籍可以进一步按块写出已完成融合的前缀。

```bash
python bert_punct.py benchmark --model models/punctuation --batch 8 --chars 4000 --repeats 3
python bert_punct.py predict --model models/punctuation --text '学而时习之不亦说乎有朋自远方来不亦乐乎'
python bert_punct.py predict --model models/punctuation --input ancient.txt
python bert_punct.py evaluate --model models/punctuation
```

测速使用留出的测试段落拼成数千字长文，比较串行句号截断、固定窗口 B=1、固定窗口 B=8。后两个方案才是只改变 batch 的受控对比；串行方案窗口上下文不同，精度可能不同。预热后同步设备计时，重复多次取中位数，包括词表查找、数据传输、forward 和融合，不含模型加载。拼接的段落未必来自相邻文章，所以该测试主要验证长文工程流程，不能代表连贯古文的精度。

Batch 不保证在所有 CPU/MPS/GPU 上更快，不能用调用次数的比例代替真实速度提升。若内存不足应减小 batch；本代码不掩盖 OOM 或自动偷换窗口长度。现代新闻训练的模型存在古文领域偏移，不能把新闻测试集指标宣称为古文指标。

## 参考

- 作业来源：上级目录 `2026-秋季-大语言模型与应用实践-L2-part2.pptx` 第 68 页。
- 数据来源：上级目录随附 `1998-上半年-带音(1).rar`，只使用 1998 年 1 月文件。
- 预训练模型：https://huggingface.co/google-bert/bert-base-chinese
