# NanoChat 代码解读指南

> 基于 Karpathy 的 NanoChat 项目 - "用 $100 训练一个 LLM"
> 约 8000 行 PyTorch 代码，覆盖 LLM 全栈开发

---

## 📚 项目概述

NanoChat 是一个极简、可 hack、依赖少的 LLM 全栈实现，包含：
- **Tokenization** (分词器训练)
- **Pretraining** (预训练)  
- **Fine-tuning** (SFT微调 + 强化学习)
- **Evaluation** (评估)
- **Inference** (推理)
- **Chat UI** (Web对话界面)

**成本对比**:
- 2019年 GPT-2 训练成本: ~$43,000
- 2025年 NanoChat GPT-2级别: ~$73-100 (8×H100, 约3小时)

---

## 🗂️ 代码结构与文章对照

### 阶段一：分词器 (Tokenization)

| 文件 | 功能 | 关键点 |
|------|------|--------|
| `nanochat/tokenizer.py` | BPE分词器 | RustBPE训练 + tiktoken推理 |
| `scripts/tok_train.py` | 分词器训练 | 32K词表 |
| `scripts/tok_eval.py` | 压缩率评估 | bytes/token |

```python
# tokenizer.py 核心代码
class RustBPETokenizer:
    def train_from_iterator(cls, text_iterator, vocab_size):
        """训练BPE分词器，vocab_size=32768"""
        
    def render_for_completion(self, conversation):
        """将对话格式化为tokens，包含特殊token"""
```

---

### 阶段二：模型架构 (GPT Model)

| 文件 | 功能 | 关键点 |
|------|------|--------|
| `nanochat/gpt.py` | GPT Transformer | 现代改进架构 |
| `nanochat/flash_attention.py` | 注意力优化 | FA3/SDPA自动切换 |

**架构特点** (vs 原版GPT-2):
- ✅ RoPE旋转位置编码 (无需学习位置嵌入)
- ✅ QK Norm (稳定训练)
- ✅ ReLU² 激活函数 (MLP)
- ✅ RMSNorm (无可学习参数)
- ✅ 滑动窗口注意力 (SSSL pattern)
- ✅ Value Embedding (VE)

```python
# gpt.py 核心结构
@dataclass
class GPTConfig:
    sequence_len: int = 2048    # 上下文长度
    vocab_size: int = 32768     # 词表大小
    n_layer: int = 12           # 层数 (深度=12即GPT-1规模)
    n_head: int = 6             # 注意力头数
    n_kv_head: int = 6          # KV头数 (GQA)
    n_embd: int = 768           # 隐藏维度
    window_pattern: str = "SSSL" # 滑动窗口模式

class CausalSelfAttention(nn.Module):
    def forward(self, x, ve, cos_sin, window_size, kv_cache):
        # Flash Attention 3 / SDPA 自动切换
        y = flash_attn.flash_attn_func(q, k, v, causal=True, window_size=window_size)

class GPT(nn.Module):
    def estimate_flops(self):
        """计算FLOPs用于MFU统计"""
    def num_scaling_params(self):
        """返回参数量用于scaling law分析"""
```

---

### 阶段三：数据加载 (DataLoader)

| 文件 | 功能 | 关键点 |
|------|------|--------|
| `nanochat/dataset.py` | 数据下载 | FineWeb-EDU |
| `nanochat/dataloader.py` | 分布式加载 | BOS对齐+BestFit |

```python
# dataloader.py - BOS对齐的BestFit打包
def tokenizing_distributed_data_loader_with_state_bos_bestfit(
    tokenizer, B, T, split, ...
):
    """
    每行以BOS开头，使用BestFit算法最小化裁剪
    - 100%利用率 (无padding)
    - ~35% tokens被裁剪
    """
```

---

### 阶段四：优化器 (Optimizer)

| 文件 | 功能 | 关键点 |
|------|------|--------|
| `nanochat/optim.py` | MuonAdamW优化器 | 混合优化策略 |

**优化策略**:
- **AdamW**: 用于1D参数 (embedding, bias, norm)
- **Muon**: 用于2D矩阵参数 (attention, MLP权重)

```python
# optim.py 核心代码
class MuonAdamW:
    """
    Muon = MomentUm Orthogonalized by Newton-schulz
    - 使用Polar Express加速Newton-Schulz迭代
    - 比AdamW收敛更快
    """
    
class DistMuonAdamW:
    """分布式版本 - 优化AllReduce通信"""
```

---

### 阶段五：预训练 (Pretraining)

| 文件 | 功能 | 关键点 |
|------|------|--------|
| `scripts/base_train.py` | 预训练入口 | torchrun分布式 |
| `nanochat/checkpoint_manager.py` | 检查点管理 | 保存/加载 |

```bash
# 运行预训练 (8×H100, ~3小时)
torchrun --nproc_per_node=8 -m scripts.base_train -- \
    --depth=24 \
    --run="speedrun" \
    --target-flops=4.1e19
```

```python
# base_train.py 训练循环核心
for step in range(num_iterations):
    # 1. 获取数据
    x, y = next(train_loader)
    
    # 2. 前向+反向 (梯度累积)
    for micro_step in range(grad_accum_steps):
        with autocast_ctx:
            loss = model(x, y)
        loss.backward()
    
    # 3. 优化器更新
    optimizer.step()
    
    # 4. 评估 (CORE score, BPB)
    if step % eval_every == 0:
        evaluate_core(model, tokenizer)
```

---

### 阶段六：SFT微调 (Supervised Fine-Tuning)

| 文件 | 功能 | 关键点 |
|------|------|--------|
| `scripts/chat_sft.py` | SFT训练 | TaskMixture数据 |
| `tasks/*.py` | 任务数据集 | SmolTalk, MMLU等 |

```python
# chat_sft.py - 数据混合
train_dataset = TaskMixture([
    SmolTalk(split="train"),        # 460K通用对话
    MMLU(subset="auxiliary_train"), # 100K选择题
    GSM8K(split="train"),           # 数学问题
    SpellingBee(),                  # 拼写任务
    CustomJSON("identity.jsonl"),   # 自定义身份
])
```

---

### 阶段七：强化学习 (RL)

| 文件 | 功能 | 关键点 |
|------|------|--------|
| `scripts/chat_rl.py` | RL训练 | 策略梯度 |
| `nanochat/execution.py` | 代码执行 | Calculator工具 |

---

### 阶段八：推理服务 (Inference)

| 文件 | 功能 | 关键点 |
|------|------|--------|
| `nanochat/engine.py` | 推理引擎 | KVCache + 采样 |
| `scripts/chat_web.py` | Web服务 | FastAPI + WorkerPool |
| `scripts/chat_cli.py` | CLI对话 | 命令行交互 |
| `nanochat/ui.html` | 前端界面 | ChatGPT风格UI |

```python
# engine.py 核心组件
class KVCache:
    """FA3风格KV缓存 - (B, T, H, D)格式"""
    
class Engine:
    def generate(self, tokens, num_samples=1, max_tokens=None, 
                 temperature=1.0, top_k=None, seed=42):
        """自回归生成 - Prefill + Decode循环"""

# chat_web.py - 多GPU数据并行
class WorkerPool:
    """每个GPU一个Worker，负载均衡分发请求"""
```

```bash
# 启动Web服务
python -m scripts.chat_web
# 访问 http://localhost:8000
```

---

### 阶段九：评估 (Evaluation)

| 文件 | 功能 | 关键点 |
|------|------|--------|
| `nanochat/core_eval.py` | CORE评分 | DCLM基准 |
| `nanochat/loss_eval.py` | BPB评估 | bits per byte |
| `scripts/chat_eval.py` | 对话评估 | 任务准确率 |

**CORE Score**: 超越GPT-2需达到 > 0.256525

---

## 🎯 核心技术亮点

### 1. 高效训练优化
```
硬件: H100 GPU (比A100快~2x)
软件: Flash Attention 3, torch.compile
算法: Muon优化器, 滑动窗口注意力
数据: FineWeb-EDU高质量语料
```

### 2. 关键超参数
```python
# depth=24 (GPT-2级别)
n_layer = 24
n_head = 8
n_kv_head = 8  
n_embd = 1024
sequence_len = 2048
vocab_size = 32768
total_batch_size = 524288  # 0.5M tokens/step
```

### 3. 训练成本
| 规模 | 深度 | 参数量 | 时间 | 成本 |
|------|------|--------|------|------|
| GPT-1 | d12 | ~100M | ~5分钟 | ~$2 |
| GPT-2 | d24 | ~800M | ~3小时 | ~$73 |
| 更强 | d26+ | ~1B+ | ~42小时 | ~$1000 |

---

## 📁 完整文件清单

```
nanochat/
├── nanochat/                    # 核心库
│   ├── gpt.py                   # GPT模型 (GPTConfig, Block, Attention, MLP)
│   ├── flash_attention.py       # FA3/SDPA统一接口
│   ├── optim.py                 # MuonAdamW优化器
│   ├── tokenizer.py             # BPE分词器
│   ├── dataloader.py            # 分布式数据加载
│   ├── dataset.py               # 数据集工具
│   ├── engine.py                # 推理引擎 (KVCache)
│   ├── execution.py             # 代码执行工具
│   ├── checkpoint_manager.py    # 检查点管理
│   ├── core_eval.py             # CORE评估
│   ├── loss_eval.py             # BPB评估
│   ├── report.py                # 训练报告
│   ├── common.py                # 通用工具
│   └── ui.html                  # Web前端
│
├── scripts/                     # 入口脚本
│   ├── tok_train.py             # 分词器训练
│   ├── tok_eval.py              # 分词器评估
│   ├── base_train.py            # 预训练
│   ├── base_eval.py             # 基座评估
│   ├── chat_sft.py              # SFT微调
│   ├── chat_rl.py               # 强化学习
│   ├── chat_eval.py             # 对话评估
│   ├── chat_web.py              # Web服务
│   └── chat_cli.py              # CLI对话
│
├── tasks/                       # 任务数据集
│   ├── common.py                # Task基类, TaskMixture
│   ├── smoltalk.py              # 通用对话
│   ├── mmlu.py                  # 多领域选择题
│   ├── arc.py                   # 科学选择题
│   ├── gsm8k.py                 # 数学问题
│   ├── humaneval.py             # 代码任务
│   ├── spellingbee.py           # 拼写任务
│   └── customjson.py            # 自定义JSONL
│
└── runs/                        # 训练脚本
    ├── speedrun.sh              # $100 GPT-2训练
    ├── scaling_laws.sh          # Scaling law实验
    ├── miniseries.sh            # 模型系列训练
    └── runcpu.sh                # CPU/MPS运行示例
```

---

## 🚀 快速开始

```bash
# 1. 克隆仓库
git clone https://github.com/karpathy/nanochat
cd nanochat

# 2. 安装依赖
pip install -e .

# 3. 训练GPT-2 (8×H100)
bash runs/speedrun.sh

# 4. 与模型对话
python -m scripts.chat_web
```
 
