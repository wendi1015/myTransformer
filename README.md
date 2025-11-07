# myTransformer

本项目基于 **PyTorch** 实现标准的 Encoder–Decoder **Transformer**，  
用于 CNN/DailyMail 数据集的新闻摘要任务，支持多头注意力、RoPE 编码、  
多 GPU 训练、早停与结果复现。

---

## ⚙️ 环境与硬件要求

- Python ≥ 3.9  
- CUDA ≥ 11.8, NVIDIA 驱动 ≥ 525  
- GPU：4张A4000  

安装依赖：
```bash
pip install -r requirements.txt
````

`requirements.txt`：

```text
torch>=2.1.0
transformers>=4.40.0
datasets>=2.20.0
rouge-score>=0.1.2
matplotlib>=3.8.0
tqdm>=4.66.0
numpy>=1.24.0
pandas>=2.2.0
```

---

## 📂 项目结构

```
project_root/
├── data_hf.py        # 数据加载与预处理
├── myTransformer.py  # 模型结构（Encoder–Decoder）
├── train.py          # 训练与验证逻辑
├── run.sh            # 多卡运行脚本
│
├── data/             # CNN/DailyMail 数据集
├── tokenizer/        # 分词器文件
└── outputs/          # 模型与日志输出
```

---

## 🚀 快速开始

### 多卡分布式训练

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 train.py \
  --tokenizer_dir ./tokenizer --data_dir ./data \
  --output_dir ./outputs/multigpu --use_rope \
  --N 4 --num_heads 8 --d_model 256 --seed 42
```

---

## 🔁 复现实验

* 固定随机数种子：`--seed 42`
* 禁用 TF32，启用确定性算法：

  ```bash
  export CUBLAS_WORKSPACE_CONFIG=:4096:8
  ```
* 数据划分比例固定：训练集 95%，验证集 5%。

训练结果（loss 与 ROUGE 曲线）保存在：

```
outputs/training_metrics.png
```

最佳模型参数：

```
outputs/checkpoints/best.ckpt
```

---

## 📈 消融实验示例

```bash
bash run.sh --N 8 --num_heads 8 --d_model 256 \
            --output_dir ./outputs/N8_H8_D256 --seed 42
```

---

