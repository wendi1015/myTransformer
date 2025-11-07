import os, math, argparse, csv, random, time
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from transformers import AutoTokenizer
from rouge_score import rouge_scorer

from myTransformer import Transformer
from data_hf import build_datasets_hf, hf_collate_fn


# ----------------------------
# DDP helpers
# ----------------------------
def ddp_initialized():
    return dist.is_available() and dist.is_initialized()

def is_main_process():
    return (not ddp_initialized()) or dist.get_rank() == 0


# ----------------------------
# Deterministic seeds (multi-GPU safe)
# ----------------------------
def set_seed_deterministic(seed: int, rank: int = 0):
    """
    为多卡复现做的严格随机控制：
    - 各进程使用 seed+rank，避免 RNG 状态冲突；
    - 关闭所有非确定性算法路径；
    - 禁用 TF32，固定 cuDNN 算法选择。
    """
    base = int(seed) + int(rank)

    # Python / NumPy
    random.seed(base)
    np.random.seed(base)

    # PyTorch CPU/GPU
    torch.manual_seed(base)
    torch.cuda.manual_seed_all(base)

    # cuDNN & 算法确定性
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # 禁用 TF32，避免数值路径差异
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    # 强制确定性算法（遇到不确定算子会给出警告/异常）
    torch.use_deterministic_algorithms(True, warn_only=True)


def make_worker_init_fn(base_seed: int):
    """
    DataLoader worker 的初始化，用于固定 Python/NumPy/Torch 的子 RNG。
    注意：DataLoader 也会配合一个显式的 torch.Generator。
    """
    def _init_fn(worker_id: int):
        s = base_seed + worker_id
        random.seed(s)
        np.random.seed(s)
        torch.manual_seed(s)
    return _init_fn


def make_generator(base_seed: int):
    g = torch.Generator(device='cpu')
    g.manual_seed(base_seed)
    return g


# ----------------------------
# 可视化
# ----------------------------
def plot_metrics(log_path, output_dir):
    epochs, tr, va, r1, r2, rL = [], [], [], [], [], []
    with open(log_path) as f:
        rd = csv.DictReader(f)
        for row in rd:
            # 跳过 TOTAL 行
            if str(row["epoch"]).strip().upper() == "TOTAL":
                continue
            epochs.append(int(row["epoch"]))
            tr.append(float(row["train_loss"]))
            va.append(float(row["val_loss"]))
            v1 = row["rouge1"]; v2 = row["rouge2"]; vL = row["rougeL"]
            r1.append(float(v1) if v1 != "" else 0.0)
            r2.append(float(v2) if v2 != "" else 0.0)
            rL.append(float(vL) if vL != "" else 0.0)

    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    plt.plot(epochs, tr, label="Train Loss")
    plt.plot(epochs, va, label="Val Loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend(); plt.title("Training vs Validation Loss")

    plt.subplot(1,2,2)
    plt.plot(epochs, r1, label="ROUGE-1")
    plt.plot(epochs, r2, label="ROUGE-2")
    plt.plot(epochs, rL, label="ROUGE-L")
    plt.xlabel("Epoch"); plt.ylabel("Score"); plt.legend(); plt.title("ROUGE Scores")

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    out_png = os.path.join(output_dir, "training_metrics.png")
    plt.savefig(out_png)
    if is_main_process():
        print(f"[plot] Saved -> {out_png}")


# ----------------------------
# 生成
# ----------------------------
@torch.no_grad()
def greedy_generate(model, tokenizer, src_tokens, max_new_tokens, min_len=0, no_repeat_ngram_size=3):
    mdl = model.module if hasattr(model, "module") else model
    mdl.eval()
    device = next(mdl.parameters()).device
    src_tokens = src_tokens.to(device, non_blocking=True)

    pad_id = tokenizer.pad_token_id
    bos_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else pad_id
    if tokenizer.eos_token_id is not None:
        eos_id = tokenizer.eos_token_id
    elif getattr(tokenizer, "sep_token_id", None) is not None:
        eos_id = tokenizer.sep_token_id
    else:
        eos_id = pad_id

    memory, src_mask = mdl.encoder(src_tokens)
    B = src_tokens.size(0)
    ys = torch.full((B, 1), bos_id, dtype=torch.long, device=device)
    finished = torch.zeros(B, dtype=torch.bool, device=device)

    for _ in range(max_new_tokens):
        logits = mdl.decoder(ys, memory, src_mask)[:, -1, :]

        cur_len_wo_bos = ys.size(1) - 1
        if cur_len_wo_bos < min_len:
            logits[:, eos_id] = -1e9
        if ys.size(1) == 1:
            # 首步屏蔽 bos/pad/unk，避免立即结束或自环
            for bad_id in [bos_id, tokenizer.pad_token_id, getattr(tokenizer, "unk_token_id", None)]:
                if bad_id is not None:
                    logits[:, bad_id] = -1e9

        if no_repeat_ngram_size > 0 and ys.size(1) >= no_repeat_ngram_size:
            n = no_repeat_ngram_size
            for b in range(B):
                if finished[b]:
                    continue
                toks = ys[b].tolist()
                grams = {}
                for i in range(len(toks) - n + 1):
                    prev = tuple(toks[i:i+n-1]); nxt = toks[i+n-1]
                    grams.setdefault(prev, set()).add(nxt)
                prev = tuple(toks[-(n-1):])
                if prev in grams and len(grams[prev]) > 0:
                    logits[b, list(grams[prev])] = -1e9

        next_tok = torch.argmax(logits, dim=-1, keepdim=True)
        ys = torch.cat([ys, next_tok], dim=1)
        finished |= (next_tok.squeeze(1) == eos_id)
        if torch.all(finished):
            break

    gen_ids = ys[:, 1:]
    return tokenizer.batch_decode(gen_ids.tolist(), skip_special_tokens=True)


@torch.no_grad()
def sample_generate(model, tokenizer, src_tokens, max_new_tokens,
                    temperature=1.0, top_k=50, top_p=0.95,
                    min_len=0, no_repeat_ngram_size=3):
    mdl = model.module if hasattr(model, "module") else model
    mdl.eval()
    device = next(mdl.parameters()).device
    src_tokens = src_tokens.to(device, non_blocking=True)

    pad_id = tokenizer.pad_token_id
    bos_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else pad_id
    if tokenizer.eos_token_id is not None:
        eos_id = tokenizer.eos_token_id
    elif getattr(tokenizer, "sep_token_id", None) is not None:
        eos_id = tokenizer.sep_token_id
    else:
        eos_id = pad_id

    memory, src_mask = mdl.encoder(src_tokens)
    B = src_tokens.size(0)
    ys = torch.full((B, 1), bos_id, dtype=torch.long, device=device)
    finished = torch.zeros(B, dtype=torch.bool, device=device)

    for _ in range(max_new_tokens):
        logits = mdl.decoder(ys, memory, src_mask)[:, -1, :]

        cur_len_wo_bos = ys.size(1) - 1
        if cur_len_wo_bos < min_len:
            logits[:, eos_id] = -1e9
        if ys.size(1) == 1:
            for bad_id in [bos_id, tokenizer.pad_token_id, getattr(tokenizer, "unk_token_id", None)]:
                if bad_id is not None:
                    logits[:, bad_id] = -1e9

        if no_repeat_ngram_size > 0 and ys.size(1) >= no_repeat_ngram_size:
            n = no_repeat_ngram_size
            for b in range(B):
                if finished[b]:
                    continue
                toks = ys[b].tolist()
                grams = {}
                for i in range(len(toks) - n + 1):
                    prev = tuple(toks[i:i+n-1]); nxt = toks[i+n-1]
                    grams.setdefault(prev, set()).add(nxt)
                prev = tuple(toks[-(n-1):])
                if prev in grams and len(grams[prev]) > 0:
                    logits[b, list(grams[prev])] = -1e9

        logits = logits / max(temperature, 1e-6)
        probs = torch.softmax(logits, dim=-1)

        if top_k and top_k > 0:
            top_vals, top_idx = torch.topk(probs, k=min(top_k, probs.size(-1)))
            mask = torch.zeros_like(probs).scatter(1, top_idx, 1.0)
            probs = probs * mask
            probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)

        if top_p and 0.0 < top_p < 1.0:
            sorted_probs, sorted_idx = torch.sort(probs, dim=-1, descending=True)
            cumsum = torch.cumsum(sorted_probs, dim=-1)
            cutoff = (cumsum > top_p).float()
            cutoff[..., 1:] = cutoff[..., :-1].clone()
            cutoff[..., 0] = 0
            sorted_probs = sorted_probs * (1 - cutoff)
            sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            next_sorted = torch.multinomial(sorted_probs, num_samples=1)
            next_tok = sorted_idx.gather(1, next_sorted)
        else:
            next_tok = torch.multinomial(probs, num_samples=1)

        ys = torch.cat([ys, next_tok], dim=1)
        finished |= (next_tok.squeeze(1) == eos_id)
        if torch.all(finished):
            break

    gen_ids = ys[:, 1:]
    return tokenizer.batch_decode(gen_ids.tolist(), skip_special_tokens=True)


def compute_rouge(pred_texts, ref_texts):
    scorer = rouge_scorer.RougeScorer(["rouge1","rouge2","rougeL"], use_stemmer=True)
    agg = {"rouge1":0.0,"rouge2":0.0,"rougeL":0.0}
    n = len(pred_texts)
    for p, r in zip(pred_texts, ref_texts):
        s = scorer.score(r, p)
        for k in agg: agg[k] += s[k].fmeasure
    for k in agg: agg[k] /= max(n,1)
    return agg


# ----------------------------
# 主训练
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer_dir", required=True)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--output_dir", default="./outputs")

    # 模型结构
    ap.add_argument("--use_rope", action="store_true")
    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--N", type=int, default=4)
    ap.add_argument("--num_heads", type=int, default=8)
    ap.add_argument("--d_ff", type=int, default=1024)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--label_smoothing", type=float, default=0.0)
    ap.add_argument("--freeze_encoder", action="store_true")
    ap.add_argument("--freeze_decoder", action="store_true")

    # 训练
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--scheduler", type=str, default="linear", choices=["none","linear","cosine"])
    ap.add_argument("--warmup_ratio", type=float, default=0.05)
    ap.add_argument("--clip_norm", type=float, default=1.0)
    ap.add_argument("--grad_accum", type=int, default=1)
    ap.add_argument("--subset", type=int, default=30000)
    ap.add_argument("--max_source_len", type=int, default=512)
    ap.add_argument("--max_target_len", type=int, default=160)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=16)  # per-GPU
    ap.add_argument("--seed", type=int, default=42)

    # 评估与早停
    ap.add_argument("--eval_every", type=int, default=1)
    ap.add_argument("--eval_samples", type=int, default=256)
    ap.add_argument("--disable_rouge", action="store_true")
    ap.add_argument("--eval_max_new_tokens", type=int, default=128)
    ap.add_argument("--min_gen_len", type=int, default=40)
    ap.add_argument("--no_repeat_ngram_size", type=int, default=3)
    ap.add_argument("--decode_for_eval", type=str, default="greedy", choices=["greedy","sample"])
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_k", type=int, default=50)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--eval_skip_epochs", type=int, default=0)
    ap.add_argument("--early_stop_metric", type=str, default="val_loss", choices=["val_loss","rougeL"])
    ap.add_argument("--early_stop_patience", type=int, default=3)

    # DDP
    ap.add_argument("--local_rank", type=int, default=-1)

    # Loader
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--prefetch_factor", type=int, default=4)

    # 检查点
    ap.add_argument("--ckpt_dir", type=str, default=None)

    args = ap.parse_args()

    # --- 初始化 DDP ---
    ddp = False
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl", init_method="env://")
        ddp = True
    rank = dist.get_rank() if ddp_initialized() else 0
    local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank if args.local_rank != -1 else 0))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    # 设定可复现随机源（每进程 seed 不同）
    set_seed_deterministic(args.seed, rank=rank)

    if is_main_process():
        os.makedirs(args.output_dir, exist_ok=True)
        if args.ckpt_dir is None:
            args.ckpt_dir = os.path.join(args.output_dir, "checkpoints")
        os.makedirs(args.ckpt_dir, exist_ok=True)

    log_path = os.path.join(args.output_dir, "metrics_log.csv")
    if is_main_process():
        with open(log_path, "w", newline="") as f:
            csv.DictWriter(
                f,
                fieldnames=["epoch","train_loss","val_loss","rouge1","rouge2","rougeL","epoch_time_sec","total_time_sec"]
            ).writeheader()

    # 分词器
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_dir, local_files_only=True, use_fast=False, legacy=False
        )
    except TypeError:
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_dir, local_files_only=True, use_fast=False
        )
    pad_id = tokenizer.pad_token_id

    # 数据集
    train_ds, valid_ds = build_datasets_hf(
        tokenizer, args.data_dir, args.max_source_len, args.max_target_len,
        subset=args.subset, valid_ratio=0.05, seed=args.seed, cache_dir=None
    )
    collate = lambda batch: hf_collate_fn(batch, tokenizer)

    # DataLoader
    base_worker_seed = args.seed * 1000 + rank * 10_000
    loader_gen = make_generator(base_worker_seed)
    worker_init = make_worker_init_fn(base_worker_seed)

    # Sampler / Loader
    train_sampler = DistributedSampler(
        train_ds, shuffle=True, drop_last=True, seed=args.seed
    ) if ddp else None

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size,
        shuffle=(train_sampler is None), drop_last=True,
        collate_fn=collate, sampler=train_sampler,
        pin_memory=True, num_workers=args.num_workers,
        persistent_workers=True, prefetch_factor=args.prefetch_factor,
        generator=loader_gen, worker_init_fn=worker_init
    )
    valid_loader = DataLoader(
        valid_ds, batch_size=args.batch_size, shuffle=False, drop_last=False,
        collate_fn=collate, sampler=None,
        pin_memory=True, num_workers=args.num_workers,
        persistent_workers=True, prefetch_factor=args.prefetch_factor,
        generator=loader_gen, worker_init_fn=worker_init
    )

    # 模型
    model = Transformer(
        src_vocab=tokenizer.vocab_size, tgt_vocab=tokenizer.vocab_size,
        d_model=args.d_model, N=args.N, num_heads=args.num_heads, d_ff=args.d_ff,
        pad_idx=pad_id, dropout=args.dropout, use_rope=args.use_rope, rope_max_len=4096
    ).to(device)

    # 冻结
    if args.freeze_encoder:
        for p in model.encoder_net.parameters(): p.requires_grad = False
    if args.freeze_decoder:
        for p in model.decoder_net.parameters(): p.requires_grad = False
    trainable_params = [p for p in model.parameters() if p.requires_grad]

    # 优化器 / DDP
    criterion = nn.CrossEntropyLoss(ignore_index=pad_id, label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, betas=(0.9,0.98), eps=1e-9, weight_decay=args.weight_decay)
    if ddp:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    # 调度器
    num_update_steps_per_epoch = max(len(train_loader), 1) // max(args.grad_accum, 1)
    num_train_steps = max(num_update_steps_per_epoch * args.epochs, 1)
    warmup_steps = int(args.warmup_ratio * num_train_steps)

    def _sched_linear(step):
        if step < warmup_steps: return (step + 1) / max(1, warmup_steps)
        return 1.0
    def _sched_cosine(step):
        if step < warmup_steps: return (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, (num_train_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    if args.scheduler == "linear":
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_sched_linear)
    elif args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_sched_cosine)
    else:
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: 1.0)

    # 早停
    monitor_metric = args.early_stop_metric
    if args.disable_rouge and monitor_metric != "val_loss":
        monitor_metric = "val_loss"
    mode = "min" if monitor_metric == "val_loss" else "max"
    best_score = float("inf") if mode == "min" else -float("inf")
    patience = args.early_stop_patience
    bad_epochs = 0
    def _is_better(curr, best): return (curr < best) if mode == "min" else (curr > best)

    # 检查点工具
    def _unwrap(m): return m.module if hasattr(m, "module") else m
    def save_checkpoint(tag, epoch, best_score_val):
        if not is_main_process(): return
        os.makedirs(args.ckpt_dir, exist_ok=True)
        ckpt_path = os.path.join(args.ckpt_dir, f"{tag}.ckpt")
        state = {
            "epoch": epoch,
            "model": _unwrap(model).state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_score": best_score_val,
            "args": vars(args),
            "rng_state": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all(),
            },
        }
        torch.save(state, ckpt_path)
        print(f"[ckpt] Saved -> {ckpt_path}")

    # 训练循环
    global_step = 0
    stop_flag = False
    total_start = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        if stop_flag: break
        epoch_start = time.perf_counter()

        # DDP Sampler 的 epoch 种子
        if ddp and isinstance(train_loader.sampler, DistributedSampler):
            train_loader.sampler.set_epoch(epoch)

        # 训练
        running = 0.0
        if is_main_process():
            pbar = tqdm(total=len(train_loader), desc=f"Epoch {epoch}/{args.epochs} [train]", leave=False)
        model.train()
        optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(train_loader, 1):
            src = batch["src_tokens"].to(device, non_blocking=True)
            dec_in = batch["tgt_in"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            logits = model(src, dec_in)
            loss = criterion(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
            loss = loss / max(args.grad_accum, 1)

            if not torch.isfinite(loss):
                if is_main_process(): print("[WARN] non-finite loss detected. skip step.")
                optimizer.zero_grad(set_to_none=True)
                continue

            loss.backward()

            if step % args.grad_accum == 0:
                if args.clip_norm and args.clip_norm > 0:
                    params = (model.module.parameters() if hasattr(model,"module") else model.parameters())
                    torch.nn.utils.clip_grad_norm_(params, args.clip_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                global_step += 1

            running += loss.item() * max(args.grad_accum, 1)
            if is_main_process():
                pbar.set_postfix(loss=f"{(running/step):.4f}",
                                 lr=f"{optimizer.param_groups[0]['lr']:.2e}")
                pbar.update(1)
        if is_main_process():
            pbar.close()
        train_loss = running / max(len(train_loader), 1)

        # 验证 loss
        val_loss = 0.0
        if is_main_process():
            val_sum = 0.0
            pbar_v = tqdm(valid_loader, desc=f"Epoch {epoch}/{args.epochs} [eval]", leave=False)
            with torch.no_grad():
                model.eval()
                for vb in pbar_v:
                    src = vb["src_tokens"].to(device, non_blocking=True)
                    dec_in = vb["tgt_in"].to(device, non_blocking=True)
                    labels = vb["labels"].to(device, non_blocking=True)
                    logits = model(src, dec_in)
                    loss = criterion(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
                    val_sum += loss.item()
                    pbar_v.set_postfix(val_loss=f"{loss.item():.4f}")
            val_loss = val_sum / max(len(valid_loader), 1)

        # ROUGE
        rouge = {"rouge1": None, "rouge2": None, "rougeL": None}
        if is_main_process() and (not args.disable_rouge) and (epoch % args.eval_every == 0) and (epoch > args.eval_skip_epochs):
            pred_texts, ref_texts, count = [], [], 0
            pbar_r = tqdm(valid_loader, desc=f"Epoch {epoch}/{args.epochs} [gen+rouge]", leave=False)
            for vb in pbar_r:
                src = vb["src_tokens"][:8]
                labels = vb["labels"][:8]
                refs = tokenizer.batch_decode(labels.tolist(), skip_special_tokens=True)
                max_new = min(args.eval_max_new_tokens, args.max_target_len)
                if args.decode_for_eval == "greedy":
                    preds = greedy_generate(
                        model, tokenizer, src,
                        max_new_tokens=max_new,
                        min_len=args.min_gen_len,
                        no_repeat_ngram_size=args.no_repeat_ngram_size
                    )
                else:
                    preds = sample_generate(
                        model, tokenizer, src,
                        max_new_tokens=max_new,
                        temperature=args.temperature, top_k=args.top_k, top_p=args.top_p,
                        min_len=args.min_gen_len, no_repeat_ngram_size=args.no_repeat_ngram_size
                    )
                pred_texts.extend(preds); ref_texts.extend(refs)
                count += len(preds)
                pbar_r.set_postfix(collected=count)
                if count >= args.eval_samples:
                    break
            rouge = compute_rouge(pred_texts, ref_texts)

        # 统计本 epoch 用时
        epoch_time = time.perf_counter() - epoch_start

        # 打印与记录
        if is_main_process():
            r1 = "" if rouge["rouge1"] is None else f"{rouge['rouge1']:.4f}"
            r2 = "" if rouge["rouge2"] is None else f"{rouge['rouge2']:.4f}"
            rL = "" if rouge["rougeL"] is None else f"{rouge['rougeL']:.4f}"
            print(f"Epoch {epoch} | TrainLoss {train_loss:.4f} | ValLoss {val_loss:.4f} "
                  f"| R1 {r1} | R2 {r2} | RL {rL} | epoch_time {epoch_time:.2f}s")
            with open(log_path, "a", newline="") as f:
                csv.DictWriter(
                    f,
                    fieldnames=["epoch","train_loss","val_loss","rouge1","rouge2","rougeL","epoch_time_sec","total_time_sec"]
                ).writerow({
                    "epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                    "rouge1": "" if rouge["rouge1"] is None else rouge["rouge1"],
                    "rouge2": "" if rouge["rouge2"] is None else rouge["rouge2"],
                    "rougeL": "" if rouge["rougeL"] is None else rouge["rougeL"],
                    "epoch_time_sec": epoch_time, "total_time_sec": ""
                })

        # === 保存检查点 ===
        if is_main_process():
            save_checkpoint("last", epoch, best_score)

        # 早停（判定 + 保存 best）
        if is_main_process():
            current = val_loss if (args.disable_rouge or args.early_stop_metric == "val_loss") else (
                rouge["rougeL"] if rouge["rougeL"] is not None else -float("inf")
            )
            improved = _is_better(current, best_score)
            if improved:
                best_score = current
                bad_epochs = 0
                save_checkpoint("best", epoch, best_score)
            else:
                bad_epochs += 1
                print(f"[EarlyStop] {args.early_stop_metric} 未改善：{bad_epochs}/{patience}")
                if bad_epochs >= patience:
                    print("[EarlyStop] 触发早停。")
                    stop_flag = True

        # 广播 stop_flag
        if ddp:
            flag = torch.tensor([1 if stop_flag else 0], device=device)
            dist.broadcast(flag, src=0)
            stop_flag = bool(flag.item())

    # 总用时
    total_time = time.perf_counter() - total_start

    if is_main_process():
        with open(log_path, "a", newline="") as f:
            csv.DictWriter(
                f,
                fieldnames=["epoch","train_loss","val_loss","rouge1","rouge2","rougeL","epoch_time_sec","total_time_sec"]
            ).writerow({
                "epoch": "TOTAL", "train_loss": "", "val_loss": "",
                "rouge1": "", "rouge2": "", "rougeL": "",
                "epoch_time_sec": "", "total_time_sec": total_time
            })
        plot_metrics(log_path, args.output_dir)
        print(f"[time] total training time: {total_time:.2f}s")
        print(f"[done] logs: {log_path}")

    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
