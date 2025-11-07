import os, random
from typing import Dict, Iterator, List, Tuple, Any, Optional

from datasets import Dataset, DatasetDict, Features, Value
import torch

# ---------- 读取与解析 ----------
def _iter_story_files(root_dir: str) -> List[str]:
    res = []
    for sub in ("cnn/stories", "dailymail/stories"):
        p = os.path.join(root_dir, sub)
        if os.path.isdir(p):
            for fn in os.listdir(p):
                if fn.endswith(".story"):
                    res.append(os.path.join(p, fn))
    return res

def _read_story(fp: str) -> str:
    with open(fp, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def _parse_article_summary(raw: str) -> Tuple[str, str]:
    lines = [ln.strip() for ln in raw.splitlines()]
    highlights, article_lines, in_high = [], [], False
    for ln in lines:
        if not ln:
            continue
        if ln.lower().startswith("@highlight"):
            in_high = True
            continue
        (highlights if in_high else article_lines).append(ln)
    return " ".join(article_lines).strip(), " ".join(highlights).strip()

def _gen_examples(paths: List[str]) -> Iterator[Dict[str, str]]:
    for fp in paths:
        raw = _read_story(fp)
        art, summ = _parse_article_summary(raw)
        if not art: art = " "
        if not summ: summ = "."
        yield {"article": art, "summary": summ}

# ---------- 构建数据集 ----------
def build_datasets_hf(
    tokenizer,
    data_dir: str,
    max_source_len: int,
    max_target_len: int,
    subset: Optional[int] = None,
    valid_ratio: float = 0.05,
    seed: int = 42,
    cache_dir: Optional[str] = None,
):
    paths = _iter_story_files(data_dir)
    if not paths:
        raise FileNotFoundError(f"{data_dir} 未找到 cnn/stories 或 dailymail/stories")

    random.Random(seed).shuffle(paths)
    if subset is not None:
        paths = paths[:subset]

    n_valid = max(1000, int(len(paths) * valid_ratio))
    valid_paths, train_paths = paths[:n_valid], paths[n_valid:]

    features = Features({"article": Value("string"), "summary": Value("string")})

    train_raw = Dataset.from_generator(lambda: _gen_examples(train_paths), features=features)
    valid_raw = Dataset.from_generator(lambda: _gen_examples(valid_paths), features=features)

    # 并行分词 + 截断；只保留 input_ids / labels 字段
    def _tok(batch):
        src = tokenizer(
            batch["article"], truncation=True, max_length=max_source_len,
        )
        tgt = tokenizer(
            batch["summary"], truncation=True, max_length=max_target_len,
        )
        # 给 labels 末尾补 eos（若 tok 没有 eos 则用 pad 代替）
        eos = tokenizer.eos_token_id or tokenizer.pad_token_id
        labels = [ids + [eos] for ids in tgt["input_ids"]]
        return {"input_ids": src["input_ids"], "labels": labels}

    num_proc = max(1, os.cpu_count() // 2)
    train_tok = train_raw.map(
        _tok, batched=True, remove_columns=train_raw.column_names,
        num_proc=num_proc, load_from_cache_file=True, cache_file_name=None
    )
    valid_tok = valid_raw.map(
        _tok, batched=True, remove_columns=valid_raw.column_names,
        num_proc=num_proc, load_from_cache_file=True, cache_file_name=None
    )

    # 用“python”格式（列表）交给自定义 collate 动态 padding；也可 set_format("torch") 走定长
    train_tok = train_tok.with_format("python")
    valid_tok = valid_tok.with_format("python")

    return train_tok, valid_tok

# ---------- 动态 padding 的 collate ----------
def hf_collate_fn(examples: List[Dict[str, Any]], tokenizer) -> Dict[str, torch.Tensor]:
    pad_id = tokenizer.pad_token_id
    # 取出
    src = [e["input_ids"] for e in examples]
    tgt = [e["labels"] for e in examples]
    # 动态 padding（最长对齐）
    def _pad_to_max_len(seqs, pad_val):
        m = max(len(s) for s in seqs)
        return [s + [pad_val] * (m - len(s)) for s in seqs]
    src = _pad_to_max_len(src, pad_id)
    tgt = _pad_to_max_len(tgt, pad_id)
    # decoder 输入（左移并加 bos==pad）
    bos_id = pad_id
    dec_in = [[bos_id] + y[:-1] for y in tgt]
    return {
        "src_tokens": torch.tensor(src, dtype=torch.long),
        "tgt_in": torch.tensor(dec_in, dtype=torch.long),
        "labels": torch.tensor(tgt, dtype=torch.long),
    }
