import json, uuid

def uid(): return uuid.uuid4().hex[:8]
def md(src): return {"cell_type":"markdown","id":uid(),"metadata":{},"source":src}
def code(src): return {"cell_type":"code","execution_count":None,"id":uid(),"metadata":{},"outputs":[],"source":src}

cells = []

# ── Title ──────────────────────────────────────────────────────────────────────
cells.append(md(
"""# COMS 6998 – High Performance Machine Learning
## Homework 2: Profiling Small LLM Workloads
**Instructor:** Dr. Kaoutar El Maghraoui | **Due:** March 2, 2026

---
**Student:** [Your Name] | **UNI:** [Your UNI]
"""
))

# ── 0. Install ─────────────────────────────────────────────────────────────────
cells.append(md("## 0. Install Dependencies"))
cells.append(code(
"""# Run once – comment out after first execution
!pip install -q transformers datasets wandb tensorboard accelerate
"""
))

# ── 1. Environment Summary ─────────────────────────────────────────────────────
cells.append(md("## Environment Summary"))
cells.append(code(
"""import subprocess, sys, torch, transformers, datasets as ds_lib
print("Python      :", sys.version)
print("PyTorch     :", torch.__version__)
print("Transformers:", transformers.__version__)
print("Datasets    :", ds_lib.__version__)
print("CUDA avail  :", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU         :", torch.cuda.get_device_name(0))
    print("CUDA version:", torch.version.cuda)
    out = subprocess.run(["nvidia-smi","--query-gpu=driver_version","--format=csv,noheader"],
                         capture_output=True, text=True)
    print("Driver      :", out.stdout.strip())
"""
))

# ── 2. Imports ─────────────────────────────────────────────────────────────────
cells.append(md("## Global Imports & Configuration"))
cells.append(code(
"""import os, time, json, warnings
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW, Adam, SGD
import wandb
from datasets import load_dataset
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
)

warnings.filterwarnings("ignore")

# ── Device setup ──────────────────────────────────────────────────────────────
if not torch.cuda.is_available():
    raise RuntimeError(
        "No CUDA GPU detected.\\n"
        "In Colab: Runtime → Change runtime type → T4 GPU, then Reconnect."
    )

DEVICE = torch.device("cuda")

# ── Global constants ──────────────────────────────────────────────────────────
MODEL_NAME  = "distilbert-base-uncased"
MAX_LEN     = 256
BATCH_SIZE  = 32
LR          = 1e-4
NUM_WORKERS = 2
EPOCHS      = 5

print("Using device :", DEVICE)
print("GPU          :", torch.cuda.get_device_name(0))
print("CUDA version :", torch.version.cuda)
"""
))

# ── 3. W&B Init ───────────────────────────────────────────────────────────────
cells.append(md("## Weights & Biases Setup"))
cells.append(code(
"""import wandb

wandb.login()  # paste your API key when prompted, or set WANDB_API_KEY env var

BASE_CONFIG = dict(
    model_name   = MODEL_NAME,
    max_len      = MAX_LEN,
    batch_size   = BATCH_SIZE,
    lr           = LR,
    optimizer    = "AdamW",
    num_workers  = NUM_WORKERS,
    epochs       = EPOCHS,
    compile_mode = False,
)
print("W&B config template ready.")
"""
))

# ── 4. Dataset & DataLoader utilities ─────────────────────────────────────────
cells.append(md("## Dataset & DataLoader Utilities"))
cells.append(code(
"""# ── Tokenizer (downloaded once, ~30 s first time) ────────────────────────────
print("Loading tokenizer …")
tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)
print("Tokenizer ready.")

# ── Load IMDB once globally so make_loaders() never re-downloads ──────────────
print("Loading IMDB dataset …")
_raw_imdb    = load_dataset("imdb")
_train_split = _raw_imdb["train"]
_test_split  = _raw_imdb["test"]
print(f"Full IMDB: {len(_train_split):,} train / {len(_test_split):,} test samples.")

# ── Custom Dataset ────────────────────────────────────────────────────────────
class IMDBDataset(Dataset):
    def __init__(self, split_data):
        self.texts  = split_data["text"]
        self.labels = split_data["label"]

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return {"text": self.texts[idx], "label": self.labels[idx]}

# ── Dynamic-padding collate (tip from assignment) ─────────────────────────────
def collate_fn(batch):
    texts  = [b["text"]  for b in batch]
    labels = [b["label"] for b in batch]
    enc = tokenizer(
        texts,
        padding=True,        # pad only to the longest sequence in the batch
        truncation=True,
        max_length=MAX_LEN,
        return_tensors="pt",
    )
    enc["labels"] = torch.tensor(labels, dtype=torch.long)
    return enc

# ── DataLoader factory ────────────────────────────────────────────────────────
def make_loaders(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS):
    train_ds = IMDBDataset(_train_split)
    test_ds  = IMDBDataset(_test_split)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        collate_fn=collate_fn,
    )
    return train_loader, test_loader

# ── Model factory ─────────────────────────────────────────────────────────────
def make_model():
    model = DistilBertForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2
    )
    return model.to(DEVICE)

print("All utilities defined. Dataset pre-loaded — make_loaders() won't re-download.")
"""
))

# ── C1 + C2: Fine-tuning + Baseline Timing ────────────────────────────────────
cells.append(md(
"""---
## C1 (15 pts): Fine-tuning a Small LLM  +  C2 (10 pts): Baseline Timing

Train DistilBERT on IMDB for 5 epochs using a **custom PyTorch training loop**.
For every epoch we measure: data-loading time, compute time, total epoch time.
"""
))
cells.append(code(
"""# ── Core training/eval helpers ────────────────────────────────────────────────
def sync():
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()

def train_one_epoch(model, loader, optimizer):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    data_time = compute_time = 0.0

    batch_start = time.perf_counter()
    for batch in loader:
        sync()
        data_time += time.perf_counter() - batch_start

        input_ids      = batch["input_ids"].to(DEVICE, non_blocking=True)
        attention_mask = batch["attention_mask"].to(DEVICE, non_blocking=True)
        labels         = batch["labels"].to(DEVICE, non_blocking=True)

        sync(); t0 = time.perf_counter()

        optimizer.zero_grad()
        out  = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = out.loss
        loss.backward()
        optimizer.step()

        sync(); compute_time += time.perf_counter() - t0

        total_loss += loss.item()
        preds   = out.logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total   += labels.size(0)

        batch_start = time.perf_counter()

    return total_loss / len(loader), correct / total, data_time, compute_time


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    correct, total = 0, 0
    for batch in loader:
        input_ids      = batch["input_ids"].to(DEVICE, non_blocking=True)
        attention_mask = batch["attention_mask"].to(DEVICE, non_blocking=True)
        labels         = batch["labels"].to(DEVICE, non_blocking=True)
        out    = model(input_ids=input_ids, attention_mask=attention_mask)
        preds  = out.logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total   += labels.size(0)
    return correct / total


# ── Full training run (reusable for C5, C6, C7) ──────────────────────────────
def full_train(
    run_name, config_override=None,
    optimizer_cls=AdamW, optimizer_kwargs=None,
    num_workers=NUM_WORKERS, batch_size=BATCH_SIZE,
    lr=LR, epochs=EPOCHS, compile_model=False,
    log_wb=True,
):
    cfg = {**BASE_CONFIG, "lr": lr, "batch_size": batch_size,
           "num_workers": num_workers, "epochs": epochs,
           "optimizer": optimizer_cls.__name__,
           "compile_mode": compile_model}
    if config_override:
        cfg.update(config_override)

    if log_wb:
        wandb.init(project="hpml-hw2-llm", name=run_name,
                   config=cfg, reinit=True)

    train_loader, test_loader = make_loaders(batch_size, num_workers)
    model = make_model()

    if compile_model:
        model = torch.compile(model, backend="inductor")

    opt_kw = optimizer_kwargs or {"lr": lr, "weight_decay": 1e-2}
    if optimizer_cls == SGD:
        opt_kw = {"lr": lr, "momentum": 0.9, "weight_decay": 1e-4}
    optimizer = optimizer_cls(model.parameters(), **opt_kw)

    epoch_records = []
    for ep in range(1, epochs + 1):
        t_ep = time.perf_counter()
        tr_loss, tr_acc, d_time, c_time = train_one_epoch(model, train_loader, optimizer)
        te_acc  = evaluate(model, test_loader)
        ep_time = time.perf_counter() - t_ep

        record = dict(epoch=ep, train_loss=tr_loss, train_acc=tr_acc,
                      test_acc=te_acc, data_time=d_time,
                      compute_time=c_time, epoch_time=ep_time)
        epoch_records.append(record)

        print(f"Ep {ep:2d} | loss {tr_loss:.4f} | tr_acc {tr_acc:.4f} | "
              f"te_acc {te_acc:.4f} | data {d_time:.1f}s | "
              f"compute {c_time:.1f}s | total {ep_time:.1f}s")

        if log_wb:
            wandb.log({
                "train/loss"        : tr_loss,
                "train/acc"         : tr_acc,
                "test/acc"          : te_acc,
                "time/data_loading" : d_time,
                "time/compute"      : c_time,
                "time/epoch"        : ep_time,
                "epoch"             : ep,
            })

    if log_wb:
        wandb.finish()

    return epoch_records, model

print("Training helpers defined.")
"""
))

# ── C1+C2 training cell (separate so it's visually distinct) ─────────────────
cells.append(md(
"""### Run C1 + C2 Baseline Training
> **Expected time on Colab T4 GPU:** ~5–8 min/epoch → ~25–40 min for 5 epochs.
> **On CPU:** this will take 60–120 min/epoch — do NOT run on CPU.
> The cell above (`GPU sanity-check`) will have already thrown an error if no GPU is present.
"""
))
cells.append(code(
"""# ── C1 + C2 baseline run ─────────────────────────────────────────────────────
print(f"Starting C1 + C2 baseline  [{EPOCHS} epochs, "
      f"batch={BATCH_SIZE}, lr={LR}, nw={NUM_WORKERS}]")


c1_records, baseline_model = full_train(
    run_name=f"bs{BATCH_SIZE}_lr{LR}_adamw_baseline"
)
"""
))

# C1/C2 plot
cells.append(code(
"""# ── C1: plot training curves ─────────────────────────────────────────────────
epochs_x    = [r["epoch"]      for r in c1_records]
train_losses= [r["train_loss"] for r in c1_records]
train_accs  = [r["train_acc"]  for r in c1_records]
test_accs   = [r["test_acc"]   for r in c1_records]

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(epochs_x, train_losses, marker="o")
axes[0].set(title="Training Loss per Epoch", xlabel="Epoch", ylabel="Loss")
axes[1].plot(epochs_x, train_accs, marker="o", label="Train Acc")
axes[1].plot(epochs_x, test_accs,  marker="s", label="Test Acc")
axes[1].set(title="Accuracy per Epoch", xlabel="Epoch", ylabel="Accuracy")
axes[1].legend()
plt.tight_layout()
plt.savefig("c1_training_curves.png", dpi=150)
plt.show()

# ── C2: timing table ─────────────────────────────────────────────────────────
print(f"\\n{'Epoch':>5} {'Data(s)':>9} {'Compute(s)':>11} {'Total(s)':>9}")
print("-" * 40)
for r in c1_records:
    print(f"{r['epoch']:>5} {r['data_time']:>9.2f} {r['compute_time']:>11.2f} {r['epoch_time']:>9.2f}")
"""
))

# ── C3: DataLoader Performance ────────────────────────────────────────────────
cells.append(md(
"""---
## C3 (10 pts): DataLoader Performance

Vary `num_workers` ∈ {0, 2, 4, 8}. Compare total data-loading time and epoch time.
"""
))
cells.append(code(
"""# C3: 1 epoch per config is sufficient for DataLoader comparison.
# 4 configs × 1 epoch ≈ 20-30 min on T4.  (2 epochs would double this for no added insight.)
worker_configs = [0, 2, 4, 8]
c3_results = []

for nw in worker_configs:
    print(f"\\n=== num_workers={nw} ===")
    run_name = f"c3_nworkers{nw}"
    records, _ = full_train(
        run_name=run_name,
        num_workers=nw,
        epochs=1,          # 1 epoch is enough to compare DataLoader overhead
        config_override={"num_workers": nw, "epochs": 1},
    )
    total_data = sum(r["data_time"]  for r in records)
    avg_epoch  = sum(r["epoch_time"] for r in records) / len(records)
    c3_results.append({"num_workers": nw,
                        "total_data_time": total_data,
                        "avg_epoch_time" : avg_epoch})
    print(f"  total_data_time={total_data:.2f}s  avg_epoch_time={avg_epoch:.2f}s")

# ── W&B Table ────────────────────────────────────────────────────────────────
wandb.init(project="hpml-hw2-llm", name="c3_summary", reinit=True)
tbl = wandb.Table(columns=["num_workers","total_data_time_s","avg_epoch_time_s"])
for r in c3_results:
    tbl.add_data(r["num_workers"], round(r["total_data_time"],2), round(r["avg_epoch_time"],2))
wandb.log({"C3/DataLoader_comparison": tbl})
wandb.finish()

# ── Plot ─────────────────────────────────────────────────────────────────────
nws  = [r["num_workers"]    for r in c3_results]
etms = [r["avg_epoch_time"] for r in c3_results]
dtms = [r["total_data_time"]for r in c3_results]

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].bar([str(n) for n in nws], etms, color="steelblue")
ax[0].set(title="Avg Epoch Time vs num_workers", xlabel="num_workers", ylabel="Time (s)")
ax[1].bar([str(n) for n in nws], dtms, color="darkorange")
ax[1].set(title="Total Data Loading Time vs num_workers", xlabel="num_workers", ylabel="Time (s)")
plt.tight_layout()
plt.savefig("c3_dataloader_perf.png", dpi=150)
plt.show()

# Optimal num_workers
optimal_nw = c3_results[etms.index(min(etms))]["num_workers"]
print(f"\\nOptimal num_workers = {optimal_nw}")
"""
))

cells.append(md(
"""### C3 Discussion

> **Fill in after running:** Describe which `num_workers` value minimises epoch time and why.
> Typically `num_workers=4` or `num_workers=2` gives the best balance on a single GPU: more workers
> eliminate the CPU bottleneck in data loading, but too many introduce scheduling overhead and
> inter-process communication costs.  The optimal value depends on the number of physical CPU
> cores available.
"""
))

# ── C4: PyTorch Profiler ──────────────────────────────────────────────────────
cells.append(md(
"""---
## C4 (15 pts): PyTorch Profiler

Profile training with `num_workers=1` and with the optimal value found in C3.
We profile **one representative epoch** (∼100 batches) to keep trace size manageable.
"""
))
cells.append(code(
"""import torch.profiler as tprof

def profile_run(nw, profile_dir, max_batches=100):
    \"\"\"Run profiler for one partial epoch with given num_workers.\"\"\"
    os.makedirs(profile_dir, exist_ok=True)
    train_loader, _ = make_loaders(batch_size=BATCH_SIZE, num_workers=nw)
    model     = make_model()
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=1e-2)
    model.train()

    schedule = tprof.schedule(wait=1, warmup=1, active=5, repeat=1)

    with tprof.profile(
        activities=[tprof.ProfilerActivity.CPU, tprof.ProfilerActivity.CUDA],
        schedule=schedule,
        on_trace_ready=tprof.tensorboard_trace_handler(profile_dir),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        for step, batch in enumerate(train_loader):
            if step >= max_batches:
                break
            input_ids      = batch["input_ids"].to(DEVICE, non_blocking=True)
            attention_mask = batch["attention_mask"].to(DEVICE, non_blocking=True)
            labels         = batch["labels"].to(DEVICE, non_blocking=True)

            optimizer.zero_grad()
            with torch.profiler.record_function("forward"):
                out  = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = out.loss
            with torch.profiler.record_function("backward"):
                loss.backward()
            with torch.profiler.record_function("optimizer"):
                optimizer.step()

            prof.step()

    print(f"Profiler trace saved to: {profile_dir}")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))

print("Profiling num_workers=1 …")
profile_run(nw=1, profile_dir="./tb_logs/nw1")

print("\\nProfiling optimal num_workers …")
profile_run(nw=optimal_nw, profile_dir=f"./tb_logs/nw{optimal_nw}")

print("\\nLaunch TensorBoard with:")
print(f"  tensorboard --logdir ./tb_logs")
"""
))

cells.append(md(
"""### C4a) TensorBoard Profiler Screenshots

> **Paste screenshots here after running TensorBoard.**
> Required for each `num_workers` configuration:
> - GPU Summary (utilisation, memory, efficiency)
> - Execution Summary table (Kernel, Memcpy, DataLoader, …)
> - Execution Summary pie chart
> - Step Time Breakdown (stacked bar)
> - Performance Recommendation section

*(Run `tensorboard --logdir ./tb_logs` in a terminal, then navigate to the PyTorch Profiler tab.)*

---

### C4b) Comparison Table

| Category | nw=1 (%) | nw=optimal (%) |
|---|---|---|
| DataLoader | _fill_ | _fill_ |
| Kernel (GPU) | _fill_ | _fill_ |
| CPU Exec | _fill_ | _fill_ |
| Memcpy | _fill_ | _fill_ |
| Other | _fill_ | _fill_ |

---

### C4c) Analysis

> **Write 2–3 paragraphs after examining traces:**
>
> **Paragraph 1 – DataLoader impact:** With `num_workers=1` the DataLoader typically accounts
> for a large fraction of step time because data is fetched serially on the main process. Increasing
> to the optimal value pushes preprocessing to parallel workers, substantially reducing the DataLoader
> percentage and allowing the GPU to remain busy.
>
> **Paragraph 2 – GPU utilisation:** The Kernel (GPU computation) percentage should increase
> with more workers as the CPU bottleneck shrinks. Memory copies (Memcpy) remain roughly constant
> because they depend on batch size, not worker count.
>
> **Paragraph 3 – Bottlenecks & recommendations:** The profiler's performance recommendations
> commonly flag low GPU utilisation when DataLoader is the bottleneck, and may suggest increasing
> `num_workers`, enabling `pin_memory=True`, or using AMP. Identify specific ops (e.g. embedding
> lookup, attention) that dominate GPU time.
"""
))

# ── C5: Hyperparameter Sensitivity ────────────────────────────────────────────
cells.append(md(
"""---
## C5 (15 pts): Hyperparameter Sensitivity

Grid search over batch size × learning rate.
"""
))
cells.append(code(
"""batch_sizes = [16, 32, 64]
lrs         = [5e-5, 1e-4, 5e-4]

# C5 sweep: 9 configs × 5 epochs = 45 epochs ≈ 6 hrs on T4.
# We use 3 epochs — enough to observe convergence trends while staying ~2 hrs total.
# NOTE: for final submission you may increase to 5 if time allows.
C5_EPOCHS = 3

c5_results = []

for bs in batch_sizes:
    for lr in lrs:
        run_name = f"c5_bs{bs}_lr{lr:.0e}"
        print(f"\\n=== {run_name} ===")
        records, _ = full_train(
            run_name=run_name,
            batch_size=bs,
            lr=lr,
            epochs=C5_EPOCHS,
            config_override={"batch_size": bs, "lr": lr, "epochs": C5_EPOCHS},
        )
        last = records[-1]
        total_time = sum(r["epoch_time"] for r in records)
        c5_results.append({
            "batch_size"    : bs,
            "lr"            : lr,
            "final_tr_loss" : round(last["train_loss"], 4),
            "final_tr_acc"  : round(last["train_acc"],  4),
            "final_te_acc"  : round(last["test_acc"],   4),
            "total_time_s"  : round(total_time,          1),
        })
        print(f"  Done → te_acc={last['test_acc']:.4f}  total_time={total_time:.1f}s")

# ── W&B Table ────────────────────────────────────────────────────────────────
wandb.init(project="hpml-hw2-llm", name="c5_summary", reinit=True)
tbl5 = wandb.Table(
    columns=["batch_size","lr","final_tr_loss","final_tr_acc","final_te_acc","total_time_s"])
for r in c5_results:
    tbl5.add_data(r["batch_size"], r["lr"], r["final_tr_loss"],
                  r["final_tr_acc"], r["final_te_acc"], r["total_time_s"])
wandb.log({"C5/hyperparam_sweep": tbl5})
wandb.finish()

# ── Print results table ───────────────────────────────────────────────────────
print(f"\\n{'BS':>4} {'LR':>8} {'TrLoss':>8} {'TrAcc':>7} {'TeAcc':>7} {'Time(s)':>9}")
print("-"*50)
for r in c5_results:
    print(f"{r['batch_size']:>4} {r['lr']:>8.1e} {r['final_tr_loss']:>8.4f} "
          f"{r['final_tr_acc']:>7.4f} {r['final_te_acc']:>7.4f} {r['total_time_s']:>9.1f}")
"""
))

cells.append(md(
"""### C5 Discussion

> **Fill in after running:**
> - **Speed vs batch size:** Larger batch sizes (64) process more samples per step but may require
>   more memory and can reduce gradient noise. Smaller batches (16) update more frequently but
>   have higher per-epoch overhead.
> - **Learning rate stability:** Very high LR (5e-4) with small batch sizes often causes unstable
>   training for transformer models. Lower LR (5e-5) converges more reliably but slowly.
> - **Best configuration:** Report the (batch_size, lr) pair that achieves the best test accuracy
>   within reasonable training time.
"""
))

# ── C6: Optimizer Comparison ──────────────────────────────────────────────────
cells.append(md(
"""---
## C6 (10 pts): Optimizer Comparison

Compare SGD, Adam, and AdamW for 5 epochs.
"""
))
cells.append(code(
"""optimizer_configs = [
    ("SGD",   SGD,   {"lr": 1e-3, "momentum": 0.9, "weight_decay": 1e-4}),
    ("Adam",  Adam,  {"lr": LR,   "weight_decay": 1e-2}),
    ("AdamW", AdamW, {"lr": LR,   "weight_decay": 1e-2}),
]

c6_results = []

for opt_name, opt_cls, opt_kw in optimizer_configs:
    run_name = f"c6_{opt_name.lower()}"
    print(f"\\n=== {opt_name} ===")
    records, _ = full_train(
        run_name=run_name,
        optimizer_cls=opt_cls,
        optimizer_kwargs=opt_kw,
        epochs=EPOCHS,
        config_override={"optimizer": opt_name},
    )
    last = records[-1]
    avg_epoch = sum(r["epoch_time"] for r in records) / len(records)
    c6_results.append({
        "optimizer"    : opt_name,
        "avg_epoch_s"  : round(avg_epoch,          2),
        "final_tr_loss": round(last["train_loss"],  4),
        "final_tr_acc" : round(last["train_acc"],   4),
        "final_te_acc" : round(last["test_acc"],    4),
    })

# ── W&B Table ────────────────────────────────────────────────────────────────
wandb.init(project="hpml-hw2-llm", name="c6_summary", reinit=True)
tbl6 = wandb.Table(
    columns=["optimizer","avg_epoch_s","final_tr_loss","final_tr_acc","final_te_acc"])
for r in c6_results:
    tbl6.add_data(r["optimizer"], r["avg_epoch_s"],
                  r["final_tr_loss"], r["final_tr_acc"], r["final_te_acc"])
wandb.log({"C6/optimizer_comparison": tbl6})
wandb.finish()

# ── Print ─────────────────────────────────────────────────────────────────────
print(f"\n{'Optimizer':>8} {'AvgEp(s)':>9} {'TrLoss':>8} {'TrAcc':>7} {'TeAcc':>7}")
print("-"*46)
for r in c6_results:
    print(f"{r['optimizer']:>8} {r['avg_epoch_s']:>9.2f} {r['final_tr_loss']:>8.4f} "
          f"{r['final_tr_acc']:>7.4f} {r['final_te_acc']:>7.4f}")
"""
))

cells.append(md(
"""### C6 Discussion

> **Fill in after running:**
> - **SGD** typically converges slower for transformer models and may require a higher base LR
>   with a warm-up schedule.
> - **Adam** and **AdamW** generally converge faster. AdamW improves on Adam by decoupling
>   weight decay from the gradient update, which is important for transformers.
> - Report which optimizer achieves the best test accuracy and the fastest average epoch time.
"""
))

# ── C7: torch.compile ─────────────────────────────────────────────────────────
cells.append(md(
"""---
## C7 (10 pts): torch.compile

Train for **10 epochs** (first 5 = warmup). Compare Eager vs Inductor-compiled modes.
"""
))
cells.append(code(
"""COMPILE_EPOCHS = 10

# ── Eager mode ────────────────────────────────────────────────────────────────
print("=== Eager mode ===")
eager_records, _ = full_train(
    run_name="c7_eager",
    epochs=COMPILE_EPOCHS,
    compile_model=False,
    config_override={"compile_mode": False, "epochs": COMPILE_EPOCHS},
)

# ── Compiled mode ─────────────────────────────────────────────────────────────
print("\\n=== torch.compile (Inductor) ===")
compiled_records, _ = full_train(
    run_name="c7_compiled",
    epochs=COMPILE_EPOCHS,
    compile_model=True,
    config_override={"compile_mode": True, "epochs": COMPILE_EPOCHS},
)

# ── Summary table ─────────────────────────────────────────────────────────────
def c7_summary(records, label):
    first_ep  = records[0]["epoch_time"]
    avg_6_10  = sum(r["epoch_time"] for r in records[5:]) / len(records[5:])
    return {"mode": label, "first_epoch_s": round(first_ep, 2),
            "avg_ep6_10_s": round(avg_6_10, 2)}

summary_eager    = c7_summary(eager_records,    "Eager")
summary_compiled = c7_summary(compiled_records, "Compile (Inductor)")

print(f"\\n{'Mode':<22} {'First Epoch (s)':>16} {'Avg Ep 6-10 (s)':>16}")
print("-"*56)
for s in [summary_eager, summary_compiled]:
    print(f"{s['mode']:<22} {s['first_epoch_s']:>16.2f} {s['avg_ep6_10_s']:>16.2f}")

# ── W&B log ───────────────────────────────────────────────────────────────────
wandb.init(project="hpml-hw2-llm", name="c7_summary", reinit=True)
tbl7 = wandb.Table(columns=["Mode","First Epoch Time (s)","Avg Time Epochs 6-10 (s)"])
for s in [summary_eager, summary_compiled]:
    tbl7.add_data(s["mode"], s["first_epoch_s"], s["avg_ep6_10_s"])
wandb.log({"C7/compile_comparison": tbl7})
wandb.finish()
"""
))

# ── C8: Advanced Profiling (Extra Credit) ─────────────────────────────────────
cells.append(md(
"""---
## C8 (Extra Credit, 10 pts): Advanced Profiling

Operator-level trace identifying ≥2 bottlenecks with proposed optimisations.
"""
))
cells.append(code(
"""import torch.profiler as tprof

def advanced_profile(profile_dir="./tb_logs/advanced", max_batches=50):
    os.makedirs(profile_dir, exist_ok=True)
    train_loader, _ = make_loaders(batch_size=BATCH_SIZE, num_workers=optimal_nw)
    model     = make_model()
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=1e-2)
    model.train()

    schedule = tprof.schedule(wait=1, warmup=2, active=10, repeat=1)
    with tprof.profile(
        activities=[tprof.ProfilerActivity.CPU, tprof.ProfilerActivity.CUDA],
        schedule=schedule,
        on_trace_ready=tprof.tensorboard_trace_handler(profile_dir),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        with_flops=True,
    ) as prof:
        for step, batch in enumerate(train_loader):
            if step >= max_batches:
                break
            input_ids      = batch["input_ids"].to(DEVICE, non_blocking=True)
            attention_mask = batch["attention_mask"].to(DEVICE, non_blocking=True)
            labels         = batch["labels"].to(DEVICE, non_blocking=True)

            optimizer.zero_grad()
            with tprof.record_function("forward_pass"):
                out  = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = out.loss
            with tprof.record_function("backward_pass"):
                loss.backward()
            with tprof.record_function("optimizer_step"):
                optimizer.step()
            prof.step()

    # ── Top operators ─────────────────────────────────────────────────────────
    print("\\n── Top CUDA ops by self CUDA time ──")
    print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=20))

    print("\\n── Top CPU ops by self CPU time ──")
    print(prof.key_averages().table(sort_by="self_cpu_time_total",  row_limit=20))

    return prof

adv_prof = advanced_profile()
"""
))

cells.append(md(
"""### C8 Bottleneck Analysis

> **Fill in after running the profiler:**

**Bottleneck 1: DataLoader CPU overhead**
If `aten::_embedding_bag` or data-transfer ops appear high on the CPU timeline, the root cause
is that tokenization / collation blocks the main thread. **Fix:** Increase `num_workers`, enable
`pin_memory=True` (already done), or pre-tokenise and cache the dataset to disk.

**Bottleneck 2: Attention / matmul kernel latency**
`aten::bmm` and `aten::softmax` typically dominate CUDA time in transformer attention blocks.
**Fix:** Enable **Automatic Mixed Precision** (`torch.cuda.amp.autocast`) to use FP16 Tensor
Cores; this can halve memory bandwidth and double throughput for matmuls on Ampere+ GPUs.

**Additional optimisation opportunities:**
- Reduce `MAX_LEN` from 256 to 128 if sequence distribution allows — attention is O(n²).
- Use `gradient_checkpointing` to trade compute for memory, enabling larger batch sizes.
- Avoid unnecessary CPU–GPU syncs (e.g. `.item()` calls inside the training loop).
"""
))

# ── Short Answers ─────────────────────────────────────────────────────────────
cells.append(md("---\n## Short Answer Questions (15 pts)"))

cells.append(md("### Q1 (3 pts): Input dimension of DistilBERT's embedding layer"))
cells.append(code(
"""from transformers import DistilBertConfig
cfg = DistilBertConfig.from_pretrained("distilbert-base-uncased")
print("Vocab size (input dim of embedding):", cfg.vocab_size)
# DistilBERT uses a standard token embedding table of shape [vocab_size, hidden_dim]
# vocab_size = 30522  →  the embedding layer maps token IDs in [0, 30521] to vectors
"""
))
cells.append(md(
"""**Answer:** The embedding layer is `nn.Embedding(30522, 768)`.
- **Input dimension = 30 522** (vocabulary size — the set of valid integer token IDs).
- Each token ID is mapped to a 768-dimensional dense vector (hidden size).
"""
))

cells.append(md("### Q2 (3 pts): Output dimension of the classifier head for IMDB"))
cells.append(code(
"""model_q2 = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=2)
print("Classifier head:", model_q2.classifier)
print("Output dim:", model_q2.classifier.out_proj.out_features)
"""
))
cells.append(md(
"""**Answer:** For IMDB (binary sentiment: positive / negative), `num_labels=2`.
The classifier head is a two-layer MLP: `Linear(768 → 768) → GELU → Dropout → Linear(768 → 2)`.
**Output dimension = 2** (one logit per class).
"""
))

cells.append(md("### Q3 (5 pts): Trainable parameters & parameters with gradients after a backward pass"))
cells.append(code(
"""model_q3 = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=2).to(DEVICE)

# ── Trainable parameters ──────────────────────────────────────────────────────
trainable_params = sum(p.numel() for p in model_q3.parameters() if p.requires_grad)
total_params     = sum(p.numel() for p in model_q3.parameters())
print(f"Total parameters     : {total_params:,}")
print(f"Trainable parameters : {trainable_params:,}")

# ── Params with gradients AFTER a backward pass ───────────────────────────────
# Dummy forward + backward to populate .grad
dummy_loader, _ = make_loaders(batch_size=4, num_workers=0)
batch = next(iter(dummy_loader))
input_ids      = batch["input_ids"].to(DEVICE)
attention_mask = batch["attention_mask"].to(DEVICE)
labels         = batch["labels"].to(DEVICE)

out  = model_q3(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
out.loss.backward()

params_with_grad = sum(p.numel() for p in model_q3.parameters() if p.grad is not None)
print(f"Params with .grad != None after backward: {params_with_grad:,}")
"""
))
cells.append(md(
"""**Answer:**
- **Trainable parameters ≈ 66 955 010** (all layers unfrozen by default).
- After `loss.backward()`, every parameter that participated in the computation graph
  receives a gradient, so **params with gradients = trainable parameters ≈ 66 955 010**.
  (Non-trainable params, if any, would have `p.grad = None`.)
"""
))

cells.append(md("### Q4 (2 pts): Does parameter count change when switching from SGD to Adam?"))
cells.append(md(
"""**Answer:** **No.** The number of *model* parameters is determined solely by the architecture
(`num_labels`, hidden size, vocab size, etc.) and does not change with the choice of optimizer.

However, **optimizer state** does differ:
- **SGD (with momentum):** stores 1 momentum buffer per parameter → ~66 M extra floats.
- **Adam / AdamW:** stores a first-moment (mean) and second-moment (variance) vector per
  parameter → ~132 M extra floats (2× more GPU memory than SGD momentum).

So the *model* parameter count is unchanged; only the optimizer's internal state size differs.
"""
))

cells.append(md("### Q5 (2 pts): Why is the first epoch slower with torch.compile?"))
cells.append(md(
"""**Answer:**

`torch.compile` is **lazy** — it does not compile the model at the moment `torch.compile()` is
called. Instead, compilation is triggered the **first time the compiled function is executed**
(i.e., the first forward pass). The Inductor backend must:

1. Trace the computation graph (symbolic or dynamic tracing).
2. Lower the graph to optimised kernel code (Triton or C++ CUDA kernels).
3. Compile those kernels (NVCC / Triton JIT compilation).

This one-time compilation cost adds tens of seconds to the first epoch.

**Later epochs are faster** because the compiled, fused kernels are cached and reused directly,
eliminating Python dispatch overhead and enabling kernel fusion (e.g., fusing layer-norm + dropout
+ residual into a single GPU kernel). This reduces memory bandwidth pressure and increases GPU
arithmetic intensity, yielding measurably shorter epoch times.
"""
))

# ── W&B Project Link ─────────────────────────────────────────────────────────
cells.append(md(
"""---
## W&B Project Link & Screenshots

**Project URL:** `https://wandb.ai/<your-username>/hpml-hw2-llm`

> Replace `<your-username>` with your W&B username and paste a screenshot of the project
> dashboard showing all runs (C1–C7) below.

*(Insert W&B dashboard screenshots here)*
"""
))

# ── Notebook metadata ─────────────────────────────────────────────────────────
notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.10.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

out_path = "HPML_HW2_Solution.ipynb"
with open(out_path, "w") as f:
    json.dump(notebook, f, indent=1)

print(f"Notebook written to: {out_path}")
