#!/usr/bin/env python3
"""
HPML Final Exam Comprehensive Study Notes Generator
Generates a detailed, well-formatted PDF study guide from all lecture notes.
"""

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, HRFlowable, KeepTogether
)
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
from reportlab.platypus import ListFlowable, ListItem
import datetime

OUTPUT_PATH = "/Users/rajvardhan/Desktop/Projects/HPML/HPML_Final_Exam_Notes.pdf"

# ── Colour Palette ────────────────────────────────────────────────────────────
C_NAVY    = colors.HexColor("#1a2744")
C_BLUE    = colors.HexColor("#2563eb")
C_LBLUE   = colors.HexColor("#dbeafe")
C_GREEN   = colors.HexColor("#166534")
C_LGREEN  = colors.HexColor("#dcfce7")
C_ORANGE  = colors.HexColor("#c2410c")
C_LORANGE = colors.HexColor("#ffedd5")
C_RED     = colors.HexColor("#991b1b")
C_LRED    = colors.HexColor("#fee2e2")
C_PURPLE  = colors.HexColor("#6b21a8")
C_LPURPLE = colors.HexColor("#f3e8ff")
C_GRAY    = colors.HexColor("#374151")
C_LGRAY   = colors.HexColor("#f3f4f6")
C_MGRAY   = colors.HexColor("#e5e7eb")
C_DGRAY   = colors.HexColor("#9ca3af")
C_BLACK   = colors.HexColor("#111827")
C_WHITE   = colors.white

# ── Style Sheet ───────────────────────────────────────────────────────────────
def build_styles():
    base = getSampleStyleSheet()

    styles = {}

    styles["cover_title"] = ParagraphStyle(
        "cover_title", fontSize=28, fontName="Helvetica-Bold",
        textColor=C_WHITE, alignment=TA_CENTER, leading=36, spaceAfter=10)

    styles["cover_sub"] = ParagraphStyle(
        "cover_sub", fontSize=14, fontName="Helvetica",
        textColor=C_LBLUE, alignment=TA_CENTER, leading=20, spaceAfter=6)

    styles["cover_info"] = ParagraphStyle(
        "cover_info", fontSize=11, fontName="Helvetica",
        textColor=C_LGRAY, alignment=TA_CENTER, leading=16)

    styles["h1"] = ParagraphStyle(
        "h1", fontSize=18, fontName="Helvetica-Bold",
        textColor=C_WHITE, alignment=TA_LEFT, leading=24,
        spaceBefore=14, spaceAfter=8,
        leftIndent=0, borderPadding=(6, 10, 6, 10))

    styles["h2"] = ParagraphStyle(
        "h2", fontSize=13, fontName="Helvetica-Bold",
        textColor=C_NAVY, alignment=TA_LEFT, leading=18,
        spaceBefore=12, spaceAfter=4)

    styles["h3"] = ParagraphStyle(
        "h3", fontSize=11, fontName="Helvetica-Bold",
        textColor=C_BLUE, alignment=TA_LEFT, leading=15,
        spaceBefore=8, spaceAfter=3)

    styles["body"] = ParagraphStyle(
        "body", fontSize=9, fontName="Helvetica",
        textColor=C_BLACK, alignment=TA_JUSTIFY, leading=13,
        spaceBefore=2, spaceAfter=2)

    styles["bullet"] = ParagraphStyle(
        "bullet", fontSize=9, fontName="Helvetica",
        textColor=C_BLACK, alignment=TA_LEFT, leading=13,
        spaceBefore=1, spaceAfter=1,
        leftIndent=12, firstLineIndent=-8)

    styles["sub_bullet"] = ParagraphStyle(
        "sub_bullet", fontSize=8.5, fontName="Helvetica",
        textColor=C_GRAY, alignment=TA_LEFT, leading=12,
        spaceBefore=1, spaceAfter=1,
        leftIndent=24, firstLineIndent=-8)

    styles["code"] = ParagraphStyle(
        "code", fontSize=7.5, fontName="Courier",
        textColor=C_NAVY, alignment=TA_LEFT, leading=11,
        spaceBefore=2, spaceAfter=2,
        leftIndent=16, backColor=C_LGRAY,
        borderPadding=(4, 6, 4, 6))

    styles["formula"] = ParagraphStyle(
        "formula", fontSize=9, fontName="Courier-Bold",
        textColor=C_PURPLE, alignment=TA_CENTER, leading=14,
        spaceBefore=4, spaceAfter=4,
        backColor=C_LPURPLE, borderPadding=(4, 8, 4, 8))

    styles["key_fact"] = ParagraphStyle(
        "key_fact", fontSize=9, fontName="Helvetica-Bold",
        textColor=C_GREEN, alignment=TA_LEFT, leading=13,
        spaceBefore=3, spaceAfter=3,
        leftIndent=10, backColor=C_LGREEN,
        borderPadding=(4, 8, 4, 8))

    styles["warning"] = ParagraphStyle(
        "warning", fontSize=9, fontName="Helvetica-Bold",
        textColor=C_ORANGE, alignment=TA_LEFT, leading=13,
        spaceBefore=3, spaceAfter=3,
        leftIndent=10, backColor=C_LORANGE,
        borderPadding=(4, 8, 4, 8))

    styles["exam_tip"] = ParagraphStyle(
        "exam_tip", fontSize=9, fontName="Helvetica-Bold",
        textColor=C_RED, alignment=TA_LEFT, leading=13,
        spaceBefore=3, spaceAfter=3,
        leftIndent=10, backColor=C_LRED,
        borderPadding=(4, 8, 4, 8))

    styles["toc_entry"] = ParagraphStyle(
        "toc_entry", fontSize=10, fontName="Helvetica",
        textColor=C_NAVY, alignment=TA_LEFT, leading=16,
        leftIndent=16)

    styles["toc_h1"] = ParagraphStyle(
        "toc_h1", fontSize=11, fontName="Helvetica-Bold",
        textColor=C_BLUE, alignment=TA_LEFT, leading=18,
        spaceBefore=4)

    return styles

# ── Helper Builders ───────────────────────────────────────────────────────────
S = None  # set globally after build_styles()

def h1(text, color=C_NAVY):
    bg = Table([[Paragraph(text, S["h1"])]],
               colWidths=[7.5*inch])
    bg.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,-1), color),
        ("ROWBACKGROUNDS", (0,0), (-1,-1), [color]),
        ("TOPPADDING", (0,0), (-1,-1), 6),
        ("BOTTOMPADDING", (0,0), (-1,-1), 6),
        ("LEFTPADDING", (0,0), (-1,-1), 10),
        ("RIGHTPADDING", (0,0), (-1,-1), 10),
        ("ROUNDEDCORNERS", [4]),
    ]))
    return bg

def h2(text): return Paragraph(text, S["h2"])
def h3(text): return Paragraph(text, S["h3"])
def body(text): return Paragraph(text, S["body"])
def bullet(text, indent=0):
    st = S["sub_bullet"] if indent else S["bullet"]
    prefix = "◦ " if indent else "• "
    return Paragraph(prefix + text, st)
def code(text): return Paragraph(text, S["code"])
def formula(text): return Paragraph(text, S["formula"])
def key_fact(text): return Paragraph("✓ " + text, S["key_fact"])
def warning(text): return Paragraph("⚠ " + text, S["warning"])
def exam_tip(text): return Paragraph("★ EXAM: " + text, S["exam_tip"])
def sp(n=1): return Spacer(1, n*0.12*inch)
def hr(): return HRFlowable(width="100%", thickness=0.5, color=C_MGRAY, spaceAfter=4, spaceBefore=4)

def make_table(headers, rows, col_widths=None, header_color=C_NAVY, stripe=True):
    if col_widths is None:
        n = len(headers)
        col_widths = [7.5*inch/n]*n
    data = [[Paragraph(f"<b><font color='white'>{h}</font></b>", S["body"]) for h in headers]]
    for i, row in enumerate(rows):
        data.append([Paragraph(str(c), S["body"]) for c in row])
    t = Table(data, colWidths=col_widths)
    ts = [
        ("BACKGROUND", (0,0), (-1,0), header_color),
        ("GRID", (0,0), (-1,-1), 0.3, C_MGRAY),
        ("TOPPADDING", (0,0), (-1,-1), 4),
        ("BOTTOMPADDING", (0,0), (-1,-1), 4),
        ("LEFTPADDING", (0,0), (-1,-1), 5),
        ("RIGHTPADDING", (0,0), (-1,-1), 5),
        ("VALIGN", (0,0), (-1,-1), "TOP"),
    ]
    if stripe:
        for i in range(1, len(data)):
            if i % 2 == 0:
                ts.append(("BACKGROUND", (0,i), (-1,i), C_LGRAY))
    t.setStyle(TableStyle(ts))
    return t

def section_break(title, color=C_NAVY):
    return [PageBreak(), h1(title, color), sp()]

# ── CONTENT BUILDERS ──────────────────────────────────────────────────────────

def cover_page(story):
    from reportlab.platypus import Image as RLImage
    # Big coloured banner
    banner = Table([
        [Paragraph("HPML", ParagraphStyle("t1",fontSize=52,fontName="Helvetica-Bold",textColor=C_WHITE,alignment=TA_CENTER))],
        [Paragraph("High-Performance Machine Learning", ParagraphStyle("t2",fontSize=18,fontName="Helvetica",textColor=C_LBLUE,alignment=TA_CENTER,leading=24))],
        [Paragraph("Final Exam — Complete Study Notes", ParagraphStyle("t3",fontSize=14,fontName="Helvetica-Bold",textColor=C_LGRAY,alignment=TA_CENTER,leading=20))],
    ], colWidths=[7.5*inch])
    banner.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,-1),C_NAVY),
        ("TOPPADDING",(0,0),(-1,-1),24),
        ("BOTTOMPADDING",(0,0),(-1,-1),24),
        ("LEFTPADDING",(0,0),(-1,-1),20),
        ("RIGHTPADDING",(0,0),(-1,-1),20),
    ]))
    story.append(banner)
    story.append(sp(2))

    info_rows = [
        ["Course", "COMS E6998 — High-Performance Machine Learning"],
        ["University", "Columbia University — Spring 2026"],
        ["Instructor", "Dr. Kaoutar El Maghraoui"],
        ["Exam Date", "Monday, May 11, 2026 · 7:00 PM – 9:00 PM"],
        ["Location", "Schermerhorn Hall, Room 614"],
        ["Format", "Closed book · One cheat sheet (8.5×11, both sides, ≥8pt font)"],
        ["Weight", "20% of final grade · ~25–35 questions"],
        ["Generated", datetime.datetime.now().strftime("%B %d, %Y")],
    ]
    t = Table([[Paragraph(f"<b>{r[0]}</b>",S["body"]), Paragraph(r[1],S["body"])] for r in info_rows],
              colWidths=[1.8*inch,5.7*inch])
    t.setStyle(TableStyle([
        ("GRID",(0,0),(-1,-1),0.3,C_MGRAY),
        ("BACKGROUND",(0,0),(0,-1),C_LGRAY),
        ("TOPPADDING",(0,0),(-1,-1),5),
        ("BOTTOMPADDING",(0,0),(-1,-1),5),
        ("LEFTPADDING",(0,0),(-1,-1),8),
        ("RIGHTPADDING",(0,0),(-1,-1),8),
    ]))
    story.append(t)
    story.append(sp(2))

    # Tier weights
    story.append(h2("Exam Coverage by Tier"))
    tier1 = make_table(
        ["Lecture","Topic","Tier"],
        [["L3","PyTorch Performance and Profiling","Tier 1 (Heavy — ~70%)"],
         ["L4","CUDA Basics","Tier 1 (Heavy — ~70%)"],
         ["L5","Advanced CUDA","Tier 1 (Heavy — ~70%)"],
         ["L6","Distributed Deep Learning","Tier 1 (Heavy — ~70%)"],
         ["L9","Efficient Transformers & FlashAttention","Tier 1 (Heavy — ~70%)"],
         ["L1","Intro HPC & AI (Amdahl, scaling)","Tier 2 (~25%)"],
         ["L2","Performance Methodology & Optimizers","Tier 2 (~25%)"],
         ["L7","Quantization (PTQ vs QAT, INT8/FP8)","Tier 2 (~25%)"],
         ["L8","Pruning & Sparsity","Tier 2 (~25%)"],
         ["L10","Knowledge Distillation","Tier 2 (~25%)"]],
        col_widths=[0.6*inch,3.8*inch,2.1*inch])
    story.append(tier1)
    story.append(PageBreak())


def numbers_cheatsheet(story):
    story += section_break("CRITICAL NUMBERS TO MEMORIZE", C_RED)
    story.append(exam_tip("These numbers appear directly in exam questions. Know them cold."))
    story.append(sp())

    story.append(h2("GPU & CUDA Constants"))
    t = make_table(
        ["Item","Value","Why It Matters"],
        [["Warp size","32 threads","Basic scheduling unit; warp divergence is costly"],
         ["Max threads per block","1024","Cannot exceed when setting blockDim"],
         ["Shared memory banks","32 banks, 4-byte granule","Bank conflicts degrade performance"],
         ["Common block sizes","128, 256, 512","Multiples of 32; maximize occupancy"],
         ["HBM latency (A100)","~400–600 cycles","Motivates caching in shared memory"],
         ["SRAM (shared mem) BW","~19 TB/s","~13× faster than HBM"],
         ["HBM bandwidth (A100)","~1.5–2 TB/s (HBM2e)","Bottleneck for memory-bound kernels"],
         ["HBM bandwidth (H100)","~3.35 TB/s (HBM3)","Still the bottleneck"],
         ["SRAM (on-chip) capacity","~20 MB per GPU","FlashAttention tiles into this"],
         ["HBM capacity (A100/H100)","~40–80 GB","Limits model size per GPU"],
         ["A100 CUDA cores","6,912","Ampere architecture"],
         ["H100 CUDA cores","18,432","Hopper architecture — 2.7× more"],
         ["A100 Tensor cores","432 (3rd gen)","FP16/TF32/INT8"],
         ["H100 Tensor cores","640 (4th gen)","Adds FP8"],
         ["A100 L2 cache","40 MB","Chip-wide shared cache"],
         ["H100 L2 cache","50 MB","Larger for bigger models"]],
        col_widths=[2.2*inch,2.0*inch,3.3*inch])
    story.append(t)
    story.append(sp())

    story.append(h2("Memory Precision Reductions"))
    t = make_table(
        ["Conversion","Memory Reduction","Use Case"],
        [["FP32 → FP16","2×","Training with mixed precision"],
         ["FP32 → BF16","2×","LLM pre-training (same range as FP32)"],
         ["FP32 → INT8","4×","Inference quantization"],
         ["FP32 → INT4","8×","Aggressive inference quantization"],
         ["FP32 → FP8 (E4M3/E5M2)","4×","H100 training (forward/backward)"],
         ["FP32 → FP4","8×","Blackwell (B200) inference"]],
        col_widths=[2.5*inch,2.0*inch,3.0*inch])
    story.append(t)
    story.append(sp())

    story.append(h2("FlashAttention Key Numbers"))
    t = make_table(
        ["Item","Value"],
        [["Standard attention memory","O(N²) — stores full N×N matrix in HBM"],
         ["FlashAttention HBM I/O","O(N²·d²/M) — M = SRAM size; d = head dim"],
         ["FlashAttention FLOPs","O(N²·d) — same as standard attention"],
         ["Wall-clock speedup","2–4× (up to 6× with masking+dropout)"],
         ["Memory reduction","10–20×"],
         ["Block size Bc","⌈M/4d⌉"],
         ["Block size Br","min(⌈M/4d⌉, d)"],
         ["Standard attention runtime (GPT-2 medium)","41.7 ms"],
         ["FlashAttention runtime (GPT-2 medium)","7.3 ms (5.7× faster)"],
         ["HBM R/W standard","40.3 GB"],
         ["HBM R/W FlashAttention","4.4 GB (~9× less)"]],
        col_widths=[3.5*inch,4.0*inch])
    story.append(t)
    story.append(sp())

    story.append(h2("Distributed Training Key Numbers"))
    t = make_table(
        ["Item","Value"],
        [["Ring AllReduce comm cost per GPU","2N(P-1)/P ≈ 2N for large P"],
         ["Parameter Server comm cost","2N(P-1) — scales poorly with P"],
         ["LLaMA 3 405B memory (fp16)","810 GB — needs 10+ H100s just for weights"],
         ["GPT-3 175B memory w/ Adam","~2.8 TB (16 bytes/param)"],
         ["ZeRO-1 max model (N=64 GPUs)","~19B params"],
         ["ZeRO-2 max model (N=64 GPUs)","~36B params"],
         ["ZeRO-3 max model (N=64 GPUs)","~320B params"],
         ["DDP overhead vs ideal","~5–15% from AllReduce"],
         ["TSM video model speedup","50 hrs → 14 min with 1,536 GPUs (211×)"]],
        col_widths=[3.5*inch,4.0*inch])
    story.append(t)


def sec_hpc_intro(story):
    story += section_break("L1 & L2 — HPC Intro, Performance Methodology & Optimizers", C_NAVY)

    story.append(h2("HPML Definition & Course Structure"))
    story.append(body("<b>HPML</b> = The science of making AI <b>Fast, Efficient, and Scalable</b>"))
    story.append(sp())
    t = make_table(
        ["Module","Topics"],
        [["I. Foundations","HPC/AI Intro, PyTorch Fundamentals, Profiling"],
         ["II. GPU Computing","CUDA Basics, Advanced CUDA, Distributed DL"],
         ["III. Compression","Pruning, Quantization, Knowledge Distillation"],
         ["IV. Efficient LLMs","FlashAttention, vLLM, Speculative Decoding, LoRA"],
         ["V. NAS & Projects","Neural Architecture Search, Final Presentations"]],
        col_widths=[2.0*inch,5.5*inch])
    story.append(t)
    story.append(sp())

    story.append(h2("Roofline Model"))
    story.append(body("A throughput-based performance model with three key metrics:"))
    story.append(bullet("<b>Peak FLOPS</b> = #cores × frequency × FLOP/cycle"))
    story.append(bullet("<b>Memory Bandwidth</b> = data/time [GB/s]"))
    story.append(bullet("<b>Arithmetic Intensity (AI)</b> = #arithmetic_ops / DRAM_bytes [FLOP/byte]"))
    story.append(sp(0.5))
    story.append(formula("Actual FLOPS = min(Peak FLOPS, AI × Memory BW)"))
    story.append(sp(0.5))
    story.append(bullet("Low AI → memory-bound → optimize data reuse (cache blocking)"))
    story.append(bullet("High AI → compute-bound → use SIMD/vectorization"))
    story.append(bullet("<b>DAXPY example:</b> 2 FLOP / 24 bytes = 0.083 FLOP/byte → memory-bound"))
    story.append(bullet("<b>CrossPoint:</b> Peak FLOPS / Memory BW = threshold (e.g., 3.17 FLOP/byte for Xeon E5630)"))
    story.append(sp())

    story.append(h2("Amdahl's Law"))
    story.append(formula("S(p, s) = 1 / [(1 − p) + p/s]"))
    story.append(bullet("S = speedup of entire app; p = fraction of time in improved section; s = speedup of that section"))
    story.append(bullet("Overall speedup is limited by the serial fraction"))
    story.append(exam_tip("Always optimize the section with the highest p (critical path) first"))
    story.append(sp())

    story.append(h2("Performance Optimization Cycle"))
    story.append(body("<b>Step 1 MEASURE</b> → Execute workload, profile, trace, time"))
    story.append(body("<b>Step 2 ANALYZE</b> → Identify critical path + bottleneck via Roofline"))
    story.append(body("<b>Step 3 OPTIMIZE</b> → Apply correct technique based on bottleneck type"))
    story.append(sp(0.5))
    story.append(h3("Key Means for Performance Benchmarking"))
    t = make_table(
        ["Mean Type","Formula","Best For"],
        [["Arithmetic","(1/n)Σxᵢ","Execution time, latency, CPU performance"],
         ["Harmonic","n / Σ(1/xᵢ)","Throughput, rates, network BW, F1 score"],
         ["Geometric","(∏xᵢ)^(1/n)","Speedup ratios, benchmarking comparisons"]],
        col_widths=[1.8*inch,2.2*inch,3.5*inch])
    story.append(t)
    story.append(sp())

    story.append(h2("Scaling Definitions"))
    story.append(bullet("<b>Strong Scaling:</b> Fixed total problem size, add more GPUs → time decreases. Efficiency = t₁/(N×tₙ)"))
    story.append(bullet("<b>Weak Scaling:</b> Problem size grows proportionally with GPUs → runtime stays constant"))
    story.append(exam_tip("DDP with fixed global batch + 1→8 GPUs = STRONG scaling (same total work, more GPUs)"))
    story.append(sp())

    story.append(h2("Memory Hierarchy Latencies"))
    t = make_table(
        ["Level","Latency","Notes"],
        [["Registers","1 cycle","Fastest; private to thread"],
         ["L1 Cache","~3 cycles","On-chip, per SM"],
         ["L2 Cache","~10 cycles","On-chip, shared"],
         ["L3 Cache","~75 cycles","Last level CPU cache"],
         ["DRAM","~100 cycles","Main memory"],
         ["InfiniBand","2000+ cycles","Network"],
         ["GPU mem (PCIe)","~5,000 cycles","Host→Device transfer"],
         ["Ethernet","20,000+ cycles","Slow; not for ML clusters"]],
        col_widths=[1.5*inch,1.5*inch,4.5*inch])
    story.append(t)
    story.append(sp())

    story.append(h2("Optimizer Formulas (Cheat Sheet)"))
    story.append(h3("Vanilla SGD"))
    story.append(formula("w_{t+1} = w_t − α (∂L/∂w_t)"))
    story.append(h3("Momentum"))
    story.append(formula("V_t = β·V_{t-1} + (1−β)·(∂L/∂w_t)   ;   w_{t+1} = w_t − α·V_t"))
    story.append(h3("Adam (most important)"))
    story.append(formula("V_t = β₁·V_{t-1} + (1−β₁)·g_t   [1st moment / momentum]"))
    story.append(formula("S_t = β₂·S_{t-1} + (1−β₂)·g_t²   [2nd moment / RMSProp]"))
    story.append(formula("V̂_t = V_t/(1−β₁ᵗ)  ;  Ŝ_t = S_t/(1−β₂ᵗ)   [bias-corrected]"))
    story.append(formula("w_{t+1} = w_t − α·V̂_t / (√Ŝ_t + ε)"))
    story.append(h3("AdamW (de facto LLM standard)"))
    story.append(body("Same as Adam but weight decay applied directly: <b>w_{t+1} = w_{t+1,Adam} − λ·w_t</b>"))
    story.append(body("Decouples weight decay from adaptive gradient update. Used in LLaMA, GPT-3, etc."))
    story.append(sp(0.5))
    t = make_table(
        ["Optimizer","LR","β₁","β₂","Used By"],
        [["AdamW","3e-4 (cosine)","0.9","0.95","LLaMA (Meta)"],
         ["Adam (custom)","6e-4","0.9","0.98","GPT-3 (OpenAI)"],
         ["Lion","2e-3","0.95","0.98","DeepSeek-67B"],
         ["SGD+momentum","varies","0.9","—","Vision models"]],
        col_widths=[1.5*inch,1.5*inch,0.8*inch,0.8*inch,2.9*inch])
    story.append(t)
    story.append(sp())

    story.append(h2("Training vs Inference"))
    t = make_table(
        ["Aspect","Training","Inference"],
        [["Goal","Maximize throughput","Minimize latency"],
         ["Memory","Weights + Gradients + Optimizer states","Weights + KV Cache"],
         ["Compute","Forward + Backward (backward ≈ 2× forward)","Forward pass only"],
         ["Key metric","Time-to-accuracy","Tokens/sec (TPS)"],
         ["Phases","Full training loop","Prefill (KV cache) + Decode (autoregressive)"],
         ["Bound","Compute-bound","Memory-bandwidth-bound (decode)"]],
        col_widths=[1.5*inch,3.0*inch,3.0*inch])
    story.append(t)
    story.append(exam_tip("Decode phase is memory-bandwidth-bound — reading all weights per token is the bottleneck"))


def sec_pytorch_python(story):
    story += section_break("L3 — Python & PyTorch Performance", C_BLUE)

    story.append(h2("Why Python is Slow"))
    t = make_table(
        ["Source of Overhead","Detail"],
        [["Dynamic typing","Type checked on every operation; cannot optimize at compile time"],
         ["PyObject overhead","Every float = heap-allocated PyObject (28 bytes); boxing/unboxing"],
         ["Interpreter eval loop","Bytecode executed one instruction at a time (ceval.c)"],
         ["Reference counting","Refcount updated on every object operation"],
         ["GIL","Global Interpreter Lock prevents true multi-threading"],
         ["Heap allocations","Most Python objects live on heap; cache-unfriendly"]],
        col_widths=[2.0*inch,5.5*inch])
    story.append(t)
    story.append(key_fact("1000×1000 matrix multiply: Pure Python ≈ 45s vs C+BLAS ≈ 0.05s — 900× slower"))
    story.append(sp())

    story.append(h2("Python Execution Pipeline"))
    story.append(body("<b>Source (.py)</b> → CST → AST → CFG → <b>Bytecode (.pyc)</b> → <b>PVM</b>"))
    story.append(sp(0.5))
    story.append(h3("Peephole Optimizations (applied to bytecode)"))
    story.append(bullet("Constant Folding: 2+2 → 4"))
    story.append(bullet("Dead Code Removal: unreachable branches removed"))
    story.append(bullet("Short Sequence Optimization: collapse multiple LOAD_CONST"))
    story.append(bullet("Variable Lookup: replace global lookups with faster local ones"))
    story.append(sp())

    story.append(h2("Stack Machine & Bytecode"))
    story.append(body("Python VM = stack-based machine. Key bytecodes:"))
    story.append(bullet("LOAD_CONST, LOAD_FAST, LOAD_GLOBAL, LOAD_ATTR — push to stack"))
    story.append(bullet("BINARY_ADD, BINARY_MULTIPLY — pop 2, push result"))
    story.append(bullet("CALL_FUNCTION(a) — call with a args from stack"))
    story.append(bullet("STORE_FAST, RETURN_VALUE — write/return from stack"))
    story.append(code("d = a + b*c  →  LOAD_FAST a, LOAD_FAST b, LOAD_FAST c, BINARY_MULTIPLY, BINARY_ADD, STORE_FAST d"))
    story.append(sp())

    story.append(h2("How PyTorch Bridges Python & C++"))
    t = make_table(
        ["Mechanism","How It Works","Used By"],
        [["CPython C API","C code links to libpython, manipulates PyObject* directly","NumPy, SciPy"],
         ["pybind11","Header-only C++ auto-generates Python bindings","PyTorch ATen (~2000 ops)"],
         ["Graph Serialization","Python builds graph, hands to C++ runtime","TF tf.function, ONNX"],
         ["Bytecode Capture","Intercepts Python bytecode at runtime, extracts compute graph","TorchDynamo (torch.compile)"],
         ["Dynamic Codegen","Generates C++/CUDA/Triton kernels at runtime from captured graph","TorchInductor, XLA, TVM"]],
        col_widths=[1.8*inch,3.2*inch,2.5*inch])
    story.append(t)
    story.append(sp())

    story.append(h2("torch.compile Stack (PyTorch 2.0)"))
    story.append(body("The flagship feature of PyTorch 2.0. Zero code changes required: <b>model = torch.compile(model)</b>"))
    story.append(sp(0.5))
    t = make_table(
        ["Component","Role","Key Detail"],
        [["TorchDynamo","Graph Acquisition","Intercepts CPython bytecode frame evaluation API; captures FX graph; handles dynamic control flow via GUARDS"],
         ["AOTAutograd","Graph Lowering","Captures forward AND backward graphs ahead-of-time; enables joint optimization"],
         ["TorchInductor","Graph Compilation","Default backend; generates Triton (GPU) or C++/OpenMP (CPU) kernels at runtime"],
         ["Guards","Cache Validity","Shape/dtype guards ensure compiled graph matches current input; recompiles on miss"]],
        col_widths=[1.5*inch,1.8*inch,4.2*inch])
    story.append(t)
    story.append(key_fact("torch.compile speedups on A100: +38% TIMM, +76% TorchBench, +52% HuggingFace"))
    story.append(exam_tip("TorchDynamo hooks into CPython's frame evaluation API — this is what enables zero-code-change compilation"))
    story.append(sp())

    story.append(h2("Eager vs. Graph Execution"))
    t = make_table(
        ["Aspect","Eager (Imperative)","Graph (Declarative)"],
        [["Execution","Immediate per op","Define → Compile → Run"],
         ["Debugging","Easy (intermediate values visible)","Hard (graph must execute fully)"],
         ["Optimization","No global view","Operator fusion, memory reuse"],
         ["Examples","PyTorch default","TF 1.x, @tf.function, JAX"],
         ["Dynamic control flow","Native Python support","Limited / requires annotation"]],
        col_widths=[1.8*inch,2.7*inch,3.0*inch])
    story.append(t)
    story.append(sp())

    story.append(h2("PyTorch Autograd & Computation Graphs"))
    story.append(bullet("Set <b>requires_grad=True</b> to track operations on a tensor"))
    story.append(bullet("Calling <b>.backward()</b> computes gradients via chain rule through the dynamic graph"))
    story.append(bullet("Gradients accumulate in <b>.grad</b> attribute"))
    story.append(bullet("<b>torch.no_grad()</b> context manager: no graph built — use for weight updates & inference"))
    story.append(warning("MUST call zero_grad() before each backward pass — gradients accumulate by default!"))
    story.append(formula("z = 2x² + 5x  →  dz/dx = 4x + 5  →  at x=2: grad = 13"))
    story.append(sp())

    story.append(h2("DataLoader Key Facts"))
    story.append(bullet("Abstracts batching, shuffling, and parallel loading of data"))
    story.append(bullet("<b>num_workers > 0</b>: spawns worker processes (bypasses GIL) for parallel data loading"))
    story.append(bullet("<b>pin_memory=True</b>: uses pinned (page-locked) host memory → faster H2D transfers"))
    story.append(bullet("<b>prefetch_factor</b>: preload batches ahead of training loop"))
    story.append(bullet("DataLoader does NOT update model weights"))
    story.append(exam_tip("Profiler shows DataLoader=420ms, GPU idle=410ms → fix the input pipeline (num_workers, pin_memory)"))
    story.append(sp())

    story.append(h2("Python Multiprocessing & GIL"))
    story.append(bullet("<b>GIL (Global Interpreter Lock)</b>: prevents multiple threads from executing Python bytecode in parallel"))
    story.append(bullet("Python threads: <b>concurrency but NOT parallelism</b> (good for I/O-bound, bad for CPU-bound)"))
    story.append(bullet("<b>torch.multiprocessing</b>: extends Python multiprocessing with GPU-aware tensor sharing"))
    story.append(bullet("<b>Default start method: 'spawn'</b> (creates fresh process) — CUDA-safe"))
    story.append(warning("Never use 'fork' with CUDA — inherits corrupted GPU state (CUDA context is not fork-safe)"))
    story.append(sp(0.5))
    t = make_table(
        ["Method","CUDA Safe?","Speed","Notes"],
        [["fork","UNSAFE","Fast","Inherits parent memory incl. GPU state → corruption"],
         ["spawn (default)","SAFE","Slower","Creates clean new process; default for PyTorch+CUDA"],
         ["forkserver","Unsafe","Moderate","Still unsafe for CUDA"]],
        col_widths=[1.2*inch,1.2*inch,1.2*inch,4.0*inch])
    story.append(t)


def sec_cuda_basics(story):
    story += section_break("L4 — CUDA Basics", C_BLUE)

    story.append(h2("CPU vs. GPU Design Philosophy"))
    t = make_table(
        ["CPU (Latency Optimized)","GPU (Throughput Optimized)"],
        [["Few, powerful cores (up to 60+)","Thousands of small cores (CUDA Cores)"],
         ["Sophisticated branch prediction & OOO exec","Simple control — no branch prediction"],
         ["Large L1/L2/L3 cache hierarchy","Small caches; large shared memory"],
         ["Low latency per instruction","High throughput via massive parallelism"],
         ["Best for sequential code","Best for parallel data-parallel code"],
         ["Sequential ≈ 10× faster","Parallel ≈ 10–100× faster"]],
        col_widths=[3.75*inch,3.75*inch],header_color=C_BLUE)
    story.append(t)
    story.append(key_fact("GPU kernel launch overhead ≈ 10s of microseconds. Use GPU only when parallelism > launch cost."))
    story.append(sp())

    story.append(h2("CUDA Thread Hierarchy"))
    story.append(body("Fundamental structure: <b>Thread ⊂ Block ⊂ Grid</b>"))
    story.append(sp(0.5))
    t = make_table(
        ["Concept","Maps To","Key Properties"],
        [["Thread","SP (Streaming Processor / CUDA Core)","Smallest unit; has own registers and local memory"],
         ["Block (Thread Block)","SM (Streaming Multiprocessor)","Max 1024 threads; threads can share memory and sync"],
         ["Grid","GPU (Device)","One grid per kernel launch; blocks execute in any order"],
         ["Warp","32 consecutive threads in a block","Scheduling unit; all execute same instruction (SIMT)"]],
        col_widths=[1.3*inch,2.0*inch,4.2*inch])
    story.append(t)
    story.append(sp(0.5))
    story.append(exam_tip("Warp = 32 threads. Max threads/block = 1024. Never forget these two numbers."))
    story.append(sp())

    story.append(h2("Global Thread Index Formula"))
    story.append(formula("i = blockIdx.x * blockDim.x + threadIdx.x  (1D)"))
    story.append(formula("Row = blockIdx.y*blockDim.y + threadIdx.y  ;  Col = blockIdx.x*blockDim.x + threadIdx.x  (2D)"))
    story.append(body("Always check: <b>if (i < N)</b> to avoid out-of-bounds access when N is not a multiple of blockDim."))
    story.append(exam_tip("COMMON BUG: using only threadIdx.x without blockIdx.x*blockDim.x → only block 0 works correctly!"))
    story.append(sp())

    story.append(h2("CUDA Function Qualifiers"))
    t = make_table(
        ["Qualifier","Executes On","Called From","Notes"],
        [["__global__","GPU (device)","CPU (host)","Returns void; kernel launch entry point"],
         ["__device__","GPU (device)","GPU only","Helper functions on GPU"],
         ["__host__","CPU (host)","CPU only","Default; can combine with __device__"]],
        col_widths=[1.3*inch,1.5*inch,1.5*inch,3.2*inch])
    story.append(t)
    story.append(sp())

    story.append(h2("Kernel Launch Syntax & Memory APIs"))
    story.append(code("kernel<<<gridDim, blockDim, sharedMemBytes, stream>>>(args);"))
    story.append(code("cudaMalloc(&d_ptr, N*sizeof(float));   // Allocate on device"))
    story.append(code("cudaMemcpy(d_ptr, h_ptr, N*sizeof(float), cudaMemcpyHostToDevice);"))
    story.append(code("cudaMemcpy(h_ptr, d_ptr, N*sizeof(float), cudaMemcpyDeviceToHost);"))
    story.append(code("cudaFree(d_ptr);                        // Free device memory"))
    story.append(code("cudaDeviceSynchronize();                // Force host to wait for GPU"))
    story.append(body("<b>Convention:</b> h_ prefix = host memory; d_ prefix = device memory"))
    story.append(sp())

    story.append(h2("CUDA Memory Model — Complete Hierarchy"))
    t = make_table(
        ["Level","Location","Latency","BW","Scope","Managed By","Size"],
        [["Registers","On-chip (SM)","~1 cycle","~8 TB/s","Per thread","Compiler","Thousands/SM"],
         ["Shared Memory / L1","On-chip (SM)","~20 cycles","~1.5–19 TB/s","Per block","Programmer","192KB (A100), 256KB (H100)"],
         ["L2 Cache","On-chip (die)","~200 cycles","—","All SMs","Hardware","40MB (A100), 50MB (H100)"],
         ["Global (HBM)","Off-chip","~400–600 cycles","2 TB/s (A100), 3.35 TB/s (H100)","All SMs","Programmer","40–80 GB"],
         ["Constant","Off-chip (cached)","~1 (cached)","—","All threads (read-only)","Programmer","64 KB"]],
        col_widths=[1.0*inch,1.1*inch,0.8*inch,1.2*inch,0.9*inch,1.1*inch,1.4*inch])
    story.append(t)
    story.append(sp(0.5))
    story.append(key_fact("Shared memory is programmer-managed L1 cache. Declare with __shared__. Must sync with __syncthreads()."))
    story.append(key_fact("Register spilling = spilling to local memory = actually global memory = slow!"))
    story.append(sp())

    story.append(h2("Unified Virtual Memory (UVM)"))
    story.append(bullet("cudaMallocManaged(&ptr, size) — single pointer accessible from both CPU and GPU"))
    story.append(bullet("CUDA 8+ / Pascal+: demand paging — only touched pages migrate"))
    story.append(bullet("cudaMemPrefetchAsync() — restores performance by pre-migrating data"))
    story.append(warning("UVM is for programming convenience, NOT performance. Expert manual cudaMemcpy is always faster."))
    story.append(sp())

    story.append(h2("Warp Divergence"))
    story.append(body("All threads in a warp execute the SAME instruction (SIMT). If threads take different branches:"))
    story.append(bullet("Both paths are executed <b>serially</b> — one path masked/inactive per pass"))
    story.append(bullet("Reduces effective parallelism by the divergence factor"))
    story.append(bullet("<b>Solution:</b> Organize data so threads in same warp follow the same branch"))
    story.append(exam_tip("Warp divergence = performance killer. If/else inside a warp = serialized execution."))
    story.append(sp())

    story.append(h2("__syncthreads() and Synchronization"))
    story.append(bullet("<b>__syncthreads()</b>: barrier for threads within same block. All must reach before any proceed."))
    story.append(bullet("Use AFTER writing shared memory, BEFORE reading it (to prevent race conditions)"))
    story.append(bullet("<b>cudaDeviceSynchronize()</b>: host waits for ALL GPU work to complete"))
    story.append(bullet("Kernel launches are ASYNCHRONOUS — host continues immediately after launch"))
    story.append(bullet("cudaMemcpy() is SYNCHRONOUS by default — host waits"))
    story.append(sp())

    story.append(h2("SM Occupancy"))
    story.append(body("Occupancy = Active Warps per SM / Max Warps per SM. Goal: maximize occupancy to hide latency."))
    story.append(formula("Warps per SM = (threads_per_block / 32) × blocks_assigned_to_SM"))
    story.append(body("Example (Fermi, max 1536 threads/SM, max 8 blocks):"))
    t = make_table(
        ["Block Config","Threads/Block","Warps/Block","Max Blocks","Total Warps","Occupancy"],
        [["8×8","64","2","8","16","33% (LOW)"],
         ["16×16","256","8","6","48","100% (OPTIMAL)"],
         ["32×32","1024","32","1","32","67%"]],
        col_widths=[1.2*inch,1.2*inch,1.2*inch,1.2*inch,1.2*inch,1.5*inch])
    story.append(t)
    story.append(exam_tip("16×16 block (256 threads) = optimal occupancy for Fermi. Always think in multiples of 32."))


def sec_cuda_advanced(story):
    story += section_break("L5 — Advanced CUDA", C_BLUE)

    story.append(h2("GPU Architecture Comparison"))
    t = make_table(
        ["Spec","Tesla V100 (Volta)","A100 (Ampere)","H100 (Hopper)"],
        [["CUDA Cores","5,120","6,912","18,432"],
         ["Tensor Cores","640 (1st/2nd gen)","432 (3rd gen)","640 (4th gen + FP8)"],
         ["SMs","80","108","132"],
         ["Peak FP32","15.7 TFLOPS","19.5 TFLOPS","60 TFLOPS"],
         ["Peak TF16 (Tensor)","125 TFLOPS","312 TFLOPS","1,000 TFLOPS"],
         ["HBM BW","~900 GB/s","~2 TB/s (HBM2e)","~3.35 TB/s (HBM3)"],
         ["L2 Cache","6 MB","40 MB","50 MB"],
         ["Shared Mem/SM","96 KB","192 KB","256 KB"],
         ["NVLink BW","300 GB/s","600 GB/s","900 GB/s"]],
        col_widths=[2.0*inch,1.8*inch,1.8*inch,1.9*inch])
    story.append(t)
    story.append(sp())

    story.append(h2("Tensor Cores"))
    story.append(body("Specialized hardware for mixed-precision matrix multiply-accumulate: <b>D = A×B + C</b>"))
    story.append(bullet("A, B in FP16/BF16/INT8/FP8 → C, D in FP16 or FP32 (accumulate in higher precision)"))
    story.append(bullet("Perform 16×16 matrix operations per cycle — 4–16× faster than CUDA cores for GEMM"))
    story.append(bullet("<b>3rd Gen (A100):</b> TF32, BF16, FP64, 2:4 structured sparsity (2× throughput)"))
    story.append(bullet("<b>4th Gen (H100):</b> Adds FP8 (E4M3 forward, E5M2 backward); Transformer Memory Accelerator (TMA)"))
    story.append(key_fact("Tensor Cores power ALL modern LLM training. FP8 on H100 = 2× more FLOPS than BF16."))
    story.append(sp())

    story.append(h2("CUDA Streams"))
    story.append(body("A stream = an ordered queue of GPU work. Host places work, device executes when resources free."))
    story.append(bullet("Operations in the <b>same stream</b>: ORDERED (FIFO), cannot overlap"))
    story.append(bullet("Operations in <b>different streams</b>: UNORDERED, CAN overlap"))
    story.append(bullet("Default stream: all operations go here if no stream specified"))
    story.append(code("cudaStreamCreate(&stream1);"))
    story.append(code("cudaMemcpyAsync(d_a, h_a, size, cudaMemcpyHostToDevice, stream1);"))
    story.append(code("my_kernel<<<grid, block, 0, stream1>>>(d_a);"))
    story.append(code("cudaStreamSynchronize(stream1);"))
    story.append(key_fact("Streams enable overlapping H2D transfers with kernel execution → hides memory latency"))
    story.append(sp())

    story.append(h2("Pinned (Page-Locked) Memory"))
    story.append(bullet("Allocated with cudaMallocHost() — memory cannot be paged out by OS"))
    story.append(bullet("Required for cudaMemcpyAsync() to work correctly"))
    story.append(bullet("cudaMemcpy() is faster with pinned source/dest (DMA can transfer directly)"))
    story.append(bullet("Downside: limits virtual memory available to OS; use judiciously"))
    story.append(sp())

    story.append(h2("Memory Coalescing (Critical for Global Memory Performance)"))
    story.append(body("When threads in a warp access consecutive memory addresses, HW coalesces into one transaction."))
    t = make_table(
        ["Access Pattern","Transactions","Performance"],
        [["Stride-1 (consecutive)","1 transaction","Optimal — all threads in one 128-byte line"],
         ["Stride-2","2 transactions","Half efficiency"],
         ["Stride-32 (worst)","32 transactions","Terrible — one transaction per thread"]],
        col_widths=[2.5*inch,1.8*inch,3.2*inch])
    story.append(t)
    story.append(exam_tip("Pattern A (stride-1) coalesces to 1 transaction; Pattern B (stride-32) = 32 transactions = 32× slower"))
    story.append(sp())

    story.append(h2("Shared Memory Bank Conflicts"))
    story.append(bullet("Shared memory divided into 32 banks (4-byte granularity)"))
    story.append(bullet("Consecutive 4-byte words → consecutive banks"))
    story.append(bullet("<b>Bank conflict:</b> 2+ threads in warp access different addresses in same bank → serialized"))
    story.append(bullet("<b>No conflict:</b> All threads access different banks (stride-1 for 32 threads = no conflict)"))
    story.append(bullet("<b>Broadcast:</b> All threads access same address in same bank → no conflict (1 read, broadcast)"))
    story.append(bullet("Fix: use padding (e.g., __shared__ float tile[32][33]) to shift columns across banks"))
    story.append(sp())

    story.append(h2("Tiled Matrix Multiplication"))
    story.append(body("Key optimization: load tiles of A and B into shared memory, reuse for all C elements in tile."))
    story.append(bullet("Each element of global memory loaded only once per tile instead of once per multiply"))
    story.append(bullet("Tile size TILE×TILE; larger tile = more reuse = higher arithmetic intensity"))
    story.append(bullet("Requires __syncthreads() after loading tile (before compute) and after compute (before next tile)"))
    story.append(key_fact("Tiled matmul: AI = TILE/2 FLOP/byte vs. naive AI ≈ 1/4 FLOP/byte → moves toward compute-bound"))
    story.append(sp())

    story.append(h2("Compute Capability Reference"))
    t = make_table(
        ["Compute Capability","Architecture","Key Feature"],
        [["7.0","Volta","Tensor Cores introduced; V100"],
         ["7.5","Turing","INT8 Tensor Cores; RTX 20xx"],
         ["8.0","Ampere","TF32, BF16, FP64 Tensor Cores; 2:4 sparsity; A100"],
         ["9.0","Hopper","FP8 Tensor Cores; TMA; Transformer Engine; H100"]],
        col_widths=[1.5*inch,1.5*inch,4.5*inch])
    story.append(t)
    story.append(sp())

    story.append(h2("CUDA Event API for Timing"))
    story.append(code("cudaEvent_t start, stop; float ms;"))
    story.append(code("cudaEventCreate(&start); cudaEventCreate(&stop);"))
    story.append(code("cudaEventRecord(start, 0);"))
    story.append(code("my_kernel<<<grid,block>>>(args);"))
    story.append(code("cudaEventRecord(stop, 0);"))
    story.append(code("cudaEventSynchronize(stop);   // Wait for kernel"))
    story.append(code("cudaEventElapsedTime(&ms, start, stop);  // ms in milliseconds"))
    story.append(warning("Standard CPU timers (clock(), gettimeofday()) CANNOT measure GPU timing correctly — always use CUDA Events"))


def sec_ddl(story):
    story += section_break("L6 — Distributed Deep Learning (DDL)", C_GREEN)

    story.append(h2("Why Distributed Training"))
    story.append(bullet("LLaMA 3 405B at fp16 = <b>810 GB</b> → single H100 (80 GB) cannot hold even the weights"))
    story.append(bullet("GPT-3 175B with Adam optimizer ≈ <b>2.8 TB</b> (16 bytes/param)"))
    story.append(bullet("Modern LLM training: 1,000 – 100,000+ GPUs running for weeks"))
    story.append(key_fact("TSM video model: 50 hours → 14 minutes with 1,536 GPUs (211× speedup)"))
    story.append(sp())

    story.append(h2("Parallelism Strategies"))
    t = make_table(
        ["Strategy","How It Works","Best For","Key Challenge"],
        [["Data Parallelism","Same model on each GPU; different data batches; AllReduce gradients","Most cases; batch size can grow","Communication overhead of AllReduce"],
         ["Model / Tensor Parallelism","Split weight matrices within a layer across GPUs","Huge models; single layer too large","High BW needed (NVLink); all-to-all per layer"],
         ["Pipeline Parallelism","Different layers on different GPUs; microbatches fill pipeline","Extremely deep models","Pipeline bubbles at start/end"],
         ["ZeRO / FSDP","Shard optimizer states, gradients, AND parameters across GPUs","Memory-constrained training","AllGather + Reduce-Scatter overhead"],
         ["4D / Hybrid","DP + TP + PP + SP combined","LLM pre-training at 1000s of GPUs","Complex configuration"]],
        col_widths=[1.4*inch,1.8*inch,1.6*inch,2.7*inch])
    story.append(t)
    story.append(sp())

    story.append(h2("Communication Collective Primitives"))
    story.append(exam_tip("Know what each collective does — exam question on which to use for gradient averaging"))
    t = make_table(
        ["Primitive","What It Does","DL Use Case"],
        [["Broadcast","Root → all: same data to everyone","Initialize model parameters on all workers"],
         ["Reduce","All → root: combine (sum/max/min) from all","Gradient accumulation at parameter server"],
         ["AllReduce","All → all: combine AND distribute to everyone","Gradient averaging in data parallelism (THE key collective)"],
         ["Reduce-Scatter","Each rank gets partial result (its shard)","ZeRO gradient sharding"],
         ["AllGather","Each has partial; all get full data","ZeRO parameter reconstruction before forward pass"],
         ["AllToAll","Each rank sends DIFFERENT data to each other","Expert routing in Mixture-of-Experts"]],
        col_widths=[1.3*inch,2.7*inch,3.5*inch])
    story.append(t)
    story.append(exam_tip("AllReduce = most important collective in deep learning. Used for gradient synchronization in data parallelism."))
    story.append(sp())

    story.append(h2("Ring AllReduce Algorithm"))
    story.append(body("Bandwidth-optimal collective used by NCCL. Two phases:"))
    story.append(bullet("<b>Phase 1 — Scatter-Reduce:</b> Each of N-1 iterations: send to right neighbor, receive from left; each node accumulates partial sum"))
    story.append(bullet("<b>Phase 2 — AllGather:</b> Each of N-1 iterations: send completed chunk right; all nodes receive full result"))
    story.append(formula("Communication cost per GPU = 2N(P-1)/P ≈ 2N for large P"))
    story.append(formula("Parameter Server cost = 2N(P-1)  [grows linearly with P — bottleneck!]"))
    story.append(key_fact("Ring AllReduce is nearly bandwidth-independent of #GPUs. PS becomes a bottleneck at scale."))
    story.append(sp())

    story.append(h2("Synchronous vs. Asynchronous SGD"))
    t = make_table(
        ["Aspect","Synchronous (Sync-SGD)","Asynchronous (ASGD)"],
        [["Update timing","All workers must finish before update","Workers update independently"],
         ["Gradient staleness","None — all fresh","Yes — stale gradients possible"],
         ["Determinism","Fully deterministic","Non-deterministic"],
         ["Straggler sensitivity","High — slow worker stalls all","Low — slow workers don't block others"],
         ["Convergence","Mathematically equivalent to SGD","May diverge with high staleness"],
         ["Used in","PyTorch DDP/FSDP (default)","Hogwild, some RL scenarios"]],
        col_widths=[1.8*inch,2.7*inch,3.0*inch])
    story.append(t)
    story.append(exam_tip("Slow straggler stalls synchronous data-parallel SGD = TRUE exam answer"))
    story.append(sp())

    story.append(h2("ZeRO (Zero Redundancy Optimizer)"))
    story.append(body("Shards model state across GPUs while preserving data-parallel semantics. Three stages:"))
    t = make_table(
        ["Stage","What Is Sharded","Memory/GPU (N=64)","Max Model Size"],
        [["Baseline (DP)","Nothing — full replica","~120 GB (all)","5B params"],
         ["ZeRO-1","Optimizer states only","~31 GB (~4× reduction)","~19B params"],
         ["ZeRO-2","Optimizer states + Gradients","~17 GB (~8× reduction)","~36B params"],
         ["ZeRO-3 / FSDP","Optimizer + Gradients + Parameters","~2 GB (~64× reduction)","~320B params"]],
        col_widths=[1.3*inch,2.0*inch,1.7*inch,2.5*inch])
    story.append(t)
    story.append(sp(0.5))
    story.append(key_fact("ZeRO-3 is implemented in PyTorch as FullyShardedDataParallel (FSDP). Use FSDP2 (PyTorch 2.4+) for best performance."))
    story.append(sp(0.5))
    story.append(h3("FSDP Forward/Backward Flow"))
    story.append(bullet("AllGather parameters → Forward pass → Free full weights → AllGather parameters → Backward pass → Reduce-Scatter gradients → Free full weights → Update local shard"))
    story.append(sp())

    story.append(h2("Pipeline Parallelism"))
    story.append(body("Split model layers across GPUs. Microbatches pipeline through stages."))
    t = make_table(
        ["Variant","Key Feature","Memory","Bubble"],
        [["Naive","One microbatch at a time","Low","High (~1 - 1/m)"],
         ["GPipe","M microbatches; all fwd then all bwd","O(M) activations","Moderate"],
         ["1F1B (Interleaved)","Prioritize backward; release activations ASAP","Lower","~Same as GPipe"],
         ["Zero-Bubble (ZB-H1)","Defer W-grad to fill bubbles","Same as 1F1B","~50% less"],
         ["Zero-Bubble (ZB-H2)","Near-zero bubbles","Slightly higher","~Near zero"]],
        col_widths=[1.5*inch,2.5*inch,1.3*inch,2.2*inch])
    story.append(t)
    story.append(sp())

    story.append(h2("Communication Libraries"))
    t = make_table(
        ["Library","Type","Used By","Notes"],
        [["NCCL","GPU-GPU collectives","PyTorch DDP/FSDP2, DeepSpeed, Megatron","Topology-aware; NVLink/IB; MOST POPULAR"],
         ["Gloo","CPU-based (Meta)","PyTorch CPU fallback","No NCCL available? Gloo is default"],
         ["MPI","General HPC","Horovod, academic clusters","CUDA-aware MPI for direct GPU transfers"],
         ["UCX","Low-level transport","HPC, RDMA, DeepSpeed","Ultra-low latency GPU-to-GPU"],
         ["SHARP","In-network reduction","InfiniBand Quantum-2","Hardware-accelerated AllReduce in switch"]],
        col_widths=[1.0*inch,1.3*inch,2.2*inch,3.0*inch])
    story.append(t)
    story.append(sp())

    story.append(h2("Scale-Up vs. Scale-Out"))
    story.append(bullet("<b>Scale-up (intra-node):</b> NVLink/NVSwitch → 900 GB/s (DGX H100) — fastest path"))
    story.append(bullet("<b>Scale-out (inter-node):</b> InfiniBand NDR → up to 400 Gb/s — slower but needed for large clusters"))
    story.append(bullet("<b>DDP (DistributedDataParallel):</b> Full model on each GPU; AllReduce gradients"))
    story.append(bullet("<b>FSDP (FullyShardedDataParallel):</b> ZeRO-3; shards everything; handles 100B+ models"))


def sec_quantization(story):
    story += section_break("L7 — Neural Network Quantization", C_PURPLE)

    story.append(h2("Why Quantize"))
    story.append(bullet("Reduce model memory: FP32→INT8 = 4× smaller; FP32→FP16 = 2× smaller"))
    story.append(bullet("Faster inference: INT8 matmul ~4× faster than FP32 on specialized hardware"))
    story.append(bullet("Lower energy: DRAM access = 640 pJ; INT8 mult = 0.2 pJ vs FP32 mult = 3.7 pJ (18.5× less)"))
    story.append(bullet("Enable edge deployment: smaller models fit on mobile/embedded devices"))
    story.append(key_fact("DeepSeek-R1 671B: 720 GB original → 131 GB quantized = 5.5× reduction"))
    story.append(sp())

    story.append(h2("Floating Point Formats"))
    t = make_table(
        ["Format","Sign","Exponent","Mantissa","Total","Key Property"],
        [["FP32 (IEEE)","1","8","23","32","Full precision; baseline"],
         ["FP16 (IEEE)","1","5","10","16","2× smaller; narrower range → overflow risk"],
         ["BF16 (Google)","1","8","7","16","Same range as FP32; drop-in for training"],
         ["TF32 (NVIDIA)","1","8","10","19","FP32 range + FP16 precision; A100 default"],
         ["FP8 E4M3","1","4","3","8","H100 forward pass (weights/activations)"],
         ["FP8 E5M2","1","5","2","8","H100 backward pass (gradients — needs more range)"],
         ["INT8","—","—","—","8","Fixed-point; symmetric or asymmetric"],
         ["INT4","—","—","—","4","Aggressive; needs careful calibration"]],
        col_widths=[1.0*inch,0.5*inch,0.8*inch,0.9*inch,0.5*inch,3.8*inch])
    story.append(t)
    story.append(exam_tip("BF16 has same exponent bits as FP32 → same range → safe drop-in for training. FP16 can overflow."))
    story.append(sp())

    story.append(h2("Linear (Uniform) Quantization — Core Math"))
    story.append(formula("r = S(q − Z)   [dequantization: int → float]"))
    story.append(formula("q = clamp(round(r/S) + Z, q_min, q_max)   [quantization: float → int]"))
    story.append(body("Where: <b>r</b> = real float value, <b>q</b> = quantized integer, <b>S</b> = scale (step size), <b>Z</b> = zero-point"))
    story.append(sp(0.5))
    story.append(h3("Scale and Zero-Point Formulas"))
    story.append(formula("S = (r_max − r_min) / (q_max − q_min)"))
    story.append(formula("Z = round(q_min − r_min/S)"))
    story.append(body("Example (INT8, r_min=−1.8, r_max=2.5): S = (2.5+1.8)/255 ≈ 0.01686; Z = round(−128+1.8/0.01686) = −21"))
    story.append(sp())

    story.append(h2("Symmetric vs. Asymmetric Quantization"))
    t = make_table(
        ["Mode","Zero-Point","Range","Best For"],
        [["Symmetric","Z = 0","[−|r_max|, |r_max|] — wasted if skewed","Weights (usually symmetric about 0)"],
         ["Asymmetric","Z ≠ 0","[r_min, r_max] — full utilization","Activations after ReLU (all positive)"]],
        col_widths=[1.3*inch,1.3*inch,2.3*inch,2.6*inch])
    story.append(t)
    story.append(key_fact("Best practice: Symmetric weights + Asymmetric activations = hardware-efficient combination"))
    story.append(sp())

    story.append(h2("Per-Tensor vs. Per-Channel Quantization"))
    story.append(bullet("<b>Per-Tensor:</b> Single S, Z for entire tensor. Fastest, most hardware compatible."))
    story.append(bullet("<b>Per-Channel:</b> Separate S, Z per output channel. Better accuracy. No special HW needed for weights."))
    story.append(sp())

    story.append(h2("PTQ vs. QAT"))
    t = make_table(
        ["Aspect","PTQ (Post-Training Quantization)","QAT (Quantization-Aware Training)"],
        [["Training needed","No (or calibration data only)","Yes — simulate quantization during training"],
         ["Accuracy","Good for 8-bit; degrades at 4-bit","Better, especially at low bit-widths"],
         ["Speed of setup","Fast","Slow (requires full training run)"],
         ["How it works","Quantize trained FP32 model directly","Insert fake-quantization nodes; network adapts"],
         ["Use case","Production inference; edge deployment","When PTQ accuracy is insufficient"]],
        col_widths=[1.5*inch,2.7*inch,3.3*inch])
    story.append(t)
    story.append(sp())

    story.append(h2("Static vs. Dynamic Quantization"))
    story.append(bullet("<b>Static (Offline PTQ):</b> Calibration pass determines fixed scale/zero-point for activations. Fast at inference."))
    story.append(bullet("<b>Dynamic (Runtime):</b> Quantize activations per-inference based on observed min/max. Slower but more accurate."))
    story.append(sp())

    story.append(h2("FP8 Training Details (H100)"))
    story.append(body("Two FP8 formats optimized for different roles:"))
    t = make_table(
        ["Format","Exponent","Mantissa","Has Inf?","Best For"],
        [["FP8 E4M3","4","3","No (NaN only)","Forward pass: weights & activations (more precision)"],
         ["FP8 E5M2","5","2","Yes","Backward pass: gradients (needs higher dynamic range)"]],
        col_widths=[1.2*inch,1.0*inch,1.0*inch,1.0*inch,4.3*inch])
    story.append(t)
    story.append(key_fact("FP8 on H100: 2× more FLOPS vs BF16. Key for training trillion-parameter models efficiently."))
    story.append(sp())

    story.append(h2("Simulating Quantization"))
    story.append(bullet("Insert quantizer nodes into compute graph to simulate fixed-point on FP32 hardware (CPU/GPU)"))
    story.append(bullet("Forward: quantize → dequantize (FP32 → round to INT8 levels → FP32 approximation)"))
    story.append(bullet("Backward: Straight-Through Estimator (STE) — pass gradients through quantizer as if identity"))
    story.append(bullet("Benefits: GPU acceleration, no dedicated kernel, test various bit-widths quickly"))

def sec_pruning(story):
    story += section_break("L8 — Neural Network Pruning & Sparsity", C_ORANGE)

    story.append(h2("Why Prune"))
    story.append(bullet("Neural networks are massively over-parameterized — >90% of weights can be removed with minimal accuracy loss"))
    story.append(bullet("<b>NVIDIA MLPerf Llama 2 70B:</b> 2.5× speedup, 99% accuracy retained via depth+width pruning"))
    story.append(bullet("Reduces storage, memory bandwidth, and compute requirements"))
    story.append(key_fact("Pruning 90% of AlexNet weights: 9× fewer parameters, 3× fewer MACs, nearly same accuracy"))
    story.append(sp())

    story.append(h2("Pruning Granularity Spectrum"))
    t = make_table(
        ["Granularity","Description","HW Acceleration","Compression"],
        [["Fine-grained (Unstructured)","Any individual weight","Custom HW only (EIE accelerator)","Highest"],
         ["N:M Pattern (2:4)","2 zeros per 4 consecutive weights","NVIDIA Sparse Tensor Cores (A100+): 2×","High"],
         ["Vector-level","Entire rows/columns","Standard GEMM","Moderate"],
         ["Head Pruning","Remove entire attention heads","Standard GEMM (fewer heads)","Moderate"],
         ["Width Pruning","Reduce FFN/hidden dims","Standard GEMM (smaller matrices)","Moderate"],
         ["Depth Pruning","Remove entire transformer layers","Trivial — fewer forward passes","Lower"]],
        col_widths=[1.5*inch,1.8*inch,2.0*inch,2.2*inch])
    story.append(t)
    story.append(exam_tip("Fine-grained = more compression but GPU-unfriendly. Structured (channel/depth) = GPU-friendly, less compression."))
    story.append(sp())

    story.append(h2("2:4 Structured Sparsity (NVIDIA Ampere+)"))
    story.append(body("Exactly 2 zeros in every 4 consecutive weights = 50% sparsity. Natively accelerated by Sparse Tensor Cores."))
    story.append(bullet("Compressed format: non-zero values + 2-bit indices per 4 elements"))
    story.append(bullet("A100: up to <b>2× peak throughput</b> for sparse matrix multiplication"))
    t = make_table(
        ["Model","Dense FP16","Sparse FP16 (2:4)","Accuracy Drop"],
        [["ResNet-50 ImageNet Top-1","76.1%","76.2%","None (slight gain!)"],
         ["BERT-Large SQuAD F1","91.9","91.9","None"],
         ["FairSeq Transformer BLEU","28.2","28.5","None (slight gain!)"]],
        col_widths=[2.5*inch,1.8*inch,2.0*inch,1.2*inch])
    story.append(t)
    story.append(sp())

    story.append(h2("Pruning Criteria"))
    story.append(h3("Magnitude-Based (Most Common)"))
    story.append(formula("Importance = |W|   (element-wise for unstructured)"))
    story.append(formula("Importance = ||W^(S)||_p   (group-wise for structured)  — L1 or L2 norm"))
    story.append(sp(0.5))
    story.append(h3("Wanda (Weights AND Activations)"))
    story.append(formula("Score = |W_{i,j}| × ||X_j||_2   (weight magnitude × input activation norm)"))
    story.append(body("'A small weight times a huge activation is not small.' Beats plain magnitude at all sparsity levels."))
    story.append(sp(0.5))
    story.append(h3("SparseGPT (Hessian-Based)"))
    story.append(formula("Saliency = W_{t,j}² / [H⁻¹]_{jj}   where H = 2XX^T + λI"))
    story.append(formula("Weight update = ΔW_{t,k} = −(W_{t,j}/[H⁻¹]_{jj}) × [H⁻¹]_{jk}"))
    story.append(body("One-shot unstructured pruning using approximate Hessian inverse. No retraining needed. Prunes OPT-175B in ~4 hours on 1 GPU."))
    story.append(sp())

    story.append(h2("Iterative Pruning Pipeline"))
    t = make_table(
        ["Step","Action","Notes"],
        [["1","Train Dense Model","Full model to convergence"],
         ["2","Compute Importance","Magnitude, Wanda, or SparseGPT score"],
         ["3","Prune Weights","Remove lowest-importance at target granularity"],
         ["4","Fine-Tune (Recover)","1/10–1/100 of original LR; 1–3 epochs"],
         ["5","Iterate or Deploy","Repeat steps 2–4 for higher sparsity"]],
        col_widths=[0.5*inch,1.8*inch,5.2*inch])
    story.append(t)
    story.append(key_fact("Pruning only → accuracy drops. Pruning + fine-tuning → maintains accuracy to ~90% sparsity. Iterative = best."))
    story.append(sp())

    story.append(h2("Transformer Pruning Summary"))
    t = make_table(
        ["Component","Pruning Method","Effect"],
        [["Individual weights","Unstructured / 2:4 sparsity","Max compression; needs sparse HW or Sparse Tensor Cores"],
         ["Attention heads","Head pruning","Remove d_head columns from W_Q/K/V/O; reduces attention cost"],
         ["FFN dims","Width pruning","Reduce 4d→2d intermediate; 50% FFN cost reduction"],
         ["Full layers","Depth pruning","Remove entire transformer blocks; NVIDIA Llama 2: 80→32 layers"]],
        col_widths=[1.5*inch,2.0*inch,4.0*inch])
    story.append(t)
    story.append(key_fact("NVIDIA Llama 2 70B: depth pruning (80→32 layers) + width pruning → 2.5× speedup, 99% accuracy retained"))
    story.append(sp())

    story.append(h2("Modern LLM Pruning Methods"))
    t = make_table(
        ["Method","Type","Key Idea","No Retrain?"],
        [["SparseGPT","Unstructured","Hessian-based weight updates after pruning; compensates errors","Yes"],
         ["Wanda","Unstructured","Score = |w| × ||x||; activations matter as much as weights","Yes"],
         ["Minitron (NVIDIA)","Structured depth+width","Prune + KD; 40× fewer tokens than training from scratch","No (KD)"],
         ["Sheared-LLaMA","Joint depth+width+head","Learned masks; outperforms same-size from-scratch models","No"]],
        col_widths=[1.5*inch,1.5*inch,3.0*inch,1.5*inch])
    story.append(t)


def sec_flash_attention(story):
    story += section_break("L9 — Efficient Transformers & FlashAttention", C_NAVY)

    story.append(h2("Self-Attention Mechanism"))
    story.append(formula("Attention(Q, K, V) = softmax(QK^T / √d_k) · V"))
    story.append(body("<b>Q = XW_Q</b> (what am I looking for?), <b>K = XW_K</b> (what do I have?), <b>V = XW_V</b> (what to extract?)"))
    story.append(bullet("Score_{ij} = Q_i · K_j^T / √d_k  (scaled dot-product)"))
    story.append(bullet("α_{ij} = softmax(Score_{ij})  (attention weights)"))
    story.append(bullet("Output_i = Σ_j α_{ij} · V_j  (weighted sum of values)"))
    story.append(bullet("<b>Causal (GPT):</b> attends only to preceding positions; <b>Non-causal (BERT):</b> bidirectional"))
    story.append(bullet("Standard complexity: <b>O(N²) time and memory</b> — N×N attention matrix is the bottleneck"))
    story.append(sp())

    story.append(h2("Why Standard Attention is Memory-Bound"))
    story.append(body("Standard attention requires materializing the full N×N attention matrix in HBM:"))
    story.append(bullet("Step 1: QK^T → write N×N matrix S to HBM"))
    story.append(bullet("Step 2: read S from HBM → softmax → write P to HBM"))
    story.append(bullet("Step 3: read P, V from HBM → PV → write O to HBM"))
    story.append(body("Each step = expensive HBM read+write. HBM (1.5 TB/s) is 13× slower than SRAM (19 TB/s)."))
    story.append(exam_tip("Standard attention OOMs for long sequences because N×N matrix in HBM scales quadratically"))
    story.append(sp())

    story.append(h2("FlashAttention — Core Insight"))
    story.append(key_fact("IO-aware attention: minimize HBM reads/writes, not FLOPs. Same O(N²) FLOPs, much less HBM traffic."))
    story.append(sp(0.5))
    story.append(h3("Two Key Techniques"))
    story.append(body("<b>1. Tiling:</b> Process attention in small tiles that fit in SRAM. Never materialize full N×N matrix."))
    story.append(body("<b>2. Online Softmax:</b> Compute softmax incrementally block-by-block using running statistics."))
    story.append(sp())

    story.append(h2("FlashAttention Algorithm — Key Steps"))
    story.append(body("Inputs: Q, K, V ∈ R^(N×d) in HBM; SRAM of size M"))
    story.append(formula("Block sizes: Bc = ⌈M/4d⌉,  Br = min(⌈M/4d⌉, d)"))
    story.append(sp(0.5))
    story.append(bullet("<b>Outer loop:</b> iterate over K, V blocks (T_c = ⌈N/Bc⌉ iterations)"))
    story.append(bullet("Load K_j, V_j into SRAM"))
    story.append(bullet("<b>Inner loop:</b> iterate over Q blocks (T_r = ⌈N/Br⌉ iterations)"))
    story.append(bullet("Load Q_i, O_i, ℓ_i, m_i into SRAM"))
    story.append(bullet("Compute S_ij = Q_i · K_j^T on-chip (small tile, not full N×N)"))
    story.append(bullet("Compute local max m̃_ij = rowmax(S_ij) and local sum ℓ̃_ij = rowsum(exp(S_ij − m̃_ij))"))
    story.append(bullet("Update running stats: m_new = max(m_old, m̃); ℓ_new = e^(m_old−m_new)·ℓ_old + e^(m̃−m_new)·ℓ̃"))
    story.append(bullet("Rescale and accumulate O_i with new block's contribution; write back to HBM"))
    story.append(sp(0.5))
    story.append(h3("Online Softmax Statistics"))
    story.append(formula("m(x) = max_i(x_i)  [running maximum]"))
    story.append(formula("ℓ(x) = Σ_i exp(x_i − m(x))  [running sum for denominator]"))
    story.append(formula("softmax(x) = f(x) / ℓ(x)  where f(x) = [exp(x_1−m), ..., exp(x_B−m)]"))
    story.append(sp())

    story.append(h2("FlashAttention Performance"))
    t = make_table(
        ["Metric","Standard Attention","FlashAttention","Improvement"],
        [["HBM R/W (GPT-2 medium)","40.3 GB","4.4 GB","~9× less"],
         ["Runtime (GPT-2 medium)","41.7 ms","7.3 ms","~5.7× faster"],
         ["GFLOPs (GPT-2 medium)","66.6","75.2","13% more (recomputation)"],
         ["Memory complexity","O(N²)","O(N) ← linear!","10–20× less"],
         ["Wall-clock speedup","—","2–4× (up to 6×)","With masking+dropout: 4×"],
         ["BERT training (MLPerf)","20.0 ± 1.5 min","17.4 ± 1.4 min","~15% faster"]],
        col_widths=[2.3*inch,1.7*inch,1.5*inch,2.0*inch])
    story.append(t)
    story.append(sp(0.5))
    story.append(exam_tip("FlashAttention has same O(N²) FLOPs as standard attention — it only reduces HBM I/O. This is a common trick question!"))
    story.append(sp())

    story.append(h2("Backward Pass: Recomputation"))
    story.append(bullet("FlashAttention does NOT store the N×N attention matrix P from the forward pass"))
    story.append(bullet("In backward, recompute attention from Q, K, V using saved softmax statistics (m and ℓ)"))
    story.append(bullet("Tradeoff: ~13% more FLOPs vs. 10–20× less memory"))
    story.append(bullet("This is why FlashAttention backward stores O (output) and (m, ℓ) — but not P"))
    story.append(sp())

    story.append(h2("Kernel Fusion"))
    story.append(body("A <b>kernel</b> = a GPU operation. <b>Fusion</b> = combining multiple operations into one kernel."))
    story.append(bullet("Standard: QK^T (kernel 1) → softmax (kernel 2) → dropout (kernel 3) → V (kernel 4)"))
    story.append(bullet("FlashAttention: single fused kernel — load from HBM once, all ops in SRAM, write once"))
    story.append(bullet("Eliminates repeated HBM round-trips for each intermediate result"))
    story.append(key_fact("Kernel fusion is the primary implementation technique enabling FlashAttention's speedup"))
    story.append(sp())

    story.append(h2("vLLM & PagedAttention (Bonus)"))
    story.append(bullet("Traditional LLM inference: pre-allocates max-length contiguous KV cache → 60–80% GPU memory wasted"))
    story.append(bullet("<b>PagedAttention:</b> Splits KV cache into fixed-size blocks (e.g., 16 tokens), stored non-contiguously"))
    story.append(bullet("Block Table maps logical → physical GPU addresses; allocate on-demand"))
    story.append(bullet("<b>Continuous Batching:</b> finished requests immediately replaced; no GPU idle 'bubbles'"))
    story.append(key_fact("vLLM achieves 2–4× higher throughput vs static batching"))


def sec_kd(story):
    story += section_break("L10 — Knowledge Distillation", C_GREEN)

    story.append(h2("Core Concept"))
    story.append(body("Train a small <b>student model</b> to replicate behavior of a large <b>teacher model</b>."))
    story.append(bullet("<b>Key observation:</b> Large networks are easier to train; small networks are easier to deploy"))
    story.append(bullet("<b>DistilBERT (2019):</b> 40% smaller than BERT, retains 97% of language understanding"))
    story.append(sp())

    story.append(h2("Three Types of Knowledge"))
    t = make_table(
        ["Type","Source","How Transferred"],
        [["Response-Based","Last layer logits (output)","KL divergence between soft teacher/student distributions"],
         ["Feature-Based","Intermediate layer activations","MSE or KL between teacher/student feature maps"],
         ["Relation-Based","Relationships between feature maps","FSP matrix (inner products between features from two layers)"]],
        col_widths=[1.5*inch,2.2*inch,3.8*inch])
    story.append(t)
    story.append(sp())

    story.append(h2("Hard vs. Soft Labels & Temperature"))
    story.append(body("<b>Hard labels:</b> One-hot [0,0,0,1,0,...] — no inter-class similarity information"))
    story.append(body("<b>Soft labels:</b> Teacher's probability distribution — encodes 'dark knowledge' (similarity between classes)"))
    story.append(formula("Softmax with temperature T: q_i = exp(z_i/T) / Σ_j exp(z_j/T)"))
    story.append(bullet("T = 1: standard peaked softmax"))
    story.append(bullet("T > 1: smoother distribution → more dark knowledge exposed → more informative for student"))
    story.append(bullet("T = 5, 7, 10: progressively flatten; expose class similarities more clearly"))
    story.append(sp())

    story.append(h2("Loss Functions"))
    story.append(h3("KL Divergence"))
    story.append(formula("D_KL(P||Q) = Σ_x p(x) ln(p(x)/q(x))"))
    story.append(body("Non-symmetric. Measures relative entropy between distributions. Used for distillation loss."))
    story.append(sp(0.5))
    story.append(h3("Total Distillation Loss"))
    story.append(formula("L_total = α · L_student + (1−α) · L_distill"))
    story.append(bullet("L_student = cross-entropy with hard (one-hot) labels"))
    story.append(bullet("L_distill = KL divergence between teacher and student soft distributions (at temperature T)"))
    story.append(bullet("α ∈ [0,1]: higher α → more hard labels; lower α → more soft knowledge"))
    story.append(sp())

    story.append(h2("FitNets (Feature-Based KD)"))
    story.append(body("Train deeper and thinner students using teacher's intermediate layers as 'hints'."))
    story.append(bullet("<b>Hint layer:</b> teacher intermediate layer whose output student learns to predict"))
    story.append(bullet("<b>Guided layer:</b> student layer that learns from hint"))
    story.append(bullet("Stage 1: Hints training — match intermediate features via MSE"))
    story.append(bullet("Stage 2: Full KD — train entire student with soft labels"))
    story.append(key_fact("FitNet 1: 11 layers, 250K params → 13× speedup, 36× compression vs teacher (5 layers, 9M params)"))
    story.append(sp())

    story.append(h2("Training Modes"))
    t = make_table(
        ["Mode","When","How"],
        [["Offline (most common)","Pre-trained teacher available","Teacher frozen; student trains from scratch with teacher supervision"],
         ["Online","No pre-trained teacher","Teacher and student trained simultaneously in end-to-end fashion"],
         ["Self-Distillation","No teacher needed","Same model — deeper layers supervise shallower layers; or later epoch supervises earlier"]],
        col_widths=[1.5*inch,2.0*inch,4.0*inch])
    story.append(t)
    story.append(sp())

    story.append(h2("NVIDIA Minitron: Production KD"))
    story.append(body("Prune large model then use KD to recover accuracy. 40× cheaper than training from scratch."))
    t = make_table(
        ["Model","Params","Method","MMLU"],
        [["Llama-3.1-Minitron-4B (width)","4B","Width prune + distill","60.5"],
         ["Llama-3.1-Minitron-4B (depth)","4B","Depth prune + distill","58.7"],
         ["MN-Minitron-8B","8B","Width prune + distill","69.5"]],
        col_widths=[2.5*inch,0.8*inch,2.2*inch,2.0*inch])
    story.append(t)


def sec_modern_llm(story):
    story += section_break("Modern LLM Architecture & Serving", C_NAVY)

    story.append(h2("Transformer Architecture Overview"))
    t = make_table(
        ["Component","Formula/Detail","Notes"],
        [["Self-Attention","Q=XW_Q, K=XW_K, V=XW_V; softmax(QK^T/√d_k)V","Multi-head: H independent heads, concat"],
         ["FFN","FFN(x) = max(0, xW_1+b_1)W_2+b_2","~2/3 of transformer FLOPs"],
         ["Layer Norm","(x−μ)/σ × γ + β","Pre-norm (modern) vs post-norm"],
         ["Positional Encoding","RoPE (rotary, modern) or learned","RoPE enables longer context generalization"],
         ["KV Cache","Cache K,V from all previous tokens","Size: 64 KB/token (8B model), 160 KB/token (70B)"]],
        col_widths=[1.5*inch,2.5*inch,3.5*inch])
    story.append(t)
    story.append(sp())

    story.append(h2("LLM Inference: Prefill vs. Decode"))
    t = make_table(
        ["Phase","Description","Bottleneck"],
        [["Prefill","Process all input tokens; populate KV cache","Compute-bound (large parallel matmuls)"],
         ["Decode","Generate one token at a time; read full KV cache","Memory-bandwidth-bound (read all weights per token)"]],
        col_widths=[1.2*inch,3.3*inch,3.0*inch])
    story.append(t)
    story.append(exam_tip("Decode phase = memory-bandwidth-bound. Each generated token requires reading ALL model weights."))
    story.append(sp())

    story.append(h2("Mixture of Experts (MoE)"))
    story.append(bullet("Replace dense FFN with N experts; router activates top-k per token"))
    story.append(bullet("DeepSeek-V3: 671B total params, 37B active per token (top-8 of 256 experts)"))
    story.append(bullet("Training cost: ~$5.5M — fraction of dense equivalent (GPT-4 est. $100M+)"))
    story.append(bullet("Communication: AllToAll for routing tokens to experts across GPUs"))
    story.append(key_fact("MoE = 'pruning by design' — only 5–15% of params active per token"))
    story.append(sp())

    story.append(h2("Speculative Decoding"))
    story.append(body("Small draft model proposes K tokens; large verifier model checks all K in one forward pass."))
    story.append(bullet("If all K tokens accepted → K× more throughput than sequential decoding"))
    story.append(bullet("If rejected at token j → discard tokens j+1...K, continue from j"))
    story.append(bullet("<b>Speedup: 2–4× in practice</b> when draft model is well-matched to verifier"))
    story.append(key_fact("Speculative decoding exploits that the verifier can check K tokens as fast as generating 1"))
    story.append(sp())

    story.append(h2("LoRA (Low-Rank Adaptation)"))
    story.append(formula("W' = W + ΔW = W + A·B  where A ∈ R^(d×r), B ∈ R^(r×k), r << min(d,k)"))
    story.append(bullet("Freeze pre-trained weights W; only train low-rank matrices A and B"))
    story.append(bullet("Typical rank r = 4, 8, 16 → massive parameter reduction vs full fine-tuning"))
    story.append(bullet("At inference: merge ΔW into W → no added latency"))
    story.append(key_fact("LoRA: same quality as full fine-tuning with 10,000× fewer trainable parameters for LLMs"))
    story.append(sp())

    story.append(h2("Combined Optimization Stack for LLM Deployment"))
    t = make_table(
        ["Technique","Benefit","When to Apply"],
        [["2:4 Structured Sparsity","2× throughput on Ampere+ Tensor Cores","Inference + training"],
         ["FP8 (H100)","2× more FLOPS vs BF16","H100 training/inference"],
         ["INT8 / INT4 PTQ","4–8× memory reduction","Edge/cloud inference"],
         ["FlashAttention","5× faster attention, 10× less memory","Always (use by default)"],
         ["Depth Pruning","Remove redundant middle layers","LLM compression"],
         ["Width Pruning","Reduce FFN/hidden dims","LLM compression"],
         ["Knowledge Distillation","Match quality at 40× less training cost","After pruning"],
         ["vLLM / PagedAttention","2–4× throughput for serving","Production inference"],
         ["Speculative Decoding","2–4× decode speedup","Low-latency interactive serving"]],
        col_widths=[1.8*inch,2.5*inch,3.2*inch])
    story.append(t)


def sec_practice_qa(story):
    story += section_break("PRACTICE QUESTIONS & ANSWERS", C_RED)

    story.append(exam_tip("These are representative of actual exam question types. Study all answers carefully."))
    story.append(sp())

    qa_pairs = [
        ("Q1. What is the CUDA warp size?",
         "B. 32 threads. Warps are the scheduling unit; all 32 execute the same instruction (SIMT)."),
        ("Q2. Why does tiled matrix multiplication with shared memory improve performance?",
         "C. Each element loaded from global memory is reused by many threads within the block, amortizing the HBM cost and increasing arithmetic intensity."),
        ("Q3. What does FlashAttention do differently from standard attention?",
         "B. It performs the same O(N²) FLOPs as standard attention but reduces HBM I/O by tiling and online softmax, keeping intermediates in SRAM. It does NOT reduce FLOPs."),
        ("Q4. Which collective is bandwidth-optimal for gradient synchronization (used by NCCL on rings)?",
         "B. Ring AllReduce. Cost = 2N(P-1)/P ≈ 2N per GPU — nearly independent of P. Parameter Server = 2N(P-1) — grows with P."),
        ("Q5. The decode phase of autoregressive LLM inference is:",
         "B. Memory-bandwidth-bound, because each generated token requires reading all model weights once. Arithmetic intensity is very low during decode."),
        ("Q6. Which PyTorch 2.0 component hooks into CPython's frame evaluation API?",
         "C. TorchDynamo. It intercepts Python bytecode at the frame level to capture FX graphs without source transformation."),
        ("Q7. What is the lowest-latency memory available to a CUDA thread?",
         "D. Registers. On-chip, per-thread, ~1 cycle latency. Followed by: Shared Memory (~20 cycles) → L2 (~200 cycles) → HBM (~400-600 cycles)."),
        ("Q8. DDP training, fixed global batch, 1→8 GPUs. What type of scaling?",
         "B. Strong scaling — same total work (fixed global batch), more GPUs → less time. Efficiency = T₁/(P×Tₚ)."),
        ("Q9. What is a key property of eager (imperative) computation in PyTorch?",
         "B. Operations are executed immediately as Python statements run. No need to build a full graph first; intermediate values are accessible immediately."),
        ("Q10. Which statements about CUDA shared memory are TRUE? (select all)",
         "A (scoped to single thread block), C (divided into 32 banks, conflicts possible), D (faster than global memory). NOT B — shared memory does NOT persist across kernel launches."),
        ("Q11. Which optimizations does torch.compile/TorchInductor apply?",
         "A (kernel fusion), B (Triton code generation), D (graph-level dead code elimination). NOT C — torch.compile does not automatically set up distributed training."),
        ("Q12. True or False: FlashAttention reduces the FLOP count below O(N²).",
         "FALSE. FlashAttention has the SAME O(N²) FLOPs as standard attention. It only reduces HBM I/O (memory traffic), not computation."),
        ("Q13. True or False: A slow straggler will stall synchronous data-parallel SGD.",
         "TRUE. Synchronous SGD requires ALL workers to finish before AllReduce and update. One slow GPU delays the entire batch."),
        ("Q14. CUDA vec_add bug: kernel uses int i = threadIdx.x without blockIdx.x*blockDim.x. What's wrong?",
         "B. Only block 0 computes correct results. Threads in other blocks have incorrect global indices (they all compute index 0–blockDim.x-1). Fix: i = blockIdx.x * blockDim.x + threadIdx.x."),
        ("Q15. Memory access pattern A (stride-1, consecutive) vs pattern B (stride-32). Which is faster?",
         "B → A is faster. Stride-1 coalesces all 32 warp threads into 1 HBM transaction (128-byte cache line). Stride-32 issues 32 separate transactions — 32× more HBM traffic."),
        ("Q16. Profiler: DataLoader=420ms, GPU idle=410ms, forward=38ms. What to fix?",
         "C. Fix the input pipeline: increase num_workers (more worker processes), set pin_memory=True, use prefetch_factor. GPU is starved waiting for data."),
        ("Q17. DDP, fixed global batch, 1→2→4→8 GPUs. Efficiency at 8 GPUs if T₁=400ms, T₈=70ms?",
         "B. Strong scaling. Efficiency = T₁/(P×Tₚ) = 400/(8×70) = 400/560 ≈ 0.71 = 71%."),
        ("Q18. GPU shows 12% utilization, CPU shows 100%. What's the bottleneck?",
         "B. Classic CPU-bound dataloader bottleneck. CPU cannot feed data fast enough → GPU waits. Fix: more DataLoader workers, prefetch, pin_memory."),
        ("Q19. Standard attention runs OOM for long sequences. Why?",
         "B. Q@K.T materializes a dense N×N attention matrix in HBM. Memory scales with N² — for N=32K tokens, this is 32K×32K = 1 billion elements = ~4 GB for FP32."),
        ("Q20. Why does .item() inside a training loop make it 67× slower?",
         "B. .item() forces host-device synchronization on every call — it waits for the GPU to finish computing that scalar before returning. This serializes GPU launches instead of pipelining them."),
    ]

    for i, (q, a) in enumerate(qa_pairs):
        story.append(KeepTogether([
            Paragraph(q, S["h3"]),
            Paragraph(f"<b>Answer:</b> {a}", S["body"]),
            sp(0.5),
        ]))


def sec_formulas_reference(story):
    story += section_break("QUICK REFERENCE — FORMULAS & CONSTANTS", C_NAVY)

    story.append(h2("All Key Formulas"))
    formulas = [
        ("Amdahl's Law", "S = 1 / [(1−p) + p/s]"),
        ("CUDA Global Index (1D)", "i = blockIdx.x * blockDim.x + threadIdx.x"),
        ("CUDA Global Index (2D)", "row = blockIdx.y*blockDim.y + threadIdx.y  ;  col = blockIdx.x*blockDim.x + threadIdx.x"),
        ("Self-Attention", "Attention(Q,K,V) = softmax(QK^T / √d_k) · V"),
        ("FlashAttention block sizes", "Bc = ⌈M/4d⌉  ;  Br = min(⌈M/4d⌉, d)"),
        ("FlashAttention HBM I/O", "O(N² · d² / M)  where M = SRAM size, d = head dim"),
        ("Quantization (float→int)", "q = clamp(round(r/S) + Z, q_min, q_max)"),
        ("Dequantization (int→float)", "r = S(q − Z)"),
        ("Scale S", "S = (r_max − r_min) / (q_max − q_min)"),
        ("Zero-point Z", "Z = round(q_min − r_min/S)"),
        ("Wanda pruning score", "S_{i,j} = |W_{i,j}| × ||X_j||_2"),
        ("SparseGPT saliency", "Score_{t,j} = W_{t,j}² / [H⁻¹]_{jj}  where H = 2XX^T + λI"),
        ("KL Divergence", "D_KL(P||Q) = Σ p(x) ln(p(x)/q(x))"),
        ("KD Total Loss", "L = α·L_student + (1−α)·L_distill"),
        ("Softmax with Temperature", "q_i = exp(z_i/T) / Σ_j exp(z_j/T)"),
        ("Adam optimizer (combined)", "w += −α · V̂_t / (√Ŝ_t + ε)  where V̂ = 1st moment, Ŝ = 2nd moment"),
        ("Ring AllReduce cost/GPU", "2N(P−1)/P  ≈ 2N for large P"),
        ("Scaling Efficiency (Strong)", "E = T₁ / (P × T_P)"),
        ("SM Occupancy", "Active Warps per SM / Max Warps per SM"),
        ("Roofline FLOPS bound", "FLOPS ≤ min(Peak FLOPS, AI × Memory BW)"),
        ("Arithmetic Intensity", "AI = #FLOP / DRAM_bytes  [FLOP/byte]"),
        ("LoRA weight update", "W' = W + AB  where A ∈ R^(d×r), B ∈ R^(r×k), r << d,k"),
    ]

    for name, f in formulas:
        story.append(h3(name))
        story.append(formula(f))
        story.append(sp(0.3))
    story.append(sp())

    story.append(h2("Constants & Threshold Values"))
    t = make_table(
        ["Constant","Value"],
        [["Warp size","32 threads (always)"],
         ["Max threads per block","1024"],
         ["Shared memory banks","32 banks, 4-byte granularity"],
         ["FP32 → FP16 reduction","2× memory"],
         ["FP32 → INT8 reduction","4× memory"],
         ["FP32 → INT4 reduction","8× memory"],
         ["Adam default β₁","0.9"],
         ["Adam default β₂","0.95–0.999 (LLMs: 0.95)"],
         ["AdamW weight decay","0.1 (LLaMA)"],
         ["FP32 exponent bits","8"],
         ["FP16 exponent bits","5"],
         ["BF16 exponent bits","8 (same as FP32 — same range)"],
         ["FP8 E4M3 max value","448"],
         ["FP8 E5M2 max value","57,344"],
         ["INT8 range (signed)","[−128, 127]"],
         ["BERT training speedup (FlashAttention)","~15% faster (20→17.4 min MLPerf)"],
         ["2:4 sparsity hardware speedup","Up to 2× on A100 Sparse Tensor Cores"],
         ["Minitron training cost reduction","40× fewer tokens than from scratch"],
         ["LLaMA 2 optimizer","AdamW, β₁=0.9, β₂=0.95, LR=3e-4 cosine"],
         ["DeepSeek-V3 total/active params","671B total, 37B active per token"],
         ["ZeRO-3 memory reduction (N=64)","~64× — from 120 GB to ~2 GB per GPU"],
         ["vLLM throughput gain","2–4× over static batching"],
         ["Speculative decoding speedup","2–4× end-to-end"]],
        col_widths=[3.5*inch,4.0*inch])
    story.append(t)


# ── Page Template ─────────────────────────────────────────────────────────────
def on_first_page(canvas, doc):
    pass

def on_later_pages(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 7)
    canvas.setFillColor(C_DGRAY)
    canvas.drawString(0.75*inch, 0.4*inch, "HPML Final Exam Study Notes — Columbia University Spring 2026")
    canvas.drawRightString(7.75*inch, 0.4*inch, f"Page {doc.page}")
    canvas.restoreState()


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    global S
    S = build_styles()

    doc = SimpleDocTemplate(
        OUTPUT_PATH,
        pagesize=letter,
        rightMargin=0.75*inch,
        leftMargin=0.75*inch,
        topMargin=0.75*inch,
        bottomMargin=0.6*inch,
        title="HPML Final Exam Study Notes",
        author="Rajvardhan Patil — Columbia University",
    )

    story = []
    cover_page(story)
    numbers_cheatsheet(story)
    sec_hpc_intro(story)
    sec_pytorch_python(story)
    sec_cuda_basics(story)
    sec_cuda_advanced(story)
    sec_ddl(story)
    sec_quantization(story)
    sec_pruning(story)
    sec_flash_attention(story)
    sec_kd(story)
    sec_modern_llm(story)
    sec_practice_qa(story)
    sec_formulas_reference(story)

    doc.build(story, onFirstPage=on_first_page, onLaterPages=on_later_pages)
    print(f"PDF generated: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
