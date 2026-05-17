#!/usr/bin/env python3
"""
HPML Deep-Dive Study Notes — Lectures 1,2,3,4,5,6,9
Full theory + worked numerical examples for Adam & Amdahl's Law
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
import datetime

OUT = "/Users/rajvardhan/Desktop/Projects/HPML/HPML_DeepDive_Notes.pdf"

# ── Palette ───────────────────────────────────────────────────────────────────
NAVY   = colors.HexColor("#0f172a")
BLUE   = colors.HexColor("#1d4ed8")
LBLUE  = colors.HexColor("#dbeafe")
DBLUE  = colors.HexColor("#1e40af")
GREEN  = colors.HexColor("#14532d")
LGREEN = colors.HexColor("#dcfce7")
ORANGE = colors.HexColor("#9a3412")
LORAN  = colors.HexColor("#ffedd5")
PURPLE = colors.HexColor("#581c87")
LPURP  = colors.HexColor("#f3e8ff")
RED    = colors.HexColor("#7f1d1d")
LRED   = colors.HexColor("#fee2e2")
TEAL   = colors.HexColor("#134e4a")
LTEAL  = colors.HexColor("#ccfbf1")
GRAY   = colors.HexColor("#374151")
LGRAY  = colors.HexColor("#f8fafc")
MGRAY  = colors.HexColor("#e2e8f0")
DGRAY  = colors.HexColor("#94a3b8")
BLACK  = colors.HexColor("#0f172a")
WHITE  = colors.white

# ── Styles ────────────────────────────────────────────────────────────────────
def S():
    d = {}
    d["h1"] = ParagraphStyle("h1", fontSize=16, fontName="Helvetica-Bold",
        textColor=WHITE, leading=22, spaceBefore=10, spaceAfter=6)
    d["h2"] = ParagraphStyle("h2", fontSize=13, fontName="Helvetica-Bold",
        textColor=NAVY, leading=18, spaceBefore=10, spaceAfter=4,
        borderPadding=(0,0,3,0))
    d["h3"] = ParagraphStyle("h3", fontSize=11, fontName="Helvetica-Bold",
        textColor=BLUE, leading=15, spaceBefore=7, spaceAfter=3)
    d["h4"] = ParagraphStyle("h4", fontSize=10, fontName="Helvetica-Bold",
        textColor=DBLUE, leading=14, spaceBefore=5, spaceAfter=2)
    d["body"] = ParagraphStyle("body", fontSize=9, fontName="Helvetica",
        textColor=BLACK, leading=13, spaceBefore=2, spaceAfter=2,
        alignment=TA_JUSTIFY)
    d["bullet"] = ParagraphStyle("bullet", fontSize=9, fontName="Helvetica",
        textColor=BLACK, leading=13, spaceBefore=1, spaceAfter=1,
        leftIndent=14, firstLineIndent=-10)
    d["sub"] = ParagraphStyle("sub", fontSize=8.5, fontName="Helvetica",
        textColor=GRAY, leading=12, spaceBefore=1, spaceAfter=1,
        leftIndent=26, firstLineIndent=-10)
    d["code"] = ParagraphStyle("code", fontSize=7.8, fontName="Courier",
        textColor=NAVY, leading=11, spaceBefore=2, spaceAfter=2,
        leftIndent=14, backColor=LGRAY,
        borderPadding=(4,8,4,8))
    d["formula"] = ParagraphStyle("formula", fontSize=9.5, fontName="Courier-Bold",
        textColor=PURPLE, leading=14, spaceBefore=4, spaceAfter=4,
        alignment=TA_CENTER, backColor=LPURP,
        borderPadding=(5,10,5,10))
    d["worked"] = ParagraphStyle("worked", fontSize=9, fontName="Courier",
        textColor=TEAL, leading=13, spaceBefore=2, spaceAfter=2,
        leftIndent=12, backColor=LTEAL,
        borderPadding=(4,8,4,8))
    d["key"] = ParagraphStyle("key", fontSize=9, fontName="Helvetica-Bold",
        textColor=GREEN, leading=13, spaceBefore=3, spaceAfter=3,
        leftIndent=10, backColor=LGREEN,
        borderPadding=(4,8,4,8))
    d["warn"] = ParagraphStyle("warn", fontSize=9, fontName="Helvetica-Bold",
        textColor=ORANGE, leading=13, spaceBefore=3, spaceAfter=3,
        leftIndent=10, backColor=LORAN,
        borderPadding=(4,8,4,8))
    d["exam"] = ParagraphStyle("exam", fontSize=9, fontName="Helvetica-Bold",
        textColor=RED, leading=13, spaceBefore=3, spaceAfter=3,
        leftIndent=10, backColor=LRED,
        borderPadding=(4,8,4,8))
    d["step"] = ParagraphStyle("step", fontSize=9, fontName="Helvetica",
        textColor=NAVY, leading=13, spaceBefore=2, spaceAfter=2,
        leftIndent=14, backColor=LBLUE,
        borderPadding=(4,8,4,8))
    return d

ST = S()

# ── Helpers ───────────────────────────────────────────────────────────────────
def h1(text, color=NAVY):
    t = Table([[Paragraph(text, ST["h1"])]], colWidths=[7.5*inch])
    t.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,-1),color),
        ("TOPPADDING",(0,0),(-1,-1),8),("BOTTOMPADDING",(0,0),(-1,-1),8),
        ("LEFTPADDING",(0,0),(-1,-1),12),("RIGHTPADDING",(0,0),(-1,-1),12),
    ]))
    return t

def h2(t): return Paragraph(t, ST["h2"])
def h3(t): return Paragraph(t, ST["h3"])
def h4(t): return Paragraph(t, ST["h4"])
def body(t): return Paragraph(t, ST["body"])
def b(t, ind=0): return Paragraph(("◦ " if ind else "• ")+t, ST["sub"] if ind else ST["bullet"])
def code(t): return Paragraph(t, ST["code"])
def formula(t): return Paragraph(t, ST["formula"])
def worked(t): return Paragraph(t, ST["worked"])
def key(t): return Paragraph("✓ "+t, ST["key"])
def warn(t): return Paragraph("⚠ "+t, ST["warn"])
def exam(t): return Paragraph("★ EXAM: "+t, ST["exam"])
def step(t): return Paragraph(t, ST["step"])
def sp(n=1): return Spacer(1, n*0.12*inch)
def hr(): return HRFlowable(width="100%",thickness=0.4,color=MGRAY,spaceAfter=3,spaceBefore=3)

def tbl(headers, rows, widths=None, hcol=NAVY, stripe=True):
    if widths is None:
        n = len(headers); widths=[7.5*inch/n]*n
    data=[[Paragraph(f"<b><font color='white'>{h}</font></b>",ST["body"]) for h in headers]]
    for row in rows:
        data.append([Paragraph(str(c),ST["body"]) for c in row])
    t=Table(data,colWidths=widths)
    ts=[("BACKGROUND",(0,0),(-1,0),hcol),
        ("GRID",(0,0),(-1,-1),0.3,MGRAY),
        ("TOPPADDING",(0,0),(-1,-1),4),("BOTTOMPADDING",(0,0),(-1,-1),4),
        ("LEFTPADDING",(0,0),(-1,-1),5),("RIGHTPADDING",(0,0),(-1,-1),5),
        ("VALIGN",(0,0),(-1,-1),"TOP")]
    if stripe:
        for i in range(1,len(data)):
            if i%2==0: ts.append(("BACKGROUND",(0,i),(-1,i),LGRAY))
    t.setStyle(TableStyle(ts)); return t

def section(title, color=NAVY):
    return [PageBreak(), h1(title, color), sp()]

# ── COVER ─────────────────────────────────────────────────────────────────────
def cover(story):
    banner = Table([[
        Paragraph("HPML", ParagraphStyle("t1",fontSize=54,fontName="Helvetica-Bold",
            textColor=WHITE,alignment=TA_CENTER)),
        ],[
        Paragraph("Deep-Dive Study Notes", ParagraphStyle("t2",fontSize=20,fontName="Helvetica",
            textColor=colors.HexColor("#93c5fd"),alignment=TA_CENTER,leading=26)),
        ],[
        Paragraph("Lectures 1 · 2 · 3 · 4 · 5 · 6 · 9 — Theory + Worked Numericals",
            ParagraphStyle("t3",fontSize=13,fontName="Helvetica-Bold",
            textColor=LGRAY,alignment=TA_CENTER,leading=18)),
        ]], colWidths=[7.5*inch])
    banner.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,-1),NAVY),
        ("TOPPADDING",(0,0),(-1,-1),20),("BOTTOMPADDING",(0,0),(-1,-1),20),
        ("LEFTPADDING",(0,0),(-1,-1),20),("RIGHTPADDING",(0,0),(-1,-1),20),
    ]))
    story.append(banner); story.append(sp(2))

    info = [["Course","COMS E6998 — High-Performance Machine Learning"],
            ["University","Columbia University · Spring 2026"],
            ["Instructor","Dr. Kaoutar El Maghraoui"],
            ["Exam","Monday May 11 · 7–9 PM · Schermerhorn 614"],
            ["Format","Closed book · 1 cheat sheet · ~25–35 questions"],
            ["Generated",datetime.datetime.now().strftime("%B %d, %Y")]]
    t=Table([[Paragraph(f"<b>{r[0]}</b>",ST["body"]),Paragraph(r[1],ST["body"])] for r in info],
            colWidths=[1.8*inch,5.7*inch])
    t.setStyle(TableStyle([
        ("GRID",(0,0),(-1,-1),0.3,MGRAY),
        ("BACKGROUND",(0,0),(0,-1),LGRAY),
        ("TOPPADDING",(0,0),(-1,-1),5),("BOTTOMPADDING",(0,0),(-1,-1),5),
        ("LEFTPADDING",(0,0),(-1,-1),8),("RIGHTPADDING",(0,0),(-1,-1),8),
    ]))
    story.append(t); story.append(sp(2))
    story.append(h2("What This Document Covers"))
    story.append(body("This document provides deep theoretical explanations and fully worked numerical "
        "examples for the seven lectures that form the core of the HPML final exam. "
        "Every important concept is explained from first principles, with intuition, "
        "worked examples, and common pitfalls clearly called out."))
    story.append(sp())
    story.append(tbl(["Section","Topic","Key Numericals"],
        [["L1","HPC Intro & AI Landscape","Compute scaling, energy costs, performance gains"],
         ["L2","Performance Methodology","Amdahl's Law (5 worked examples), Roofline Model, Optimizer formulas"],
         ["L3","Python & PyTorch Performance","Bytecode tracing, torch.compile stack, autograd math"],
         ["L4","CUDA Basics","Thread indexing (1D/2D worked), memory hierarchy numbers"],
         ["L5","Advanced CUDA","Coalescing, occupancy, tiled matmul, CUDA streams"],
         ["L6","Distributed Deep Learning","Ring AllReduce BW, ZeRO memory (exact), pipeline bubbles"],
         ["L9","FlashAttention & Efficient Transformers","IO tiling analysis, online softmax math, block size calc"]],
        widths=[0.6*inch,2.5*inch,4.4*inch]))
    story.append(PageBreak())

# ═══════════════════════════════════════════════════════════════════════════════
# L1 — HPC INTRO
# ═══════════════════════════════════════════════════════════════════════════════
def sec_l1(story):
    story += section("LECTURE 1 — HPC Introduction & AI Landscape", NAVY)

    story.append(h2("1.1  Why HPML? The Compute Explosion"))
    story.append(body("Training frontier AI models now requires computational resources that were unimaginable "
        "a decade ago. The table below shows the staggering growth from AlexNet (2012) to GPT-4 (2023):"))
    story.append(sp(0.5))
    story.append(tbl(["Model","Year","Parameters","Compute (PF-days)","Est. Cost"],
        [["AlexNet","2012","60 M","0.01","~$1K"],
         ["GPT-2","2019","1.5 B","40","~$50K"],
         ["GPT-3","2020","175 B","3,640","$4–12M"],
         ["GPT-4","2023","~1.8 T","~100,000","$100M+"]],
        widths=[1.2*inch,0.7*inch,1.2*inch,1.8*inch,1.6*inch]))
    story.append(key("Training costs have grown 10,000× in a decade — efficiency is now an economic necessity."))
    story.append(sp())

    story.append(h2("1.2  Training Energy Costs"))
    story.append(body("Energy consumption has become a critical constraint. Note the super-linear scaling: "
        "a 3-order-of-magnitude increase in parameters causes a 4-order-of-magnitude increase in energy."))
    story.append(tbl(["Model","Energy (MWh)","Equivalent Homes Powered"],
        [["GPT-2","1.7","~170 homes for 1 day"],
         ["Llama 2","688","~60K homes for 1 day"],
         ["GPT-3","1,287","~112K homes for 1 day"],
         ["PaLM","3,436","~300K homes for 1 day"],
         ["GPT-4","51,773","~4.5M homes for 1 day"]],
        widths=[1.5*inch,1.5*inch,4.5*inch]))
    story.append(sp())

    story.append(h2("1.3  The Parallelism Shift — Where Performance Gains Come From"))
    story.append(body("Single-core clock speeds plateaued around 2004 (the 'power wall'). "
        "Since then, ALL meaningful performance gains have come from parallelism, not transistor speed. "
        "This is why GPU computing and distributed systems are the foundation of modern ML:"))
    story.append(tbl(["Era","From Transistors","From Parallelism","Takeaway"],
        [["~1997 (Tera scale)","32×","32×","Balanced"],
         ["~2008 (Peta scale)","8×","128×","Parallelism dominates"],
         ["~2020 (Exa scale)","1.5×","670×","Almost ALL from parallelism"]],
        widths=[1.8*inch,1.5*inch,1.8*inch,2.4*inch]))
    story.append(key("670× gain from parallelism vs only 1.5× from transistors — this IS why you study HPML."))
    story.append(sp())

    story.append(h2("1.4  Inference Dominates Lifecycle Compute"))
    story.append(body("A common misconception is that training is the expensive part. In reality:"))
    story.append(b("<b>Inference costs up to 90% of total lifecycle compute</b> — training only ~10%."))
    story.append(b("For every ChatGPT request, the model must load ~140 GB of weights (Llama-3-70B) just to generate one token."))
    story.append(b("KV cache size: 64 KB/token (8B model), 160 KB/token (70B model). At batch=8, 128K context → <b>64 GB of KV cache</b>."))
    story.append(sp())

    story.append(h2("1.5  GenAI Inference — Two Phases"))
    story.append(body("Unlike traditional ML (stateless: input → output), GenAI inference is stateful:"))
    story.append(tbl(["Phase","Description","Bottleneck","Arithmetic Intensity"],
        [["Prefill","All input tokens processed; KV cache populated","Compute-bound (large parallel matmuls)","High"],
         ["Decode","One token generated at a time; reads full KV cache","Memory-bandwidth-bound","~20 FLOP/byte (vs H100 ridge ~625)"]],
        widths=[1.0*inch,2.5*inch,1.8*inch,2.2*inch]))
    story.append(exam("Decode = memory-bandwidth-bound. Hardware utilization is ~3% of peak — the central engineering challenge."))
    story.append(sp())

    story.append(h2("1.6  HPC Design Principles for AI"))
    story.append(body("Four core HPC principles, reimagined for AI superclusters:"))
    story.append(b("<b>Partition Model:</b> Separate service, compute, and I/O nodes. GPU nodes do no OS work."))
    story.append(b("<b>Network Topology:</b> Torus or Dragonfly for HPC; NVLink/NVSwitch (900 GB/s) + InfiniBand (400 Gb/s) for AI clusters."))
    story.append(b("<b>Balance of Components:</b> Compute speed must match memory bandwidth. Imbalance = GPU starvation."))
    story.append(b("<b>Scalable System Software:</b> Minimal OS overhead on compute nodes; Parallel File System (GPFS) for I/O."))

# ═══════════════════════════════════════════════════════════════════════════════
# L2 — PERFORMANCE METHODOLOGY
# ═══════════════════════════════════════════════════════════════════════════════
def sec_l2(story):
    story += section("LECTURE 2 — Performance Methodology, Roofline & Optimizers", DBLUE)

    story.append(h2("2.1  The 3-Step Optimization Cycle"))
    story.append(body("Performance optimization is never a one-shot exercise. It is a <b>cyclical, iterative process</b>:"))
    story.append(sp(0.5))
    t = Table([
        [Paragraph("<b>Step 1: MEASURE</b>", ST["h3"]),
         Paragraph("<b>Step 2: ANALYZE</b>", ST["h3"]),
         Paragraph("<b>Step 3: OPTIMIZE</b>", ST["h3"])],
        [Paragraph("• Execute workload\n• Profile with sampling or counters\n• Trace execution\n• Time sections precisely", ST["body"]),
         Paragraph("• Identify Critical Path (highest Amdahl p)\n• Identify Bottleneck: compute vs memory\n• Apply Roofline Model\n• Understand data movement", ST["body"]),
         Paragraph("• Cache blocking if memory-bound\n• Vectorization/SIMD if compute-bound\n• Reduce allocations\n• Fuse kernels\n→ then go back to Step 1", ST["body"])],
    ], colWidths=[2.5*inch, 2.5*inch, 2.5*inch])
    t.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,0),LBLUE),
        ("GRID",(0,0),(-1,-1),0.3,MGRAY),
        ("TOPPADDING",(0,0),(-1,-1),6),("BOTTOMPADDING",(0,0),(-1,-1),6),
        ("LEFTPADDING",(0,0),(-1,-1),6),("RIGHTPADDING",(0,0),(-1,-1),6),
        ("VALIGN",(0,0),(-1,-1),"TOP"),
    ]))
    story.append(t); story.append(sp())

    # ── AMDAHL'S LAW ──────────────────────────────────────────────────────────
    story.append(h2("2.2  Amdahl's Law — Complete Theory & 5 Worked Examples"))
    story.append(body("<b>Statement:</b> The overall speedup of a system is fundamentally limited by the "
        "fraction of time that cannot be parallelized. No matter how fast you make the parallelizable "
        "portion, the serial fraction sets a hard ceiling."))
    story.append(formula("S = 1 / [(1 − p) + p/s]"))
    story.append(body("<b>Variables:</b> S = overall speedup of the application; "
        "p = fraction of execution time spent in the section being improved (0 ≤ p ≤ 1); "
        "s = speedup of that improved section; (1−p) = serial (unimproved) fraction."))
    story.append(body("<b>Limiting case:</b> As s → ∞ (infinitely fast improved section), "
        "S → 1/(1−p). This is the absolute maximum speedup possible regardless of how good your optimization is."))
    story.append(sp(0.5))

    story.append(h3("Worked Example 1 — Basic Application with Two Sections"))
    story.append(body("Application: Section A takes 75% of time, Section B takes 25%. "
        "What is the speedup if we make B 5× faster? What if we make A 2× faster?"))
    story.append(worked("Given: p_A = 0.75, p_B = 0.25, total time = 100s"))
    story.append(worked(""))
    story.append(worked("Case 1: Make B 5× faster (s=5, p=0.25):"))
    story.append(worked("  S = 1 / [(1 − 0.25) + 0.25/5]"))
    story.append(worked("  S = 1 / [0.75 + 0.05]"))
    story.append(worked("  S = 1 / 0.80 = 1.25×  ← only 25% faster despite 5× optimization!"))
    story.append(worked(""))
    story.append(worked("Case 2: Make A 2× faster (s=2, p=0.75):"))
    story.append(worked("  S = 1 / [(1 − 0.75) + 0.75/2]"))
    story.append(worked("  S = 1 / [0.25 + 0.375]"))
    story.append(worked("  S = 1 / 0.625 = 1.60×  ← 60% faster with only 2× optimization!"))
    story.append(worked(""))
    story.append(worked("LESSON: Optimizing A (larger section) 2× beats optimizing B 5×."))
    story.append(worked("         ALWAYS profile first and optimize the CRITICAL PATH (high p)."))
    story.append(sp())

    story.append(h3("Worked Example 2 — Parallel GPU Training (the Classic Case)"))
    story.append(body("A neural network training loop: 90% of time is GPU matrix operations (parallelizable), "
        "10% is data loading and CPU overhead (serial). How fast can we train with more GPUs?"))
    story.append(worked("Given: p = 0.90 (GPU compute), 1-p = 0.10 (serial overhead)"))
    story.append(worked(""))
    story.append(worked("With 1 GPU  (s=1):   S = 1/[0.10 + 0.90/1]  = 1/1.00 = 1.0×"))
    story.append(worked("With 4 GPUs (s=4):   S = 1/[0.10 + 0.90/4]  = 1/[0.10+0.225] = 1/0.325 = 3.08×"))
    story.append(worked("With 8 GPUs (s=8):   S = 1/[0.10 + 0.90/8]  = 1/[0.10+0.1125] = 1/0.2125 = 4.71×"))
    story.append(worked("With 16 GPUs (s=16): S = 1/[0.10 + 0.90/16] = 1/[0.10+0.0563] = 1/0.1563 = 6.40×"))
    story.append(worked("With 100 GPUs(s=100):S = 1/[0.10 + 0.90/100]= 1/[0.10+0.009]  = 1/0.109  = 9.17×"))
    story.append(worked("Theoretical max (s→∞):S = 1/(1-0.90) = 1/0.10 = 10×  ← HARD CEILING"))
    story.append(worked(""))
    story.append(worked("Efficiency at 8 GPUs = S/(num_GPUs) = 4.71/8 = 58.9%  [strong scaling efficiency]"))
    story.append(key("With 10% serial overhead, you can NEVER exceed 10× speedup, even with infinite GPUs!"))
    story.append(sp())

    story.append(h3("Worked Example 3 — DDP Training with Communication Overhead"))
    story.append(body("Synchronous DDP: 80% compute (forward+backward), 15% AllReduce gradient sync, "
        "5% data loading. We can parallelize only the 80% compute portion."))
    story.append(worked("Given: p = 0.80 (parallelizable compute), 1-p = 0.20 (serial: comm + data load)"))
    story.append(worked(""))
    story.append(worked("With 1 GPU:   S = 1.0×"))
    story.append(worked("With 4 GPUs:  S = 1/[0.20 + 0.80/4]  = 1/[0.20+0.20] = 1/0.40 = 2.50×"))
    story.append(worked("With 8 GPUs:  S = 1/[0.20 + 0.80/8]  = 1/[0.20+0.10] = 1/0.30 = 3.33×"))
    story.append(worked("With 16 GPUs: S = 1/[0.20 + 0.80/16] = 1/[0.20+0.05] = 1/0.25 = 4.00×"))
    story.append(worked("Max speedup (s→∞): 1/0.20 = 5.0×"))
    story.append(worked(""))
    story.append(worked("Efficiency at 8 GPUs = 3.33/8 = 41.6%  ← poor! Serial fraction too large."))
    story.append(worked("FIX: Overlap AllReduce with backward pass computation (PyTorch DDP does this)"))
    story.append(worked("     If overlap makes 'p' effectively 0.95: max speedup becomes 1/0.05 = 20×"))
    story.append(sp())

    story.append(h3("Worked Example 4 — Maximum Speedup Calculation"))
    story.append(body("What fraction of code must be parallelized to achieve a 10× speedup with 16 processors?"))
    story.append(worked("Given: S_target = 10×, s = 16 processors"))
    story.append(worked("Solve Amdahl's formula for p:"))
    story.append(worked("  10 = 1 / [(1-p) + p/16]"))
    story.append(worked("  (1-p) + p/16 = 1/10 = 0.10"))
    story.append(worked("  1 - p + p/16 = 0.10"))
    story.append(worked("  1 - 0.90 = p - p/16 = p(1 - 1/16) = p(15/16)"))
    story.append(worked("  p = 0.90 × (16/15) = 0.96"))
    story.append(worked(""))
    story.append(worked("ANSWER: You need at least 96% of the code to be parallelizable"))
    story.append(worked("        to achieve 10× speedup with 16 processors."))
    story.append(sp())

    story.append(h3("Worked Example 5 — Scaling Efficiency Calculation"))
    story.append(body("DDP training: serial time T₁ = 400s, parallel time T₈ = 70s with 8 GPUs. "
        "What is the scaling efficiency? What is the implicit serial fraction?"))
    story.append(worked("Speedup: S = T₁ / T_P = 400 / 70 = 5.71×"))
    story.append(worked("Efficiency: E = S / P = 5.71 / 8 = 71.4%"))
    story.append(worked(""))
    story.append(worked("Back-calculate serial fraction from Amdahl:"))
    story.append(worked("  5.71 = 1 / [(1-p) + p/8]"))
    story.append(worked("  (1-p) + p/8 = 1/5.71 = 0.175"))
    story.append(worked("  1 - 7p/8 = 0.175"))
    story.append(worked("  7p/8 = 0.825"))
    story.append(worked("  p = 0.825 × 8/7 = 0.943"))
    story.append(worked(""))
    story.append(worked("Serial fraction = 1 - p = 5.7%  → this is the AllReduce + overhead cost"))
    story.append(worked("To improve: reduce communication overhead with faster interconnect or"))
    story.append(worked("better overlap of AllReduce with backward computation."))
    story.append(sp())

    # ── ROOFLINE ──────────────────────────────────────────────────────────────
    story.append(h2("2.3  The Roofline Performance Model — Theory & Numericals"))
    story.append(body("The Roofline model answers: <b>'Is my kernel limited by compute or memory bandwidth?'</b> "
        "It provides a visual framework showing the maximum achievable FLOPS as a function of Arithmetic Intensity."))
    story.append(sp(0.5))
    story.append(h3("Key Concepts"))
    story.append(b("<b>Arithmetic Intensity (AI)</b> = FLOP count / bytes transferred from DRAM  [FLOP/byte]"))
    story.append(b("Small AI → memory-bound (limited by DRAM bandwidth)"))
    story.append(b("Large AI → compute-bound (limited by peak FLOPS)"))
    story.append(formula("Attainable FLOPS = min(Peak FLOPS, AI × Memory Bandwidth)"))
    story.append(formula("Crossover AI = Peak FLOPS / Memory Bandwidth"))
    story.append(sp(0.5))

    story.append(h3("Worked Example 1 — DAXPY (Double-Precision A·X + Y)"))
    story.append(body("Code: <b>Z[i] = A * (X[i] + Y[i])</b>  for i = 0..N-1"))
    story.append(worked("Count FLOPs: 1 multiply (A×X[i]) + 1 add (+Y[i]) = 2 FLOPs per iteration"))
    story.append(worked("Count bytes: Read X[i]=8B, Read Y[i]=8B, Write Z[i]=8B = 24 bytes"))
    story.append(worked("  (A is a scalar, loaded once and cached — not counted per iteration)"))
    story.append(worked(""))
    story.append(worked("Arithmetic Intensity = 2 FLOP / 24 bytes = 0.083 FLOP/byte"))
    story.append(worked(""))
    story.append(worked("Hardware (Columbia Xeon E5630 @ 2.53GHz):"))
    story.append(worked("  Peak FLOPS = 2.53 GHz × 4 cores × 8 FP64/cycle = 81.3 GFLOPS"))
    story.append(worked("  DRAM Bandwidth = 25.6 GB/s"))
    story.append(worked("  Crossover AI = 81.3 / 25.6 = 3.17 FLOP/byte"))
    story.append(worked(""))
    story.append(worked("Since AI = 0.083 < 3.17 → DAXPY is MEMORY-BOUND"))
    story.append(worked("Max attainable = 0.083 × 25.6 = 2.1 GFLOPS  (out of 81.3 peak)"))
    story.append(worked("Utilization = 2.1/81.3 = 2.6% of peak compute — very poor!"))
    story.append(worked("FIX: Cache blocking can raise effective AI by reusing data from L1/L2."))
    story.append(sp())

    story.append(h3("Worked Example 2 — 7-Point 3D Stencil"))
    story.append(body("Updates each 3D grid point using its 6 neighbors + itself."))
    story.append(worked("Count FLOPs: 6 additions + 1 multiply-by-(-6.0) = 7 FLOPs per point"))
    story.append(worked("Count bytes: Read 7 unique double values (8B each) + write 1 = 8 reads × 8B = 64B"))
    story.append(worked("  (new[] write = 8B, so total = 64+8 = 72B? Conservative: 64B from DRAM)"))
    story.append(worked(""))
    story.append(worked("Arithmetic Intensity = 7 / 64 = 0.109 FLOP/byte"))
    story.append(worked("Crossover = 3.17 FLOP/byte (same hardware)"))
    story.append(worked("0.109 < 3.17 → MEMORY-BOUND (still!)"))
    story.append(worked("Max attainable = 0.109 × 25.6 = 2.8 GFLOPS"))
    story.append(worked(""))
    story.append(worked("After cache blocking:"))
    story.append(worked("  Only unique data fetched from DRAM; neighbors reused from L1"))
    story.append(worked("  If L1 cache eliminates 75% of DRAM traffic: effective bytes = 16B"))
    story.append(worked("  New AI = 7/16 = 0.44 FLOP/byte → still memory-bound but 4× closer to crossover"))
    story.append(sp())

    # ── MEAN VALUES ───────────────────────────────────────────────────────────
    story.append(h2("2.4  Mean Values — Which Mean to Use and When"))
    story.append(tbl(["Mean Type","Formula","Use When","ML Example"],
        [["Arithmetic","(1/n)Σxᵢ","Adding raw values (times, latencies)","Average epoch time: (t₁+t₂+t₃)/3"],
         ["Harmonic","n / Σ(1/xᵢ)","Averaging rates/throughput","Average tokens/sec across runs"],
         ["Geometric","(∏xᵢ)^(1/n)","Averaging ratios and speedups","Average speedup across benchmarks"]],
        widths=[1.3*inch,2.0*inch,2.2*inch,2.0*inch]))
    story.append(sp(0.5))
    story.append(worked("Numerical example: Three runs give speedups of 2×, 4×, 8×"))
    story.append(worked("  Arithmetic mean = (2+4+8)/3 = 4.67×  ← WRONG for ratios"))
    story.append(worked("  Geometric  mean = (2×4×8)^(1/3) = 64^(1/3) = 4.0×  ← CORRECT"))
    story.append(exam("Speedup/ratio → Geometric mean. Throughput/rates → Harmonic mean. Times → Arithmetic mean."))
    story.append(sp())

    # ── SCALING ───────────────────────────────────────────────────────────────
    story.append(h2("2.5  Strong vs. Weak Scaling — Theory & Examples"))
    story.append(tbl(["Scaling Type","Definition","Goal","DL Example","Efficiency Formula"],
        [["Strong Scaling","Fixed total problem size; increase #GPUs",
          "Reduce time for same problem",
          "Train on same dataset faster with more GPUs",
          "E = T₁/(P × Tₚ)"],
         ["Weak Scaling","Problem size grows proportionally with #GPUs",
          "Keep runtime constant as workload grows",
          "Train larger model with more GPUs (each holds same per-GPU batch)",
          "E = T₁/Tₚ (ideally = 1.0)"]],
        widths=[1.2*inch,1.8*inch,1.5*inch,2.0*inch,1.0*inch]))
    story.append(sp(0.5))
    story.append(worked("Strong scaling example: Fixed global batch = 1024 samples"))
    story.append(worked("  1 GPU:   T = 400s → S = 1.0×, E = 100%"))
    story.append(worked("  2 GPUs:  T = 210s → S = 1.90×, E = 95%"))
    story.append(worked("  4 GPUs:  T = 112s → S = 3.57×, E = 89%"))
    story.append(worked("  8 GPUs:  T = 65s  → S = 6.15×, E = 76.9%"))
    story.append(worked("  16 GPUs: T = 40s  → S = 10.0×, E = 62.5%  ← efficiency drops with AllReduce overhead"))
    story.append(sp())

    # ── OPTIMIZERS ────────────────────────────────────────────────────────────
    story.append(h2("2.6  Optimizer Theory — From SGD to Adam (with Numericals)"))
    story.append(body("All modern optimizers for deep learning are built on the same foundation: "
        "<b>Exponential Moving Average (EMA)</b> of gradient statistics. Understanding EMA is the key to "
        "understanding all of Momentum, RMSProp, and Adam."))
    story.append(formula("EMA:  v_t = β·v_{t-1} + (1−β)·g_t"))
    story.append(body("With β=0.9, this effectively averages over the last 1/(1-0.9) = <b>10 steps</b>. "
        "With β=0.99, it averages over <b>100 steps</b>. Higher β = smoother, less reactive."))
    story.append(sp())

    story.append(h3("SGD (Baseline)"))
    story.append(formula("w_{t+1} = w_t − α·g_t       where g_t = ∂L/∂w_t"))
    story.append(body("Problem: noisy gradients cause oscillations. Each mini-batch gradient may point in "
        "slightly wrong direction. Learning rate must be small to avoid divergence → slow convergence."))
    story.append(sp())

    story.append(h3("Momentum"))
    story.append(formula("V_t = β·V_{t-1} + (1−β)·g_t      [EMA of gradients]"))
    story.append(formula("w_{t+1} = w_t − α·V_t"))
    story.append(body("Intuition: Instead of taking a pure step in the current gradient direction, we "
        "accumulate a 'velocity' that remembers past directions. This damps oscillations in steep narrow "
        "valleys and accelerates descent along consistent directions."))
    story.append(sp())

    story.append(h3("RMSProp"))
    story.append(formula("S_t = β·S_{t-1} + (1−β)·g_t²      [EMA of squared gradients]"))
    story.append(formula("w_{t+1} = w_t − (α / √(S_t + ε))·g_t"))
    story.append(body("Intuition: Divide the learning rate by the root-mean-square of recent gradients. "
        "Parameters that consistently receive large gradients get smaller effective LR (prevents explosion). "
        "Parameters with small gradients get larger effective LR (prevents vanishing). "
        "This is called 'per-parameter adaptive learning rates'."))
    story.append(sp())

    story.append(h3("Adam — Full Theory with Bias Correction"))
    story.append(body("Adam = Momentum (1st moment) + RMSProp (2nd moment) + bias correction."))
    story.append(formula("V_t = β₁·V_{t-1} + (1−β₁)·g_t       [1st moment: mean of gradients]"))
    story.append(formula("S_t = β₂·S_{t-1} + (1−β₂)·g_t²      [2nd moment: mean of sq. gradients]"))
    story.append(formula("V̂_t = V_t / (1 − β₁ᵗ)               [bias-corrected 1st moment]"))
    story.append(formula("Ŝ_t = S_t / (1 − β₂ᵗ)               [bias-corrected 2nd moment]"))
    story.append(formula("w_{t+1} = w_t − α · V̂_t / (√Ŝ_t + ε)"))
    story.append(sp(0.5))
    story.append(body("<b>Why bias correction?</b> V₀ = 0 and S₀ = 0 (initialized to zero). At early steps (small t), "
        "V_t and S_t are biased toward zero. Dividing by (1-β^t) compensates: "
        "at t=1 with β₁=0.9: correction factor = 1/(1-0.9) = 10 → scales up V₁ appropriately."))
    story.append(sp())

    story.append(h3("★ Worked Numerical Example: Adam Step-by-Step"))
    story.append(body("Scenario: Single scalar parameter w = 2.0. Gradients at steps 1,2,3: g = [0.5, 0.3, 0.8]. "
        "Hyperparams: α=0.01, β₁=0.9, β₂=0.999, ε=1e-8. Starting: V₀=0, S₀=0."))
    story.append(sp(0.5))
    story.append(worked("=== STEP t=1 (g₁ = 0.5) ==="))
    story.append(worked("  V₁ = 0.9 × 0 + (1−0.9) × 0.5     = 0 + 0.05 = 0.05"))
    story.append(worked("  S₁ = 0.999 × 0 + (1−0.999) × 0.25 = 0 + 0.00025 = 0.00025"))
    story.append(worked("  Bias correction:"))
    story.append(worked("    V̂₁ = 0.05 / (1 − 0.9¹)   = 0.05 / 0.1   = 0.50"))
    story.append(worked("    Ŝ₁ = 0.00025 / (1 − 0.999¹) = 0.00025 / 0.001 = 0.25"))
    story.append(worked("  Update:"))
    story.append(worked("    w₁ = 2.0 − 0.01 × 0.50 / (√0.25 + 1e-8)"))
    story.append(worked("       = 2.0 − 0.01 × 0.50 / 0.5"))
    story.append(worked("       = 2.0 − 0.01 × 1.0 = 2.0 − 0.010 = 1.990"))
    story.append(sp(0.3))
    story.append(worked("=== STEP t=2 (g₂ = 0.3) ==="))
    story.append(worked("  V₂ = 0.9 × 0.05 + 0.1 × 0.3     = 0.045 + 0.03 = 0.075"))
    story.append(worked("  S₂ = 0.999 × 0.00025 + 0.001 × 0.09 = 0.0002498 + 0.00009 = 0.0003398"))
    story.append(worked("  V̂₂ = 0.075 / (1 − 0.9²)   = 0.075 / 0.19   = 0.3947"))
    story.append(worked("  Ŝ₂ = 0.0003398 / (1 − 0.999²) = 0.0003398 / 0.001999 = 0.1700"))
    story.append(worked("  Update:"))
    story.append(worked("    w₂ = 1.990 − 0.01 × 0.3947 / (√0.1700 + 1e-8)"))
    story.append(worked("       = 1.990 − 0.01 × 0.3947 / 0.4123"))
    story.append(worked("       = 1.990 − 0.01 × 0.957 = 1.990 − 0.00957 = 1.9804"))
    story.append(sp(0.3))
    story.append(worked("=== STEP t=3 (g₃ = 0.8) ==="))
    story.append(worked("  V₃ = 0.9 × 0.075 + 0.1 × 0.8     = 0.0675 + 0.08 = 0.1475"))
    story.append(worked("  S₃ = 0.999 × 0.0003398 + 0.001 × 0.64 = 0.0003395 + 0.00064 = 0.0009795"))
    story.append(worked("  V̂₃ = 0.1475 / (1 − 0.9³)   = 0.1475 / 0.271  = 0.5443"))
    story.append(worked("  Ŝ₃ = 0.0009795/(1 − 0.999³) = 0.0009795/0.002997 = 0.3268"))
    story.append(worked("  Update:"))
    story.append(worked("    w₃ = 1.9804 − 0.01 × 0.5443 / (√0.3268 + 1e-8)"))
    story.append(worked("       = 1.9804 − 0.01 × 0.5443 / 0.5716"))
    story.append(worked("       = 1.9804 − 0.01 × 0.9522 = 1.9804 − 0.00952 = 1.9709"))
    story.append(sp(0.3))
    story.append(worked("Summary: w went 2.0 → 1.990 → 1.980 → 1.971 (moving toward minimum)"))
    story.append(worked("Note: Bias correction at t=1 was crucial — without it, V̂₁=0.05 would give"))
    story.append(worked("a much smaller step (0.05/0.5 = 0.1) vs correct (0.50/0.5 = 1.0)"))
    story.append(key("Bias correction is critical at early training steps. Adam without bias correction converges much more slowly at the start."))
    story.append(sp())

    story.append(h3("AdamW vs. Adam — The Critical Difference"))
    story.append(body("<b>Problem with Adam:</b> L2 regularization λ‖w‖² added to loss → gradient includes "
        "λw term → weight decay gets <i>scaled</i> by the adaptive learning rate. For large-gradient parameters, "
        "weight decay becomes negligible. This is mathematically inconsistent with the intent of regularization."))
    story.append(formula("Adam+L2: w_{t+1} = w_t − (α/(√Ŝ_t+ε))·(g_t + λ·w_t)  ← decay scaled by 1/√Ŝ_t"))
    story.append(formula("AdamW:   w_{t+1} = w_t − (α/(√Ŝ_t+ε))·g_t − α·λ·w_t  ← decay INDEPENDENT"))
    story.append(key("AdamW decouples weight decay from adaptive gradient scaling. De facto standard for all LLM training."))

# ═══════════════════════════════════════════════════════════════════════════════
# L3 — PYTHON & PYTORCH
# ═══════════════════════════════════════════════════════════════════════════════
def sec_l3(story):
    story += section("LECTURE 3 — Python & PyTorch Performance", BLUE)

    story.append(h2("3.1  Why Python is Slow — From First Principles"))
    story.append(body("To understand why PyTorch needs a C++/CUDA backend, you must understand what Python "
        "actually does when you write <b>x = a + b</b>. This seemingly simple operation expands into many interpreter steps:"))
    story.append(sp(0.5))
    story.append(tbl(["Step","What Python Actually Does","Cost"],
        [["1","LOAD_FAST: push 'a' onto evaluation stack","~1–5 ns"],
         ["2","LOAD_FAST: push 'b' onto evaluation stack","~1–5 ns"],
         ["3","Type check: are a and b numbers? Call __add__?","~10–20 ns"],
         ["4","Reference count update on 'a' and 'b'","~5 ns"],
         ["5","BINARY_ADD: dispatch to correct C function","~10 ns"],
         ["6","Allocate new PyObject on heap (28 bytes per float!)","~50–100 ns"],
         ["7","STORE_FAST: store result, update refcount","~5 ns"],
         ["TOTAL","~7 operations for one addition","~80–150 ns"]],
        widths=[0.5*inch,4.0*inch,3.0*inch]))
    story.append(worked("Compare: NumPy/BLAS 1000×1000 matmul = ~0.05s"))
    story.append(worked("         Python nested loop matmul    = ~45s"))
    story.append(worked("         Speedup factor               = ~900×"))
    story.append(key("Solution: Push ALL computation into compiled C++/CUDA kernels. Python = glue code only."))
    story.append(sp())

    story.append(h2("3.2  Memory Layout — Stack vs. Heap"))
    story.append(body("Understanding where data lives explains many performance characteristics:"))
    story.append(tbl(["Memory Region","Speed","Size","Who Uses It","Python Impact"],
        [["Registers","~1 cycle","64 bytes","CPU ALU intermediate results","Not accessible from Python"],
         ["Stack","~3 cycles","~8 MB","Function frames, local vars in C","C/CUDA kernels use this"],
         ["Heap","~100 cycles","GBs","Dynamic allocations (malloc)","ALL Python objects live here"],
         ["DRAM","~100 cycles","GBs","Main memory","Large tensors, model weights"]],
        widths=[1.3*inch,1.0*inch,0.8*inch,2.2*inch,2.2*inch]))
    story.append(body("Every Python float is a <b>heap-allocated PyObject</b> (28 bytes). "
        "Contrast with C float (4 bytes on stack) — 7× more memory, much slower access due to pointer indirection."))
    story.append(sp())

    story.append(h2("3.3  Python Execution Pipeline — Source to Bytecode"))
    story.append(body("Python compilation pipeline (happens on every import or source change):"))
    story.append(tbl(["Stage","Name","What Happens","Output"],
        [["1","Lexing & Parsing","Tokenize source; build Concrete Syntax Tree (CST)","CST preserves ALL info including whitespace"],
         ["2","AST Generation","Lower CST to Abstract Syntax Tree","AST removes syntactic sugar"],
         ["3","Semantic Analysis","Resolve names to scopes (local/global/nonlocal/free)","Symbol table"],
         ["4","CFG Construction","Lower AST to Control Flow Graph (basic blocks)","CFG"],
         ["5","Bytecode Emission","Generate stack-machine instructions; package into PyCodeObject","Bytecode (.pyc)"],
         ["6","Peephole Optimizer","Constant folding (2+2→4), dead code removal, short sequences","Optimized bytecode"],
         ["7","PVM Execution","ceval.c executes bytecode instruction by instruction","Program output"]],
        widths=[0.4*inch,1.2*inch,2.8*inch,3.1*inch]))
    story.append(sp())

    story.append(h3("Stack Machine Trace: d = a + b*c"))
    story.append(body("Python's VM is a <b>stack-based machine</b>. Every operation pushes/pops from a stack. "
        "Let a=2, b=3, c=4. The compiled bytecode and stack state:"))
    story.append(tbl(["Instruction","Operand","Stack After","Stack State"],
        [["LOAD_FAST","0 (a)","[a]","[2]"],
         ["LOAD_FAST","1 (b)","[a, b]","[2, 3]"],
         ["LOAD_FAST","2 (c)","[a, b, c]","[2, 3, 4]"],
         ["BINARY_MULTIPLY","—","[a, b*c]","[2, 12]"],
         ["BINARY_ADD","—","[a+b*c]","[14]"],
         ["STORE_FAST","3 (d)","[]","d = 14, stack empty"]],
        widths=[1.5*inch,1.2*inch,1.8*inch,3.0*inch]))
    story.append(sp())

    story.append(h2("3.4  torch.compile — The Modern Solution"))
    story.append(body("PyTorch 2.0's <b>torch.compile</b> is a JIT compiler that captures computation graphs "
        "from Python bytecode and compiles them to optimized GPU kernels — with <b>zero code changes</b>:"))
    story.append(code("model = torch.compile(model)   # That's it!"))
    story.append(sp(0.5))
    story.append(tbl(["Component","Role","Key Innovation","Output"],
        [["TorchDynamo","Graph Acquisition","Hooks into CPython frame evaluation API at bytecode level. "
          "Captures FX graph without tracing. Uses GUARDS to validate specialization.","FX graph"],
         ["AOTAutograd","Graph Lowering","Captures both forward AND backward graphs ahead-of-time. "
          "Enables joint optimization of training loop.","Joint fwd+bwd graph"],
         ["TorchInductor","Graph Compilation","Generates Triton (GPU) or C++/OpenMP (CPU) kernels. "
          "Applies fusion, tiling, vectorization automatically.","Compiled kernels"]],
        widths=[1.3*inch,1.3*inch,3.3*inch,1.6*inch]))
    story.append(sp(0.5))
    story.append(b("<b>Guards:</b> TorchDynamo inserts runtime checks (shape, dtype, device). If guard fails, "
        "it recompiles for the new specialization. This is why the first call is slow."))
    story.append(b("<b>Kernel Fusion:</b> TorchInductor fuses multiple ops (matmul + relu + dropout) into one kernel, "
        "eliminating intermediate tensor allocations and repeated HBM loads."))
    story.append(key("torch.compile speedups on A100: +38% TIMM, +76% TorchBench, +52% HuggingFace — zero code changes."))
    story.append(sp())

    story.append(h2("3.5  PyTorch Autograd — How Gradients are Computed"))
    story.append(body("Autograd builds a dynamic computation graph during the forward pass, "
        "then traverses it backward using the chain rule:"))
    story.append(formula("Chain rule: d/dx[h(g(x))] = (dh/dg)·(dg/dx)"))
    story.append(sp(0.5))
    story.append(worked("Concrete example: z = 2x² + 5x,  x = [[2, 2], [2, 2]]"))
    story.append(worked("  Analytical gradient: dz/dx = 4x + 5"))
    story.append(worked("  At x = 2: dz/dx = 4(2) + 5 = 13"))
    story.append(worked("  PyTorch confirms: x.grad = [[13, 13], [13, 13]]"))
    story.append(sp(0.5))
    story.append(b("<b>requires_grad=True:</b> tells autograd to track operations for gradient computation"))
    story.append(b("<b>loss.backward():</b> traverses graph backward via chain rule; accumulates gradients in .grad"))
    story.append(b("<b>torch.no_grad():</b> disables graph construction for weight updates (prevents tracking non-grad ops)"))
    story.append(warn("MUST call .zero_grad() before each backward pass — gradients ACCUMULATE by default!"))

# ═══════════════════════════════════════════════════════════════════════════════
# L4 — CUDA BASICS
# ═══════════════════════════════════════════════════════════════════════════════
def sec_l4(story):
    story += section("LECTURE 4 — CUDA Basics", BLUE)

    story.append(h2("4.1  CPU vs. GPU — Why They're Different by Design"))
    story.append(body("CPUs and GPUs are built for fundamentally different goals. Understanding this determines "
        "WHEN to use each and HOW to write efficient GPU code:"))
    story.append(tbl(["Property","CPU (Latency Optimized)","GPU (Throughput Optimized)"],
        [["Silicon Area","~60% for control + cache, ~40% for compute","~90% for arithmetic units (CUDA cores)"],
         ["Core Count","4–60 powerful cores","Thousands of simple cores"],
         ["Cache","Large L1/L2/L3 (reduces memory latency)","Small, programmer-managed (shared memory)"],
         ["Control","Branch prediction, out-of-order exec","Simple: no branch prediction, SIMT"],
         ["Best for","Sequential, complex control flow","Massively parallel, data-regular tasks"],
         ["Latency","~nanoseconds per op","~microseconds for kernel launch"],
         ["Sequential code","~10× faster than GPU","~10× slower than CPU"]],
        widths=[1.8*inch,2.7*inch,3.0*inch]))
    story.append(sp())

    story.append(h2("4.2  CUDA Thread Hierarchy — Complete Theory"))
    story.append(body("CUDA organizes threads in a three-level hierarchy. Each level maps to hardware:"))
    story.append(tbl(["Level","Maps to Hardware","Max Size","Key Properties"],
        [["Thread","CUDA Core (SP)","1","Has own registers and local memory; smallest unit"],
         ["Block","Streaming Multiprocessor (SM)","1024 threads","Threads share memory and can sync with __syncthreads()"],
         ["Grid","GPU Device","2³¹−1 × 65535 × 65535","One grid per kernel launch; blocks execute in any order"],
         ["Warp","32 consecutive threads","32","Scheduling unit; all execute SAME instruction (SIMT)"]],
        widths=[0.8*inch,2.0*inch,1.3*inch,3.4*inch]))
    story.append(sp())

    story.append(h2("4.3  Global Thread Index — Theory & Worked Examples"))
    story.append(body("The most critical formula in CUDA. <b>threadIdx</b> is only unique within its block. "
        "To get a globally unique index, you must combine block and thread indices:"))
    story.append(formula("i = blockIdx.x * blockDim.x + threadIdx.x  (1D)"))
    story.append(formula("Row = blockIdx.y*blockDim.y + threadIdx.y   Col = blockIdx.x*blockDim.x + threadIdx.x  (2D)"))
    story.append(sp(0.5))

    story.append(h3("Worked Example 1 — Basic 1D Index"))
    story.append(worked("Launch config: kernel<<<4, 256>>>(args)  (4 blocks of 256 threads)"))
    story.append(worked("Total threads = 4 × 256 = 1024"))
    story.append(worked(""))
    story.append(worked("Thread in Block 2, threadIdx.x = 3:"))
    story.append(worked("  global_i = blockIdx.x * blockDim.x + threadIdx.x"))
    story.append(worked("           = 2 * 256 + 3 = 515"))
    story.append(worked(""))
    story.append(worked("Thread in Block 0, threadIdx.x = 0:   global_i = 0"))
    story.append(worked("Thread in Block 0, threadIdx.x = 255: global_i = 255"))
    story.append(worked("Thread in Block 1, threadIdx.x = 0:   global_i = 256"))
    story.append(worked("Thread in Block 3, threadIdx.x = 255: global_i = 1023"))
    story.append(sp())

    story.append(h3("Worked Example 2 — kernel<<<3,4>>> Traces"))
    story.append(body("For launch config kernel<<<3,4>>>(a) — 3 blocks, 4 threads/block, 12 total threads:"))
    story.append(tbl(["Kernel Code","Result Array (a[0]...a[11])","Explanation"],
        [["a[i] = blockDim.x","4 4 4 4 4 4 4 4 4 4 4 4","blockDim.x=4 for every thread in every block"],
         ["a[i] = threadIdx.x","0 1 2 3 | 0 1 2 3 | 0 1 2 3","threadIdx resets to 0 at each block boundary"],
         ["a[i] = blockIdx.x","0 0 0 0 | 1 1 1 1 | 2 2 2 2","blockIdx same for all threads in a block"],
         ["a[i] = i (global)","0 1 2 3 4 5 6 7 8 9 10 11","unique global index = blockIdx×blockDim+threadIdx"]],
        widths=[1.8*inch,2.8*inch,2.9*inch]))
    story.append(exam("COMMON BUG: using a[i] = threadIdx.x only gives 0,1,2,3,0,1,2,3... NOT unique global indices!"))
    story.append(sp())

    story.append(h3("Worked Example 3 — 2D Image Indexing"))
    story.append(worked("Image: 8×8 pixels. Launch: <<<dim3(2,2), dim3(4,4)>>>"))
    story.append(worked("blockDim.x=4, blockDim.y=4, gridDim.x=2, gridDim.y=2"))
    story.append(worked(""))
    story.append(worked("Thread at blockIdx=(1,0), threadIdx=(1,2):"))
    story.append(worked("  Row = blockIdx.y * blockDim.y + threadIdx.y = 0*4 + 2 = 2"))
    story.append(worked("  Col = blockIdx.x * blockDim.x + threadIdx.x = 1*4 + 1 = 5"))
    story.append(worked("  Linear index (row-major) = Row * width + Col = 2*8 + 5 = 21"))
    story.append(worked("  So this thread processes pixel Image[2][5] = Image[21]"))
    story.append(sp())

    story.append(h2("4.4  CUDA Memory Model — Complete Hierarchy"))
    story.append(tbl(["Level","Location","Latency","BW","Scope","Managed By","A100 Size"],
        [["Registers","On-chip (per SM)","~1 cycle","~8 TB/s","Per thread (private)","Compiler","~256 KB/SM"],
         ["Shared Mem / L1","On-chip (per SM)","~20 cycles","~1.5 TB/s","Per block","Programmer","192 KB/SM"],
         ["L2 Cache","On-chip (GPU die)","~200 cycles","—","All SMs","Hardware","40 MB"],
         ["Global (HBM)","Off-chip","~400–600 cycles","2 TB/s","All SMs","Programmer","80 GB"],
         ["Constant","Off-chip (cached)","~1 (cached)","—","All (read-only)","Programmer","64 KB"]],
        widths=[1.0*inch,1.2*inch,0.9*inch,0.9*inch,1.0*inch,1.0*inch,1.5*inch]))
    story.append(sp(0.5))
    story.append(key("Register spilling = variables that don't fit in registers go to 'local memory' = ACTUALLY global memory = SLOW."))
    story.append(key("Shared memory is programmer-managed L1 cache. Use __shared__ to declare. Must sync with __syncthreads()."))
    story.append(sp())

    story.append(h2("4.5  CUDA Programming Pattern"))
    story.append(code("// 1. Allocate device memory"))
    story.append(code("cudaMalloc(&d_data, N * sizeof(float));"))
    story.append(code("// 2. Copy Host → Device"))
    story.append(code("cudaMemcpy(d_data, h_data, N*sizeof(float), cudaMemcpyHostToDevice);"))
    story.append(code("// 3. Launch kernel (asynchronous!)"))
    story.append(code("dim3 grid(N/256+1), block(256);"))
    story.append(code("my_kernel<<<grid, block>>>(d_data, N);"))
    story.append(code("// 4. Synchronize (wait for GPU)"))
    story.append(code("cudaDeviceSynchronize();"))
    story.append(code("// 5. Copy Device → Host"))
    story.append(code("cudaMemcpy(h_data, d_data, N*sizeof(float), cudaMemcpyDeviceToHost);"))
    story.append(code("// 6. Free device memory"))
    story.append(code("cudaFree(d_data);"))
    story.append(warn("Kernel launches are ASYNCHRONOUS. cudaMemcpy() is SYNCHRONOUS. ALWAYS cudaDeviceSynchronize() before accessing results."))

# ═══════════════════════════════════════════════════════════════════════════════
# L5 — ADVANCED CUDA
# ═══════════════════════════════════════════════════════════════════════════════
def sec_l5(story):
    story += section("LECTURE 5 — Advanced CUDA", BLUE)

    story.append(h2("5.1  GPU Architecture Numbers"))
    story.append(tbl(["Spec","V100 (Volta)","A100 (Ampere)","H100 (Hopper)"],
        [["Year","2017","2020","2022"],
         ["SMs","80","108","132"],
         ["CUDA Cores/SM","64+64","128","128"],
         ["Total CUDA Cores","5,120","6,912 (actually 108×128)","18,432"],
         ["Tensor Core Gen","1st/2nd","3rd (TF32,BF16,FP64,2:4)","4th (+FP8, TMA)"],
         ["Peak FP32","15.7 TFLOPS","19.5 TFLOPS","60 TFLOPS"],
         ["Peak BF16 TC","—","312 TFLOPS","1,000 TFLOPS"],
         ["HBM BW","900 GB/s (HBM2)","2 TB/s (HBM2e)","3.35 TB/s (HBM3)"],
         ["L2 Cache","6 MB","40 MB","50 MB"],
         ["Shared Mem/SM","96 KB","192 KB","256 KB"],
         ["NVLink BW","300 GB/s","600 GB/s","900 GB/s"]],
        widths=[2.0*inch,1.7*inch,1.9*inch,1.9*inch]))
    story.append(sp())

    story.append(h2("5.2  Tensor Cores — Theory & Speedup Numbers"))
    story.append(body("Tensor Cores perform <b>D = A×B + C</b> (matrix multiply-accumulate) in one operation. "
        "A, B in low precision (FP16/BF16/INT8/FP8), C/D in higher precision (FP32). "
        "They operate on 16×16 matrices per cycle — far more throughput than general CUDA cores."))
    story.append(tbl(["GPU","Tensor Core Gen","FP16 Tensor TFLOPS","Speedup vs. No Tensor Cores"],
        [["V100","1st gen","125 TFLOPS","4×"],
         ["A100","3rd gen","312 TFLOPS (FP16), 624 (INT8)","8×"],
         ["H100","4th gen","1,000 TFLOPS (FP16), 2,000 (FP8)","16×"]],
        widths=[1.0*inch,1.5*inch,2.5*inch,2.5*inch]))
    story.append(key("H100 with FP8 delivers 2,000 TFLOPS — 128× more throughput than FP32 CUDA cores alone. This is why FP8 training matters."))
    story.append(sp())

    story.append(h2("5.3  Memory Coalescing — Theory & Worked Examples"))
    story.append(body("When threads in a warp access memory, the GPU hardware tries to coalesce "
        "(combine) multiple accesses into a single memory transaction. Proper coalescing "
        "is one of the most important optimizations for global memory performance."))
    story.append(sp(0.5))
    story.append(h3("Rule: Consecutive threads should access consecutive addresses"))
    story.append(tbl(["Pattern","Access Pattern","Transactions","Throughput","Example"],
        [["Stride-1 (optimal)","Thread i → address[i*1]","1 transaction for 32 threads","100% (32 elements/tx)","float A[32] where thread i reads A[i]"],
         ["Stride-2","Thread i → address[i*2]","2 transactions","50%","Reading every other element"],
         ["Stride-4","Thread i → address[i*4]","4 transactions","25%","Reading every 4th element"],
         ["Stride-32 (worst)","Thread i → address[i*32]","32 transactions","3.1% (1 element/tx)","Classic matrix column access"]],
        widths=[1.3*inch,1.8*inch,1.5*inch,1.0*inch,2.0*inch]))
    story.append(sp(0.5))
    story.append(h3("Worked Example — Matrix Access Patterns"))
    story.append(worked("Matrix A[1024][1024], stored row-major. Warp of 32 threads."))
    story.append(worked(""))
    story.append(worked("Case 1: Row access (GOOD): Thread i reads A[row][i]"))
    story.append(worked("  Addresses: A[row][0], A[row][1], ..., A[row][31]"))
    story.append(worked("  Memory offsets: 0, 4, 8, ..., 124 bytes  → stride-1 floats"))
    story.append(worked("  → 1 cache line (128 bytes) covers all 32 threads → 1 transaction"))
    story.append(worked(""))
    story.append(worked("Case 2: Column access (BAD): Thread i reads A[i][col]"))
    story.append(worked("  Addresses: A[0][col], A[1][col], ..., A[31][col]"))
    story.append(worked("  Memory offsets: 0, 4096, 8192, ..., 126976 bytes  → stride-1024 floats"))
    story.append(worked("  → Each thread accesses a different cache line → 32 transactions"))
    story.append(worked("  → 32× more memory bandwidth consumed"))
    story.append(exam("Pattern A (stride-1) = 1 transaction. Pattern B (stride-32) = 32 transactions. 32× difference in effective bandwidth."))
    story.append(sp())

    story.append(h2("5.4  Shared Memory Bank Conflicts — Theory"))
    story.append(body("Shared memory is divided into 32 banks (4-byte granularity). "
        "Consecutive 4-byte words go to consecutive banks (bank = address/4 mod 32)."))
    story.append(b("<b>No conflict:</b> All 32 threads access different banks → single cycle (ideal)"))
    story.append(b("<b>Bank conflict:</b> 2+ threads access different addresses in the SAME bank → serialized"))
    story.append(b("<b>Broadcast:</b> Multiple threads access the SAME address in the SAME bank → no conflict (broadcast)"))
    story.append(sp(0.5))
    story.append(worked("Example: __shared__ float tile[32][32]"))
    story.append(worked("Thread i accesses tile[0][i]: i=0→bank 0, i=1→bank 1, ..., i=31→bank 31 → NO CONFLICT"))
    story.append(worked("Thread i accesses tile[i][0]: i=0→addr 0→bank 0, i=1→addr 128→bank 0 → 32-WAY CONFLICT!"))
    story.append(worked("FIX: __shared__ float tile[32][33] → padding shifts column 0 to different banks"))
    story.append(sp())

    story.append(h2("5.5  SM Occupancy — Theory & Worked Numerical"))
    story.append(body("Occupancy = (active warps/SM) / (max warps/SM). "
        "High occupancy helps hide memory latency by having other warps ready to execute while one waits."))
    story.append(formula("Active Warps per SM = (threads_per_block / 32) × blocks_per_SM"))
    story.append(sp(0.5))
    story.append(h3("Worked Example — Fermi Architecture (max 1536 threads/SM, max 8 blocks/SM, max 48 warps/SM)"))
    story.append(tbl(["Block Config","Threads/Block","Warps/Block","Max Blocks Fit","Active Warps","Occupancy","Verdict"],
        [["8×8","64","2","8","2×8=16","16/48=33%","POOR — wasted SM resources"],
         ["16×16","256","8","6 (limited by 6×256=1536≤1536)","8×6=48","48/48=100%","OPTIMAL"],
         ["32×32","1024","32","1 (1×1024=1024≤1536, but max 8 blocks)","32×1=32","32/48=67%","Good but not optimal"],
         ["12×12","144","4.5→5","8","5×8=40→but only 5 complete warps","35/48=73%","OK"]],
        widths=[1.0*inch,1.0*inch,1.0*inch,1.4*inch,1.2*inch,1.1*inch,1.8*inch]))
    story.append(key("16×16 block = 256 threads = 8 warps → 6 blocks fit on Fermi SM = 100% occupancy. This is the design sweet spot."))
    story.append(sp())

    story.append(h2("5.6  CUDA Streams — Overlapping Compute and Transfer"))
    story.append(body("A CUDA Stream is an ordered queue of GPU work. Operations in the same stream execute "
        "sequentially. Operations in <b>different streams can execute concurrently</b> — this is the key "
        "to overlapping H2D transfers with kernel execution:"))
    story.append(code("cudaStream_t s1, s2;"))
    story.append(code("cudaStreamCreate(&s1); cudaStreamCreate(&s2);"))
    story.append(code("// Stream 1: copy chunk A then process it"))
    story.append(code("cudaMemcpyAsync(d_a, h_a, size, H2D, s1);"))
    story.append(code("kernel_A<<<grid, block, 0, s1>>>(d_a);"))
    story.append(code("// Stream 2: concurrently copy chunk B (while Stream 1's kernel runs!)"))
    story.append(code("cudaMemcpyAsync(d_b, h_b, size, H2D, s2);"))
    story.append(code("kernel_B<<<grid, block, 0, s2>>>(d_b);"))
    story.append(b("Pinned memory (cudaMallocHost) is REQUIRED for cudaMemcpyAsync to work correctly"))
    story.append(b("Unpinned memory forces blocking transfer — no overlap possible"))

# ═══════════════════════════════════════════════════════════════════════════════
# L6 — DISTRIBUTED DL
# ═══════════════════════════════════════════════════════════════════════════════
def sec_l6(story):
    story += section("LECTURE 6 — Distributed Deep Learning", GREEN)

    story.append(h2("6.1  Why Distributed Training Is Necessary — The Numbers"))
    story.append(tbl(["Model","Parameters","Memory Needed (FP16)","Adam Optimizer (FP32)","Single H100?"],
        [["LLaMA-3 7B","7B","14 GB","~112 GB","Barely fits with ZeRO"],
         ["LLaMA-3 70B","70B","140 GB","~1.1 TB","IMPOSSIBLE without ZeRO"],
         ["LLaMA-3 405B","405B","810 GB","~6.5 TB","Need 80+ H100s with ZeRO-3"],
         ["GPT-3 175B","175B","350 GB","~2.8 TB","Need 35+ H100s with ZeRO-3"],
         ["GPT-4 ~1.8T","~1.8T","~3.6 TB","~29 TB","Need 1000+ H100s"]],
        widths=[1.5*inch,1.0*inch,1.5*inch,1.5*inch,2.0*inch]))
    story.append(worked("Adam optimizer memory breakdown per parameter (16 bytes total):"))
    story.append(worked("  FP16 model weights: 2 bytes"))
    story.append(worked("  FP16 gradients:     2 bytes"))
    story.append(worked("  FP32 master weights: 4 bytes  (kept for numerical stability)"))
    story.append(worked("  FP32 momentum (V):  4 bytes"))
    story.append(worked("  FP32 variance (S):  4 bytes"))
    story.append(worked("  TOTAL:             16 bytes per parameter"))
    story.append(worked("  For 175B params:   175B × 16 = 2,800 GB = 2.8 TB"))
    story.append(sp())

    story.append(h2("6.2  Collective Communication Primitives — Theory"))
    story.append(tbl(["Primitive","Direction","What Happens","DL Application","Bandwidth"],
        [["Broadcast","1→all","Root sends SAME data to all ranks","Distribute model params at init","N × (P-1)"],
         ["Reduce","all→1","All ranks send; root combines (sum/max)","Gradient accumulation at PS","N × (P-1)"],
         ["AllReduce","all→all","Reduce then distribute; ALL get result","Gradient sync in DDP — most critical","2N(P-1)/P"],
         ["Reduce-Scatter","all→all (partial)","Each rank gets a different shard of result","ZeRO gradient sharding","N(P-1)/P"],
         ["AllGather","all→all (full)","Each has partial; all receive full data","ZeRO parameter reconstruction","N(P-1)/P"],
         ["AllToAll","all→all (different)","Each rank sends different data to each rank","MoE expert routing","N×P"]],
        widths=[1.0*inch,1.0*inch,2.0*inch,2.0*inch,1.5*inch]))
    story.append(exam("AllReduce = most important collective. Used in DDP for gradient averaging. Answer: Ring AllReduce."))
    story.append(sp())

    story.append(h2("6.3  Ring AllReduce — Full Algorithm with Numerics"))
    story.append(body("Ring AllReduce is bandwidth-optimal. All P GPUs arranged in a logical ring. "
        "Each GPU sends only to its right neighbor. Two phases — P-1 steps each:"))
    story.append(sp(0.5))
    story.append(h3("Phase 1: Scatter-Reduce (P-1 steps)"))
    story.append(body("Each GPU sends a chunk to its right neighbor and accumulates from its left neighbor. "
        "After P-1 steps, each GPU has ONE fully-reduced chunk (the sum across all GPUs for that chunk)."))
    story.append(sp(0.5))
    story.append(h3("Phase 2: AllGather (P-1 steps)"))
    story.append(body("GPUs circulate the fully-reduced chunks around the ring. After P-1 steps, "
        "every GPU has ALL fully-reduced chunks → complete result everywhere."))
    story.append(sp(0.5))
    story.append(h3("Worked Bandwidth Example"))
    story.append(worked("Setup: P=4 GPUs, N=1B parameters (4 GB total in FP32)"))
    story.append(worked("Each GPU's gradient array split into P=4 chunks of N/P = 1 GB each"))
    story.append(worked(""))
    story.append(worked("Phase 1 (Scatter-Reduce): 3 steps"))
    story.append(worked("  Step 1: Each GPU sends chunk[0] → neighbor (1 GB sent per GPU)"))
    story.append(worked("  Step 2: Each GPU sends accumulated chunk → neighbor (1 GB sent per GPU)"))
    story.append(worked("  Step 3: Each GPU sends accumulated chunk → neighbor (1 GB sent per GPU)"))
    story.append(worked("  Total sent per GPU: 3 × 1 GB = 3 GB = N(P-1)/P = 4GB × 3/4 = 3 GB ✓"))
    story.append(worked(""))
    story.append(worked("Phase 2 (AllGather): 3 more steps"))
    story.append(worked("  Same amount: 3 GB sent per GPU"))
    story.append(worked(""))
    story.append(worked("Total per GPU = 2 × N(P-1)/P = 2 × 3 GB = 6 GB"))
    story.append(worked("Formula: 2N(P-1)/P = 2 × 4GB × (4-1)/4 = 2 × 3 GB = 6 GB ✓"))
    story.append(worked(""))
    story.append(worked("Compare with Parameter Server (P=4):"))
    story.append(worked("  PS receives from 3 workers: 3 × 4 GB = 12 GB RX"))
    story.append(worked("  PS sends to 3 workers:      3 × 4 GB = 12 GB TX"))
    story.append(worked("  Total PS node:              24 GB = 2N(P-1) = 2×4×3 = 24 GB"))
    story.append(worked(""))
    story.append(worked("Ring AllReduce: 6 GB per GPU    (independent of P for large P → scales!)"))
    story.append(worked("Parameter Server: 24 GB at PS   (4× more, grows with P → bottleneck!)"))
    story.append(key("Ring AllReduce communication per GPU = 2N(P-1)/P ≈ 2N for large P — essentially CONSTANT regardless of cluster size."))
    story.append(sp())

    story.append(h2("6.4  ZeRO Optimizer — Memory Calculation with Exact Numbers"))
    story.append(body("ZeRO (Zero Redundancy Optimizer) shards model states across GPUs "
        "while maintaining data-parallel semantics (forward/backward pass unchanged)."))
    story.append(sp(0.5))
    story.append(h3("Setup for All Examples"))
    story.append(worked("Model: ψ = 7.5B parameters, Adam optimizer (K=12 bytes for opt states per param)"))
    story.append(worked("N = 64 GPUs, each with 80 GB HBM"))
    story.append(worked("Baseline (naive DP): (2+2+12) × 7.5B = 16 × 7.5B = 120 GB per GPU"))
    story.append(worked("  → CANNOT FIT on 80 GB GPU! Need ZeRO."))
    story.append(sp(0.5))
    story.append(h3("ZeRO Stage 1: Shard Optimizer States Only"))
    story.append(worked("Each GPU stores: full weights (2B) + full gradients (2B) + 1/N optimizer states (12/N B)"))
    story.append(worked("Memory formula: (2 + 2 + 12/64) × 7.5B = (2 + 2 + 0.1875) × 7.5B"))
    story.append(worked("             = 4.1875 × 7.5B = 31.4 GB per GPU ✓ (fits in 80 GB)"))
    story.append(worked("Memory reduction: 120 / 31.4 = ~3.8× ≈ 4× reduction"))
    story.append(worked("Max trainable model: 80 GB / 4.1875 = ~19B parameters"))
    story.append(sp(0.5))
    story.append(h3("ZeRO Stage 2: Shard Optimizer States + Gradients"))
    story.append(worked("Each GPU stores: full weights (2B) + 1/N gradients (2/N B) + 1/N optimizer states (12/N B)"))
    story.append(worked("Memory formula: (2 + 2/64 + 12/64) × 7.5B = (2 + 0.03125 + 0.1875) × 7.5B"))
    story.append(worked("             = 2.21875 × 7.5B = 16.6 GB per GPU ✓"))
    story.append(worked("Memory reduction: 120 / 16.6 = ~7.2× ≈ 8× reduction"))
    story.append(worked("Max trainable model: 80 GB / 2.22 = ~36B parameters"))
    story.append(sp(0.5))
    story.append(h3("ZeRO Stage 3 (FSDP): Shard Everything"))
    story.append(worked("Each GPU stores: 1/N weights + 1/N gradients + 1/N optimizer states"))
    story.append(worked("Memory formula: (2/64 + 2/64 + 12/64) × 7.5B = (16/64) × 7.5B = 0.25 × 7.5B"))
    story.append(worked("             = 1.875 GB per GPU ← incredible reduction!"))
    story.append(worked("Memory reduction: 120 / 1.875 = 64× = N× reduction (perfect sharding)"))
    story.append(worked("Max trainable model: 80 GB / 0.25 = 320B parameters"))
    story.append(sp(0.5))
    story.append(tbl(["ZeRO Stage","Sharded","Mem/GPU (N=64)","Reduction","Max Model Size"],
        [["Baseline (DDP)","Nothing","120 GB","1×","5B"],
         ["ZeRO-1","Optimizer states","31.4 GB","~4×","19B"],
         ["ZeRO-2","Optim + Gradients","16.6 GB","~8×","36B"],
         ["ZeRO-3 (FSDP)","Optim + Grad + Params","1.9 GB","64×","320B"]],
        widths=[1.3*inch,1.8*inch,1.5*inch,0.9*inch,1.3*inch],
        hcol=GREEN))
    story.append(sp())

    story.append(h2("6.5  Pipeline Parallelism — Bubble Fraction Formula"))
    story.append(body("Pipeline parallelism splits model layers across GPUs. Microbatches fill the pipeline "
        "to keep GPUs busy. The key challenge is the 'bubble' — idle GPU time at pipeline start and end."))
    story.append(formula("Bubble fraction = (P−1) / M   where P = pipeline stages, M = microbatches"))
    story.append(formula("Ideal speedup = P  (never reached due to bubble)"))
    story.append(sp(0.5))
    story.append(worked("Example: P=4 stages, M=8 microbatches"))
    story.append(worked("  Bubble fraction = (4-1)/8 = 3/8 = 37.5% idle time"))
    story.append(worked("  With M=16 microbatches: bubble = 3/16 = 18.75%"))
    story.append(worked("  With M=32 microbatches: bubble = 3/32 = 9.4%"))
    story.append(worked("  As M→∞: bubble→0  (but memory explodes for GPipe since O(M) activations)"))
    story.append(worked("  1F1B variant: O(P) memory instead of O(M) — much better for long pipelines"))

# ═══════════════════════════════════════════════════════════════════════════════
# L9 — FLASHATTENTION
# ═══════════════════════════════════════════════════════════════════════════════
def sec_l9(story):
    story += section("LECTURE 9 — Efficient Transformers & FlashAttention", TEAL)

    story.append(h2("9.1  Self-Attention — Complete Theory"))
    story.append(formula("Attention(Q, K, V) = softmax(QK^T / √d_k) · V"))
    story.append(body("<b>Step-by-step intuition:</b>"))
    story.append(b("<b>Q (Query):</b> 'What am I looking for?' — the current token asking a question"))
    story.append(b("<b>K (Key):</b> 'What information do I represent?' — every token advertising itself"))
    story.append(b("<b>V (Value):</b> 'What content do I provide if selected?' — the actual information to extract"))
    story.append(b("<b>Score_{ij} = Q_i · K_j^T / √d_k:</b> How relevant is token j to token i? Divide by √d_k to prevent softmax saturation."))
    story.append(b("<b>softmax:</b> Normalize scores to probability distribution (sum=1) over all tokens"))
    story.append(b("<b>× V:</b> Weighted sum of all values, weighted by how relevant each token is"))
    story.append(sp(0.5))
    story.append(worked("Concrete example: d_k=64, so √d_k=8"))
    story.append(worked("  Q_1 · K_1 = 112 → divide by 8 → score=14"))
    story.append(worked("  Q_1 · K_2 = 96  → divide by 8 → score=12"))
    story.append(worked("  softmax([14, 12]) = [e^14, e^12] / (e^14 + e^12) = [0.88, 0.12]"))
    story.append(worked("  Output_1 = 0.88 × V_1 + 0.12 × V_2  (mostly V_1)"))
    story.append(sp(0.5))
    story.append(body("<b>Standard Complexity:</b>"))
    story.append(b("Time complexity: O(N²·d) — the N×N attention score matrix is computed for every forward pass"))
    story.append(b("Memory complexity: O(N²) — the N×N attention matrix stored in HBM (THIS is the bottleneck)"))
    story.append(exam("For N=32K tokens: N² = 1 billion elements = 4 GB for FP32. For N=128K: 64 GB — larger than GPU HBM!"))
    story.append(sp())

    story.append(h2("9.2  Why Standard Attention is Memory-Bound"))
    story.append(body("Standard attention has 5 separate operations, each reading and writing from slow HBM:"))
    story.append(tbl(["Operation","HBM Read","HBM Write","Can be fused?"],
        [["QK^T matmul","Q (N×d), K (N×d)","S = QK^T (N×N)","No (output too large)"],
         ["Masking","S (N×N)","S' = Mask(S) (N×N)","Yes, into matmul"],
         ["Softmax","S' (N×N)","P = softmax(S') (N×N)","Yes, but needs all scores first"],
         ["Dropout","P (N×N)","P'' (N×N)","Yes"],
         ["PV matmul","P'' (N×N), V (N×d)","O (N×d)","Yes"]],
        widths=[1.5*inch,1.8*inch,1.8*inch,1.0*inch], hcol=TEAL))
    story.append(worked("GPT-2 medium benchmark (seq_len=1024, head_dim=64, batch=64, A100):"))
    story.append(worked("  Standard: HBM R/W = 40.3 GB, Runtime = 41.7 ms"))
    story.append(worked("  FlashAttention: HBM R/W = 4.4 GB, Runtime = 7.3 ms"))
    story.append(worked("  Reduction: 9.2× less HBM traffic, 5.7× faster despite 13% MORE FLOPs"))
    story.append(worked("  Conclusion: Attention IS memory-bound (HBM traffic dominates, not FLOPs)"))
    story.append(sp())

    story.append(h2("9.3  FlashAttention — The Two Key Innovations"))
    story.append(h3("Innovation 1: Kernel Fusion"))
    story.append(body("Instead of 5 separate HBM read-compute-write operations, fuse everything into ONE kernel:"))
    story.append(b("Load Q, K, V tiles from HBM to SRAM ONCE"))
    story.append(b("Execute QK^T, masking, softmax, dropout, PV — all inside SRAM"))
    story.append(b("Write output O to HBM ONCE"))
    story.append(b("Result: HBM traffic ∝ O(N·d) instead of O(N²)"))
    story.append(sp(0.5))
    story.append(h3("Innovation 2: Online Softmax (Tiled Softmax)"))
    story.append(body("Standard softmax requires ALL N scores simultaneously (to compute denominator). "
        "Online softmax computes the exact same result incrementally, one block at a time, "
        "using just two running statistics per row:"))
    story.append(sp(0.5))
    story.append(b("<b>m(x) = max_i(x_i)</b> — running maximum (for numerical stability)"))
    story.append(b("<b>ℓ(x) = Σ_i exp(x_i − m(x))</b> — running sum of shifted exponentials"))
    story.append(b("<b>softmax(x) = exp(x_i − m(x)) / ℓ(x)</b>"))
    story.append(sp(0.5))
    story.append(body("<b>Combining two blocks:</b> When we see a new block with max m̃ and sum ℓ̃:"))
    story.append(formula("m_new = max(m_old, m̃)"))
    story.append(formula("ℓ_new = e^(m_old − m_new)·ℓ_old + e^(m̃ − m_new)·ℓ̃"))
    story.append(formula("O_new = diag(ℓ_new)^{-1} [diag(ℓ_old)·e^(m_old−m_new)·O_old + e^(m̃−m_new)·P̃·V]"))
    story.append(body("The e^(m_old−m_new) and e^(m̃−m_new) terms <b>rescale</b> previous partial sums "
        "to account for the updated global maximum. This is mathematically exact — no approximation."))
    story.append(sp())

    story.append(h2("9.4  FlashAttention IO Analysis — Worked Tiling Example"))
    story.append(body("This example demonstrates WHY tiling reduces HBM I/O even with the same FLOPs:"))
    story.append(sp(0.5))
    story.append(h3("Setup"))
    story.append(worked("Q has 9 rows total. |Q| = size of Q matrix = 18 units. |K| = 18, |V| = 9, |O| = 9"))
    story.append(worked(""))
    story.append(h3("Standard Attention (no tiling): row-by-row processing"))
    story.append(worked("For each row of Q (9 rows total), must read ALL of K and V:"))
    story.append(worked("  IO = |Q| + r(Q) × (|K| + |V|) + |O|"))
    story.append(worked("     = 18 + 9 × (18 + 9) + 9"))
    story.append(worked("     = 18 + 9 × 27 + 9"))
    story.append(worked("     = 18 + 243 + 9 = 270 IO units"))
    story.append(sp(0.5))
    story.append(h3("FlashAttention with block size b=3 (reads 3 Q rows at once)"))
    story.append(worked("Number of Q blocks = r(Q)/b = 9/3 = 3 blocks (instead of 9 row reads)"))
    story.append(worked("  IO = |Q| + b(Q) × (|K| + |V|) + |O|"))
    story.append(worked("     = 18 + 3 × (18 + 9) + 9"))
    story.append(worked("     = 18 + 3 × 27 + 9"))
    story.append(worked("     = 18 + 81 + 9 = 108 IO units"))
    story.append(worked(""))
    story.append(worked("IO Reduction = 270 / 108 = 2.5× with block size 3"))
    story.append(worked("Note: Same number of FLOPs! IO is reduced because K and V are"))
    story.append(worked("      loaded once per Q-block instead of once per Q-row."))
    story.append(worked(""))
    story.append(worked("A100 uses block size 8×8 (not 3). With b=9 (all rows at once):"))
    story.append(worked("  IO = 18 + 1 × (18+9) + 9 = 18+27+9 = 54 IO (5× reduction)"))
    story.append(worked("  (Not achievable since K+V+Q+O must fit in 20 MB SRAM)"))
    story.append(key("Larger blocks = more reuse of K and V = less HBM traffic. Block size limited by SRAM capacity."))
    story.append(sp())

    story.append(h2("9.5  Block Size Calculation — How Large Can Tiles Be?"))
    story.append(body("The tiling block sizes are determined by SRAM capacity. At any time, "
        "one Q-tile, one K-tile, one V-tile, and one O-tile must fit in SRAM:"))
    story.append(formula("Bc = ⌈M / 4d⌉       [column block size = K/V tile size]"))
    story.append(formula("Br = min(⌈M / 4d⌉, d)  [row block size = Q/O tile size]"))
    story.append(body("Why 4d? We need 4 tiles of size B×d in SRAM: Q_tile, K_tile, V_tile, O_tile."))
    story.append(sp(0.5))
    story.append(worked("Concrete calculation for A100 GPU:"))
    story.append(worked("  M = 20 MB SRAM = 20 × 10^6 bytes"))
    story.append(worked("  d = 64 (head dimension in GPT-2 / standard transformer)"))
    story.append(worked("  Each element = 2 bytes (FP16)"))
    story.append(worked("  Bc = ⌈20×10^6 / (4 × 64 × 2)⌉ = ⌈20×10^6 / 512⌉ = ⌈39,062.5⌉ = 39,063"))
    story.append(worked("  Br = min(39,063, 64) = 64"))
    story.append(worked(""))
    story.append(worked("Interpretation:"))
    story.append(worked("  Q_tile: 64 × 64 = 4,096 elements × 2B = 8 KB"))
    story.append(worked("  K_tile: 39,063 × 64 × 2B ≈ 5 MB  [ideally]"))
    story.append(worked("  In practice, A100 uses 8×8 blocks due to alignment and warp constraints"))
    story.append(sp())

    story.append(h2("9.6  FlashAttention Performance Results"))
    story.append(tbl(["Metric","Standard Attention","FlashAttention","Improvement"],
        [["HBM R/W (GPT-2 med.)","40.3 GB","4.4 GB","9.2× less HBM traffic"],
         ["Runtime (GPT-2 med.)","41.7 ms","7.3 ms","5.7× faster"],
         ["GFLOPs (GPT-2 med.)","66.6","75.2","13% MORE (recomputation)"],
         ["Memory complexity","O(N²)","O(N)","10–20× less memory"],
         ["Wall-clock speedup","—","2–4× (up to 6×)","Higher with masking+dropout"],
         ["BERT training","20.0 min (MLPerf)","17.4 min","~15% faster"],
         ["OOM threshold","N≈2K on A100","N=128K+ easily","Enables long-context models"]],
        widths=[2.0*inch,1.7*inch,1.7*inch,2.1*inch], hcol=TEAL))
    story.append(sp(0.5))
    story.append(exam("FlashAttention does NOT reduce FLOPs — it reduces HBM I/O. Same O(N²) math, different memory access pattern."))
    story.append(exam("Memory is LINEAR in N (O(N)) vs standard O(N²). This is what enables 128K+ context lengths."))
    story.append(sp())

    story.append(h2("9.7  Recomputation in the Backward Pass"))
    story.append(body("Standard backprop through attention requires the N×N attention matrix P from the forward pass. "
        "FlashAttention does NOT store P (it's N×N = too large). Instead:"))
    story.append(b("<b>During forward pass:</b> Save only the softmax statistics (m and ℓ, each of size N) — O(N) memory"))
    story.append(b("<b>During backward pass:</b> Recompute attention from Q, K, V + saved (m, ℓ) — same tiling algorithm"))
    story.append(b("<b>Cost:</b> ~13% more FLOPs (recomputing attention twice) — WELL worth the 10-20× memory savings"))
    story.append(key("Store O(N) softmax stats; recompute O(N²) attention matrix on demand. Time-memory tradeoff in favor of memory."))

# ═══════════════════════════════════════════════════════════════════════════════
# FORMULAS REFERENCE
# ═══════════════════════════════════════════════════════════════════════════════
def sec_ref(story):
    story += section("MASTER FORMULA REFERENCE", RED)

    story.append(h2("All Critical Formulas — One Page"))
    story.append(tbl(["Formula","Name","Variables"],
        [["S = 1 / [(1−p) + p/s]","Amdahl's Law","S=speedup, p=parallel fraction, s=improvement speedup"],
         ["E = T₁ / (P × T_P)","Strong Scaling Efficiency","T₁=single GPU time, P=#GPUs, T_P=parallel time"],
         ["Attainable FLOPS = min(Peak, AI × BW)","Roofline Model","AI=arithmetic intensity [FLOP/byte], BW=memory BW"],
         ["AI = #FLOP / DRAM_bytes","Arithmetic Intensity","DRAM bytes = bytes actually fetched from off-chip DRAM"],
         ["Crossover AI = Peak FLOPS / Memory BW","Roofline Crossover","Above this: compute-bound. Below: memory-bound"],
         ["V_t = β₁V_{t-1} + (1−β₁)g_t","Adam 1st Moment","V = momentum, β₁≈0.9, g = gradient"],
         ["S_t = β₂S_{t-1} + (1−β₂)g_t²","Adam 2nd Moment","S = variance, β₂≈0.999"],
         ["V̂_t = V_t/(1−β₁ᵗ), Ŝ_t = S_t/(1−β₂ᵗ)","Adam Bias Correction","Critical at early steps (t is small)"],
         ["w_{t+1} = w_t − α·V̂_t/(√Ŝ_t+ε)","Adam Update","α=learning rate, ε=1e-8 (numerical stability)"],
         ["i = blockIdx.x*blockDim.x + threadIdx.x","CUDA 1D Global Index","The most important CUDA formula"],
         ["Row=blockIdx.y*blockDim.y+threadIdx.y, Col=blockIdx.x*blockDim.x+threadIdx.x","CUDA 2D Index","For 2D grids"],
         ["Occupancy = Active Warps/SM / Max Warps/SM","SM Occupancy","Target: 100%; Warp = 32 threads"],
         ["Attention(Q,K,V) = softmax(QK^T/√d_k)V","Self-Attention","d_k=head dimension; divide to prevent saturation"],
         ["Bc=⌈M/4d⌉, Br=min(⌈M/4d⌉,d)","FlashAttention Block Sizes","M=SRAM size, d=head dimension"],
         ["IO = |Q| + b(Q)(|K|+|V|) + |O|","FlashAttention IO Cost","b(Q)=#Q-blocks, smaller→more HBM traffic"],
         ["Ring AllReduce: 2N(P-1)/P per GPU","AllReduce BW","N=params, P=#GPUs; ≈2N for large P"],
         ["PS comm at server: 2N(P-1)","Parameter Server BW","Grows linearly with P — bottleneck!"],
         ["ZeRO-3 mem/GPU: (2+2+12)ψ/N","FSDP Memory","ψ=params, N=#GPUs; 64× reduction with N=64"],
         ["Bubble = (P-1)/M","Pipeline Bubble Fraction","P=pipeline stages, M=microbatches"],
         ["S = T_A/T_B","Speedup","T_A=baseline time, T_B=optimized time"],
         ["E_scaling = t_serial/(t_parallel×p)","Scaling Efficiency","p=#processors; ideal=1.0"],
         ["x_G = (∏xᵢ)^(1/n)","Geometric Mean","Use for speedup ratios and benchmark comparisons"]],
        widths=[3.2*inch,1.5*inch,2.8*inch]))
    story.append(sp())

    story.append(h2("Critical Numbers to Memorize"))
    story.append(tbl(["Item","Value","Context"],
        [["Warp size","32 threads","Always — scheduling unit in all CUDA GPUs"],
         ["Max threads/block","1024","Cannot exceed this in kernel launch config"],
         ["Shared mem banks","32, 4-byte granule","Bank conflicts kill shared memory performance"],
         ["FP32→FP16 reduction","2×","Half the memory, same value range for BF16"],
         ["FP32→INT8 reduction","4×","4× memory savings for inference"],
         ["FP8 E4M3 max value","448","For forward pass (weights/activations)"],
         ["FP8 E5M2 max value","57,344","For backward pass (gradients — needs more range)"],
         ["Adam β₁ default","0.9 (LLaMA/GPT)","1st moment smoothing"],
         ["Adam β₂ default","0.95–0.999","2nd moment smoothing"],
         ["DAXPY arithmetic intensity","0.083 FLOP/byte","Memory-bound: 2 FLOPs, 24 bytes"],
         ["7-point stencil AI","0.109 FLOP/byte","Still memory-bound"],
         ["FlashAttention speedup","2–4× (up to 6×)","GPT-2 medium: 5.7×"],
         ["FlashAttention HBM reduction","9.2× (GPT-2 med.)","40.3 GB → 4.4 GB"],
         ["Standard attention memory","O(N²)","N=sequence length"],
         ["FlashAttention memory","O(N)","Linear! Enables 128K+ contexts"],
         ["ZeRO-3 reduction (N=64)","64×","120 GB → 1.9 GB per GPU for 7.5B model"],
         ["Ring AllReduce cost/GPU","2N(P-1)/P ≈ 2N","Independent of P for large P"],
         ["A100 HBM bandwidth","2 TB/s (HBM2e)","Critical for memory-bound kernels"],
         ["H100 HBM bandwidth","3.35 TB/s (HBM3)","1.7× more than A100"],
         ["SRAM bandwidth","~19 TB/s","~13× faster than HBM"],
         ["Adam memory per param","16 bytes","2(wt)+2(grad)+4(master)+4(mom)+4(var)"]],
        widths=[2.5*inch,1.8*inch,3.2*inch]))

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE TEMPLATE
# ═══════════════════════════════════════════════════════════════════════════════
def on_page(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica",7); canvas.setFillColor(DGRAY)
    canvas.drawString(0.75*inch,0.35*inch,"HPML Deep-Dive Notes — Columbia University Spring 2026")
    canvas.drawRightString(7.75*inch,0.35*inch,f"Page {doc.page}")
    canvas.restoreState()

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    doc = SimpleDocTemplate(OUT, pagesize=letter,
        rightMargin=0.75*inch, leftMargin=0.75*inch,
        topMargin=0.75*inch, bottomMargin=0.6*inch,
        title="HPML Deep-Dive Study Notes",
        author="Rajvardhan Patil — Columbia University")
    story = []
    cover(story)
    sec_l1(story)
    sec_l2(story)
    sec_l3(story)
    sec_l4(story)
    sec_l5(story)
    sec_l6(story)
    sec_l9(story)
    sec_ref(story)
    doc.build(story, onFirstPage=on_page, onLaterPages=on_page)
    print(f"Done: {OUT}")

if __name__ == "__main__":
    main()
