from __future__ import annotations

from pathlib import Path
import math
import textwrap

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle, Polygon, Ellipse, FancyArrowPatch
from matplotlib.lines import Line2D
from matplotlib import patheffects as pe


ROOT = Path(__file__).resolve().parents[2]
REPO = ROOT
FIG_DIR = ROOT / "figures" / "paper_main"
OUT = ROOT / "outputs"
FIG_DIR.mkdir(parents=True, exist_ok=True)
OUT.mkdir(exist_ok=True)

PNG = FIG_DIR / "fig40_method_workflow.png"
PNG_2X = OUT / "fig40_method_workflow_4k.png"


BLUE = "#073B7A"
BLUE2 = "#0B4EA2"
BLUE3 = "#2E71C0"
PALE_BLUE = "#EEF6FF"
PANEL_BG = "#FFFFFF"
LINE = "#0A3A78"
ORANGE = "#D85F18"
ORANGE_PALE = "#FFF2E8"
GREEN = "#2F7044"
GREEN_PALE = "#EEF8F0"
PURPLE = "#7E5AAE"
TEXT = "#111827"
MUTED = "#526173"
GRID = "#D8E2EF"
RED = "#E9443C"


def get_results() -> dict[str, str]:
    summary = pd.read_csv(REPO / "results" / "strict_100Ah" / "validation_model_selector_summary_conservative_margin005.csv").iloc[0]
    source = pd.read_csv(REPO / "results" / "source5_validation_selector" / "source5_uncertainty_table.csv")
    src_acc = source[source["metric"] == "accuracy"].iloc[0]
    np_key = pd.read_csv(REPO / "results" / "specificity_attempt_round" / "np_margin_family_key_table.csv")
    np_row = np_key[np_key["variant"] == "max_alpha0.05_add0.050"].iloc[0]
    return {
        "target_acc": f"{summary.accuracy_mean:.4f} +/- {summary.accuracy_std:.4f}",
        "target_f1": f"{summary.f1_mean:.4f}",
        "target_recall": f"{summary.recall_mean:.4f}",
        "target_spec": f"{summary.specificity_mean:.4f}",
        "target_delay": f"{summary.median_delay_s_mean:.2f} s",
        "target_p95": f"{summary.p95_delay_s_mean:.2f} s",
        "source_acc": f"{src_acc['mean']:.4f} +/- {src_acc['sem']:.4f} SEM",
        "np_acc": f"{np_row.accuracy_mean:.4f}",
        "np_spec": f"{np_row.specificity_mean:.4f}",
        "np_recall": f"{np_row.recall_mean:.4f}",
    }


R = get_results()


def wrap(s: str, width: int) -> str:
    return "\n".join(textwrap.wrap(s, width=width, break_long_words=False))


def shadow(patch, alpha=0.13):
    patch.set_path_effects(
        [
            pe.SimplePatchShadow(offset=(1.2, -1.2), shadow_rgbFace="#8FA2BD", alpha=alpha),
            pe.Normal(),
        ]
    )


def rounded(ax, x, y, w, h, fc=PANEL_BG, ec=LINE, lw=1.0, r=0.045, dash=False, z=1, alpha=1.0, shade=False):
    p = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.010,rounding_size={r}",
        facecolor=fc,
        edgecolor=ec,
        linewidth=lw,
        linestyle="--" if dash else "-",
        alpha=alpha,
        zorder=z,
    )
    ax.add_patch(p)
    if shade:
        shadow(p)
    return p


def text(ax, x, y, s, size=7, color=TEXT, weight="normal", ha="center", va="center", style="normal", z=5, linespacing=1.1, rotation=0):
    return ax.text(
        x,
        y,
        s,
        fontsize=size,
        color=color,
        fontweight=weight,
        ha=ha,
        va=va,
        style=style,
        zorder=z,
        linespacing=linespacing,
        rotation=rotation,
    )


def arrow(ax, x1, y1, x2, y2, color=BLUE, lw=1.5, ms=16, z=3):
    ax.add_patch(
        FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle="-|>",
            mutation_scale=ms,
            lw=lw,
            color=color,
            shrinkA=2,
            shrinkB=2,
            zorder=z,
        )
    )


def badge(ax, x, y, n):
    rounded(ax, x, y, 0.28, 0.28, fc=BLUE, ec=BLUE, lw=0.8, r=0.045, z=6, shade=True)
    text(ax, x + 0.14, y + 0.15, str(n), size=11.5, color="white", weight="bold", z=7)


def panel(ax, n, title, x, y, w, h):
    rounded(ax, x, y, w, h, fc=PANEL_BG, ec=LINE, lw=1.05, r=0.055, z=1, shade=True)
    badge(ax, x + 0.12, y + 0.11, n)
    text(ax, x + w / 2 + 0.05, y + 0.22, title, size=12.6, color=BLUE, weight="bold", z=6)


def small_note(ax, x, y, w, h, s, ec=BLUE, color=BLUE, fc="#FFFFFF", dash=True, size=6.2):
    rounded(ax, x, y, w, h, fc=fc, ec=ec, lw=0.75, r=0.035, dash=dash, z=3)
    text(ax, x + w / 2, y + h / 2, s, size=size, color=color, weight="bold", style="italic", z=4)


def voltage_trace(ax, x, y, w, h, color=BLUE3, lw=1.1, seed=1, axes=True, drop=True):
    t = np.linspace(0, 1, 160)
    rng = np.random.default_rng(seed)
    yy = 0.34 + 0.18 * t + 0.055 * np.sin(8 * np.pi * t + 0.3)
    yy += 0.025 * np.sin(23 * np.pi * t)
    if drop:
        yy += 0.45 * np.exp(-((t - 0.55) / 0.030) ** 2)
        yy -= 0.08 * np.exp(-((t - 0.66) / 0.035) ** 2)
        yy += 0.13 * np.exp(-((t - 0.78) / 0.038) ** 2)
    yy += rng.normal(0, 0.0025, len(t))
    xs = x + w * t
    ys = y + h * np.clip(yy, 0.05, 0.92)
    if axes:
        ax.add_line(Line2D([x, x + w], [y + h * 0.88, y + h * 0.88], color="#4A5565", lw=0.55, zorder=2))
        ax.add_line(Line2D([x, x], [y + h * 0.88, y + h * 0.10], color="#4A5565", lw=0.55, zorder=2))
    ax.plot(xs, ys, color=color, lw=lw, zorder=4)


def cylinder(ax, x, y, w, h, label, body="#DDE6EF"):
    rounded(ax, x, y + 0.07, w, h - 0.14, fc=body, ec="#415267", lw=0.75, r=w * 0.23, z=3, shade=True)
    ax.add_patch(Ellipse((x + w / 2, y + 0.12), w * 0.88, h * 0.13, facecolor="#F8FAFC", edgecolor="#415267", lw=0.75, zorder=5))
    ax.add_patch(Ellipse((x + w / 2, y + 0.12), w * 0.36, h * 0.055, facecolor="#7AA7D7", edgecolor="#415267", lw=0.55, zorder=6))
    text(ax, x + w / 2, y + h * 0.55, label, size=9.0, color=TEXT, weight="bold", z=7)


def prism(ax, x, y, w, h):
    front = Polygon([(x, y + 0.18), (x + w * 0.76, y + 0.32), (x + w * 0.76, y + h), (x, y + h * 0.84)], closed=True, facecolor="#0E62A8", edgecolor="#092F5D", lw=0.9, zorder=3)
    side = Polygon([(x + w * 0.76, y + 0.32), (x + w, y + 0.18), (x + w, y + h * 0.84), (x + w * 0.76, y + h)], closed=True, facecolor="#064C8C", edgecolor="#092F5D", lw=0.9, zorder=2)
    top = Polygon([(x, y + 0.18), (x + w * 0.25, y), (x + w, y + 0.18), (x + w * 0.76, y + 0.32)], closed=True, facecolor="#2F7EC2", edgecolor="#092F5D", lw=0.9, zorder=4)
    for p in [side, front, top]:
        ax.add_patch(p)
    for cx, col in [(x + w * 0.25, RED), (x + w * 0.55, "#F0F5FA"), (x + w * 0.82, "#222")]:
        ax.add_patch(Ellipse((cx, y + 0.16), w * 0.12, h * 0.050, facecolor=col, edgecolor="#1F2937", lw=0.45, zorder=6))
    text(ax, x + w * 0.42, y + h * 0.56, "LFP\n100Ah", size=9.0, color="white", weight="bold", z=7)


def feature_icon(ax, x, y, kind, col=BLUE2):
    rounded(ax, x, y, 0.34, 0.30, fc="#FFFFFF", ec="#A8B9D3", lw=0.6, r=0.015, z=4)
    if kind == "drop":
        pts = [(x + 0.07, y + 0.07), (x + 0.14, y + 0.19), (x + 0.18, y + 0.17), (x + 0.28, y + 0.25)]
        ax.plot([p[0] for p in pts], [p[1] for p in pts], color=col, lw=1.0, zorder=5)
    elif kind == "slope":
        ax.plot([x + 0.06, x + 0.28], [y + 0.10, y + 0.24], color=col, lw=1.0, zorder=5)
    elif kind == "resid":
        pts = [(x + 0.05, y + 0.22), (x + 0.11, y + 0.10), (x + 0.17, y + 0.23), (x + 0.25, y + 0.13), (x + 0.30, y + 0.20)]
        ax.plot([p[0] for p in pts], [p[1] for p in pts], color=col, lw=1.0, zorder=5)
    elif kind == "compact":
        for i in range(8):
            ax.add_patch(Circle((x + 0.07 + 0.20 * ((i * 37) % 100) / 100, y + 0.08 + 0.15 * ((i * 53) % 100) / 100), 0.014, color=col, zorder=5))
    elif kind == "recover":
        t = np.linspace(0, 1, 30)
        ax.plot(x + 0.05 + 0.25 * t, y + 0.24 - 0.16 / (1 + np.exp(-8 * (t - 0.55))), color=col, lw=1.0, zorder=5)
    else:
        for i, ht in enumerate([0.07, 0.11, 0.16, 0.21]):
            ax.add_line(Line2D([x + 0.08 + i * 0.06, x + 0.08 + i * 0.06], [y + 0.25, y + 0.25 - ht], color=col, lw=1.4, zorder=5))


def tiny_squares(ax, x, y, n=7, color=BLUE3):
    for i in range(n):
        ax.add_patch(Rectangle((x + 0.14 * i, y), 0.09, 0.09, color=color, alpha=0.75, zorder=4))


def draw():
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7,
            "mathtext.fontset": "dejavusans",
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )
    fig, ax = plt.subplots(figsize=(16, 9), dpi=300)
    ax.set_xlim(0, 16)
    ax.set_ylim(9, 0)
    ax.axis("off")
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#FFFFFF")

    # Layout
    x1, x2, x3 = 0.06, 5.20, 10.34
    w1, w2, w3 = 5.02, 5.02, 5.60
    y1, h1 = 0.05, 3.38
    y2, h2 = 3.52, 2.82
    y3, h3 = 6.45, 1.92

    panel(ax, 1, "Data Acquisition and Domains", x1, y1, w1, h1)
    panel(ax, 2, "Voltage Prefix Construction", x2, y1, w2, h1)
    panel(ax, 3, "Feature Engineering", x3, y1, w3, h1)
    panel(ax, 4, "Leakage-Controlled Benchmark", x1, y2, w1, h2)
    panel(ax, 5, "Multi-Horizon Recognition", x2, y2, w2, h2)
    panel(ax, 6, "Validation-Selected Detector", x3, y2, w3, h2)
    panel(ax, 7, "Decision and Output", 0.06, y3, 15.88, h3)

    for a, b, yy in [(x1 + w1, x2, 1.68), (x2 + w2, x3, 1.68), (x1 + w1, x2, 4.78), (x2 + w2, x3, 4.78)]:
        arrow(ax, a + 0.02, yy, b - 0.06, yy, color=BLUE, lw=2.2, ms=24)

    # Panel 1
    text(ax, x1 + 0.84, y1 + 0.68, "Source domain\n(5 Ah, labeled)", size=7.4, color=BLUE, weight="bold")
    text(ax, x1 + 3.72, y1 + 0.68, "Target domain\n(100 Ah, grouped evaluation)", size=7.4, color=BLUE, weight="bold")
    cylinder(ax, x1 + 0.44, y1 + 1.14, 0.55, 1.34, "LFP\n5Ah")
    prism(ax, x1 + 3.45, y1 + 1.10, 1.18, 1.35)
    text(ax, x1 + 2.24, y1 + 1.14, "Voltage-only\nESC benchmark", size=7.3, color=BLUE2, weight="bold")
    voltage_trace(ax, x1 + 1.48, y1 + 1.45, 1.48, 0.83, seed=2, color=BLUE3, lw=1.05)
    text(ax, x1 + 1.24, y1 + 1.84, "Terminal voltage v(t)", size=5.2, color=TEXT, rotation=90)
    text(ax, x1 + 2.22, y1 + 2.35, "Time (s)", size=5.3, color=TEXT)
    text(ax, x1 + 3.10, y1 + 1.86, "...", size=11, color=TEXT)
    small_note(
        ax,
        x1 + 0.42,
        y1 + 2.84,
        4.22,
        0.42,
        "Task: cross-capacity ESC detection using terminal voltage only",
        size=6.4,
    )

    # Panel 2
    text(ax, x2 + 1.08, y1 + 0.62, "Raw terminal voltage v(t)", size=6.3, color=TEXT)
    voltage_trace(ax, x2 + 0.42, y1 + 0.80, 2.18, 0.86, seed=6, color="#343A40", lw=0.9)
    for xi in np.linspace(x2 + 0.78, x2 + 2.28, 6):
        ax.add_line(Line2D([xi, xi], [y1 + 0.86, y1 + 1.62], color=RED, lw=0.65, linestyle="--", zorder=4))
    text(ax, x2 + 1.50, y1 + 1.78, "Fixed-length causal prefix windows", size=6.3, color=TEXT)
    ywin = y1 + 2.05
    ax.add_line(Line2D([x2 + 0.52, x2 + 2.48], [ywin, ywin], color=ORANGE, lw=1.1, zorder=4))
    for i, lab in enumerate(["50s", "75s", "100s", "150s", "250s", "400s"]):
        xx = x2 + 0.52 + i * 0.35
        ax.add_line(Line2D([xx, xx], [ywin - 0.11, ywin + 0.11], color=ORANGE, lw=0.9, zorder=4))
        if i < 4:
            text(ax, xx + 0.03, ywin + 0.30, lab, size=4.9, color=TEXT)
    rounded(ax, x2 + 2.96, y1 + 0.86, 1.72, 0.82, fc=PALE_BLUE, ec="#8EB5E3", lw=0.75, r=0.04)
    text(ax, x2 + 3.82, y1 + 1.09, "Causal prefixes", size=6.6, color=BLUE, weight="bold")
    for i, lab in enumerate(["50", "75", "100", "150", "250", "400"]):
        bx = x2 + 3.05 + 0.25 * i
        rounded(ax, bx, y1 + 1.25, 0.20, 0.24, fc="#FFFFFF", ec="#A5BBD7", lw=0.45, r=0.012)
        text(ax, bx + 0.10, y1 + 1.19, lab, size=4.7, color=TEXT)
    rounded(ax, x2 + 3.10, y1 + 1.86, 1.45, 0.56, fc=GREEN_PALE, ec="#8AB58E", lw=0.75, r=0.04)
    text(ax, x2 + 3.82, y1 + 2.04, "Fold-wise preprocessing", size=6.3, color=GREEN, weight="bold")
    text(ax, x2 + 3.82, y1 + 2.24, "impute + scale on train folds", size=5.6, color=TEXT)
    small_note(ax, x2 + 0.55, y1 + 2.82, 3.95, 0.36, "Principle: causal prefixes avoid look-ahead bias.", size=6.2)

    # Panel 3
    text(ax, x3 + 0.74, y1 + 0.82, "Voltage-only prefix\nfeature vector z_i,j", size=6.2, color=TEXT)
    voltage_trace(ax, x3 + 0.36, y1 + 1.25, 0.98, 0.58, seed=8, color=BLUE3, axes=False, lw=0.9)
    arrow(ax, x3 + 1.38, y1 + 1.55, x3 + 1.72, y1 + 1.55, color="#111111", lw=1.1, ms=13)
    rounded(ax, x3 + 1.82, y1 + 1.06, 0.36, 1.35, fc="#F8FBFF", ec="#A5BBD7", lw=0.75, r=0.04)
    for i in range(7):
        ax.add_patch(Circle((x3 + 2.00, y1 + 1.22 + 0.15 * i), 0.047, color="#82AAD8", zorder=5))
    text(ax, x3 + 2.00, y1 + 2.03, "...", size=9, color=TEXT, z=6)
    arrow(ax, x3 + 2.25, y1 + 1.55, x3 + 2.62, y1 + 1.55, color="#111111", lw=1.1, ms=13)
    rounded(ax, x3 + 2.72, y1 + 0.55, 2.65, 2.55, fc="#FFFFFF", ec="#334E77", lw=0.85, r=0.04)
    rows = [
        ("drop", "Drop magnitude", "drop_norm, max_drop_norm"),
        ("slope", "Slope evolution", "initial/min slope, dv/dt"),
        ("resid", "Residual roughness", "res_std, res_absdiff"),
        ("compact", "Compactness", "abs/signed motion, variance"),
        ("recover", "Recovery behavior", "recovery fraction, ratio"),
        ("mono", "Monotonicity", "sign-change rate, trend score"),
    ]
    for i, (kind, head, desc) in enumerate(rows):
        yy = y1 + 0.68 + i * 0.39
        if i:
            ax.add_line(Line2D([x3 + 2.72, x3 + 5.37], [yy - 0.08, yy - 0.08], color=GRID, lw=0.55, zorder=4))
        feature_icon(ax, x3 + 2.84, yy - 0.13, kind)
        text(ax, x3 + 3.30, yy - 0.03, head, size=6.8, color=BLUE, weight="bold", ha="left")
        text(ax, x3 + 3.30, yy + 0.14, desc, size=5.2, color=TEXT, ha="left")
    rounded(ax, x3 + 0.44, y1 + 2.14, 1.22, 0.48, fc=ORANGE_PALE, ec=ORANGE, lw=0.8, dash=True, r=0.04)
    text(ax, x3 + 1.05, y1 + 2.38, "Hard-negative\nnormal windows\nincluded", size=5.9, color=ORANGE, weight="bold")
    rounded(ax, x3 + 0.16, y1 + 2.94, 2.10, 0.18, fc="#F8FBFF", ec=GRID, lw=0.45, r=0.02)
    text(ax, x3 + 1.21, y1 + 3.03, "No current / power / temperature / impedance features.", size=4.9, color=BLUE)

    # Panel 4
    text(ax, x1 + 0.20, y2 + 0.78, r"$D_s=\{(z_i^s,y_i^s)\}$" + "\n(labeled source)", size=7.8, color=TEXT, ha="left")
    text(ax, x1 + 0.20, y2 + 1.42, r"$D_t=\{(z_i^t,y_i^t,g_i^t)\}$" + "\n(target with group ID)", size=7.5, color=TEXT, ha="left")
    rounded(ax, x1 + 0.24, y2 + 2.00, 1.28, 0.42, fc="#FFFFFF", ec="#A0A0A0", lw=0.65, dash=True, r=0.04)
    text(ax, x1 + 0.88, y2 + 2.21, r"$G^t_{train/val}\cap G^t_{test}=\varnothing$", size=7.2, color=TEXT)
    ax.add_line(Line2D([x1 + 1.82, x1 + 1.82], [y2 + 0.55, y2 + 2.36], color="#8490A0", lw=0.75, linestyle="--", zorder=3))
    text(ax, x1 + 3.25, y2 + 0.64, "Target groups G^t", size=6.7, color=TEXT)
    rounded(ax, x1 + 2.25, y2 + 0.82, 2.32, 0.52, fc="#FFFFFF", ec="#4B5563", lw=0.75, r=0.04)
    for group, col in enumerate([BLUE3, ORANGE, PURPLE]):
        for k in range(5):
            ax.add_patch(Circle((x1 + 2.42 + group * 0.66 + k * 0.10, y2 + 1.14), 0.037, color=col, zorder=5, alpha=0.92))
    for i, (lab, col) in enumerate([("Train", BLUE3), ("Validation", ORANGE), ("Test", PURPLE)]):
        bx = x1 + 2.08 + i * 0.84
        rounded(ax, bx, y2 + 1.58, 0.68, 0.53, fc="#FFFFFF", ec="#B5BCC9", lw=0.65, dash=True, r=0.035)
        text(ax, bx + 0.34, y2 + 1.73, lab + "\nGroups", size=5.7, color=TEXT)
        for k in range(4):
            ax.add_patch(Circle((bx + 0.16 + k * 0.11, y2 + 1.98), 0.033, color=col, zorder=5, alpha=0.88))
    small_note(ax, x1 + 0.95, y2 + 2.40, 3.26, 0.24, "Prevent copied/similar target samples from crossing train/val/test.", size=5.2)

    # Panel 5
    rounded(ax, x2 + 0.18, y2 + 0.42, 2.10, 0.28, fc="#FFFFFF", ec=BLUE, lw=0.7, r=0.035)
    text(ax, x2 + 1.23, y2 + 0.56, "H = {50, 75, 100, 150, 250, 400} s", size=5.9, color=TEXT)
    text(ax, x2 + 0.70, y2 + 1.06, "Voltage\nsequence v(t)", size=6.1, color=TEXT)
    voltage_trace(ax, x2 + 0.35, y2 + 1.30, 1.18, 0.63, seed=12, color=BLUE3, lw=0.95)
    for i, (lab, col, yy) in enumerate([("50", BLUE3, y2 + 1.08), ("75", ORANGE, y2 + 1.55), ("400", PURPLE, y2 + 2.02)]):
        arrow(ax, x2 + 1.65, yy, x2 + 2.17, yy, color="#111111", lw=1.0, ms=11)
        text(ax, x2 + 2.34, yy, rf"$z_i^{{({lab})}}$", size=6.3, color=TEXT)
        tiny_squares(ax, x2 + 2.65, yy - 0.05, n=7, color=col)
        arrow(ax, x2 + 3.72, yy, x2 + 4.04, yy, color="#111111", lw=1.0, ms=11)
        rounded(ax, x2 + 4.08, yy - 0.15, 0.54, 0.30, fc="#FFFFFF", ec="#555555", lw=0.65, r=0.035)
        text(ax, x2 + 4.35, yy, rf"$f_m^{{({lab})}}(\cdot)$", size=5.3, color=TEXT)
    text(ax, x2 + 3.40, y2 + 1.76, r"$\vdots$", size=11, color=TEXT)
    small_note(
        ax,
        x2 + 0.50,
        y2 + 2.42,
        3.90,
        0.30,
        r"Horizon probability:  $p_i^{(h)}=\sum_m w_m f_m^{(h)}(z_i^{(h)})$",
        size=5.9,
    )

    # Panel 6
    text(ax, x3 + 2.76, y2 + 0.60, "Candidate models (per horizon h)", size=6.3, color=TEXT)
    for i, (name, sub) in enumerate([
        ("ResCompact-HGB", "Hist. Gradient Boosting"),
        ("ResCompact-ET", "Extra Trees"),
        ("Global-shape ET", "Global Extra Trees"),
    ]):
        bx = x3 + 0.38 + i * 1.55
        rounded(ax, bx, y2 + 0.82, 1.25, 0.43, fc=PALE_BLUE, ec=BLUE3, lw=0.75, r=0.04)
        text(ax, bx + 0.625, y2 + 0.99, name, size=5.9, color=BLUE, weight="bold")
        text(ax, bx + 0.625, y2 + 1.16, f"({sub})", size=4.5, color=TEXT)
        arrow(ax, bx + 0.625, y2 + 1.27, bx + 0.625, y2 + 1.58, color="#111111", lw=0.9, ms=10)
    rounded(ax, x3 + 1.12, y2 + 1.55, 2.78, 0.62, fc="#FFFFFF", ec=BLUE3, lw=0.8, r=0.04)
    text(ax, x3 + 2.51, y2 + 1.74, "Validation-selected model pool", size=6.2, color=BLUE, weight="bold")
    text(ax, x3 + 2.51, y2 + 1.98, r"$p_i^{(h)}=\sum_m w_m f_m^{(h)}(z_i^{(h)})$", size=6.6, color=TEXT)
    arrow(ax, x3 + 3.94, y2 + 1.86, x3 + 4.28, y2 + 1.86, color="#111111", lw=0.95, ms=11)
    rounded(ax, x3 + 4.32, y2 + 1.55, 1.04, 0.62, fc=GREEN_PALE, ec=GREEN, lw=0.75, r=0.04)
    text(ax, x3 + 4.84, y2 + 1.77, "NP safety-margin\noperating point", size=5.4, color=GREEN, weight="bold")
    text(ax, x3 + 4.84, y2 + 2.04, "(false-alarm control)", size=4.7, color=TEXT)
    rounded(ax, x3 + 0.34, y2 + 2.48, 5.00, 0.30, fc=ORANGE_PALE, ec=ORANGE, lw=0.75, dash=True, r=0.035)
    text(ax, x3 + 2.84, y2 + 2.63, "Model pool, weights, thresholds, and safety margin are chosen on validation only.", size=5.8, color=ORANGE, weight="bold", style="italic")

    # Panel 7
    text(ax, 1.54, y3 + 0.40, "Horizon probabilities", size=6.4, color=TEXT)
    for i, lab in enumerate(["50", "75", "100", "150", "250", "400"]):
        xx = 0.34 + i * 0.47
        text(ax, xx, y3 + 0.74, rf"$p_i^{{({lab})}}$", size=6.0, color=TEXT)
        ax.add_patch(Circle((xx, y3 + 1.06), 0.045, color="#6C94CF", zorder=5))
        text(ax, xx, y3 + 1.34, rf"$\tau_{{{lab}}}$", size=5.4, color=TEXT)
    text(ax, 1.54, y3 + 1.57, "Horizon thresholds", size=5.7, color=TEXT)
    arrow(ax, 3.13, y3 + 1.02, 3.62, y3 + 1.02, color=BLUE, lw=1.8, ms=19)
    rounded(ax, 3.70, y3 + 0.42, 3.80, 1.20, fc="#FFFFFF", ec=BLUE, lw=0.85, dash=True, r=0.04)
    text(ax, 5.60, y3 + 0.60, "Decision rule", size=6.7, color=BLUE, weight="bold")
    text(ax, 5.60, y3 + 0.92, r"$\hat{y}_i=I[\max_{h \in H}(p_i^{(h)}-\tau_h)\geq 0]$", size=9.5, color=TEXT)
    text(ax, 5.60, y3 + 1.32, r"$\hat{t}_i=\min\{h\in H:p_i^{(h)}\geq\tau_h\}$", size=9.1, color=TEXT)
    arrow(ax, 7.62, y3 + 1.02, 8.06, y3 + 1.02, color=BLUE, lw=1.8, ms=19)
    rounded(ax, 8.14, y3 + 0.42, 3.10, 1.20, fc="#FFFFFF", ec=ORANGE, lw=0.85, r=0.04)
    outputs = [
        r"ESC probability: $\max_{h \in H}p_i^{(h)}$",
        r"Predicted label: $\hat{y}_i \in \{0,1\}$",
        r"Estimated alarm time: $\hat{t}_i$ (s)",
    ]
    for i, s in enumerate(outputs):
        yy = y3 + 0.62 + i * 0.35
        if i:
            ax.add_line(Line2D([8.14, 11.24], [yy - 0.10, yy - 0.10], color="#E8C6B3", lw=0.55, zorder=4))
        text(ax, 9.68, yy, s, size=6.1, color=TEXT)
    rounded(ax, 11.68, y3 + 0.18, 3.82, 1.48, fc="#FFFFFF", ec=BLUE, lw=0.9, r=0.04)
    ax.add_patch(Rectangle((11.68, y3 + 0.18), 3.82, 0.28, facecolor=BLUE, edgecolor=BLUE, lw=0, zorder=4))
    text(ax, 13.59, y3 + 0.32, "Verified strict 100Ah performance (grouped test)", size=6.1, color="white", weight="bold", style="italic")
    metrics = [
        ("Accuracy", R["target_acc"]),
        ("F1 score", R["target_f1"]),
        ("Recall (sensitivity)", R["target_recall"]),
        ("Specificity", R["target_spec"]),
        ("Median detection delay", R["target_delay"]),
        ("P95 detection delay", R["target_p95"]),
    ]
    for i, (k, v) in enumerate(metrics):
        yy = y3 + 0.62 + i * 0.16
        ax.add_line(Line2D([11.85, 15.34], [yy + 0.09, yy + 0.09], color=GRID, lw=0.45, zorder=4))
        text(ax, 11.94, yy, k, size=5.7, color=TEXT, ha="left")
        text(ax, 13.62, yy, v, size=5.7, color=TEXT, ha="left")

    rounded(ax, 0.08, 8.48, 15.84, 0.42, fc="#FFFFFF", ec=BLUE, lw=0.75, dash=True, r=0.035)
    text(
        ax,
        8.0,
        8.69,
        "Evaluation contract: 30 admissible random seeds; duplicate-aware grouped repeated splits; validation-only selection/calibration; voltage-only 100Ah 95%+ accuracy is not claimed.",
        size=6.1,
        color=BLUE,
        weight="bold",
        style="italic",
    )

    fig.savefig(PNG, bbox_inches="tight", pad_inches=0.02, dpi=300)
    fig.savefig(PNG_2X, bbox_inches="tight", pad_inches=0.02, dpi=240)
    plt.close(fig)


if __name__ == "__main__":
    draw()
    print(PNG)
    print(PNG_2X)
