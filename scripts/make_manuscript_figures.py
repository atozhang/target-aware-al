#!/usr/bin/env python
"""Redraw the Fig 3D and Fig 4C data panels at manuscript size and style.

Writes to ``figures/manuscript/``:

    fig3d_wittig_published_split.{pdf,png}   Fig 3D: Wittig transfer-task learning curves
                                             (mean +/- SEM, 2-12 % window shaded) with the 80 %
                                             efficiency target as a dotted red line;
                                             3.3 x 2.3 in, 300 dpi
    fig4c_benchmark_performance.{pdf,png}    Fig 4C: highest-mean (solid) and statistically
                                             competitive (pale) shares of tasks at AUC2-12, by
                                             split family and by dataset; 6.0 in, 600 dpi,
                                             transparent

Values are read from the released tables only:

    results/wittig_transfer/transfer_curve.csv
    results/efficiency/efficiency_per_task.csv (the transfer task's q = 0.8 target R2)
    supporting_information/tables/competitiveness/{highest_mean,competitive}_by_{split_family,dataset}.csv

Arial is used when installed, otherwise the bundled TeX Gyre Heros. Run from the
repository root:

    python scripts/make_manuscript_figures.py
"""
from __future__ import annotations

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as font_manager  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from make_figures import COLORS, DISPLAY, ORDER  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "figures/manuscript"
TARGET_LINE_COLOR = "#CC3311"   # dotted red line at the 80 % efficiency target (Fig 3D)
TARGET_LINE_LABEL = "80% of achievable improvement"


def register_fonts() -> str:
    """Arial if installed, else the bundled TeX Gyre Heros."""
    for path in sorted((REPO / "figures/fonts").glob("**/*.[ot]tf")):
        font_manager.fontManager.addfont(str(path))
    names = {f.name for f in font_manager.fontManager.ttflist}
    return "Arial" if "Arial" in names else "TeX Gyre Heros" if "TeX Gyre Heros" in names else "DejaVu Sans"


def base_style(family: str, size: float, **extra) -> None:
    matplotlib.rcdefaults()
    plt.rcParams.update({
        "font.family": family, "font.size": size, "axes.labelsize": size, "xtick.labelsize": size,
        "ytick.labelsize": size, "legend.fontsize": size, "axes.spines.top": False,
        "axes.spines.right": False, "axes.linewidth": 0.8, "axes.edgecolor": "#333333",
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white", **extra})


# --------------------------------------------------------------------------- #
#  Fig 3D -- half-column Wittig transfer-task curve
# --------------------------------------------------------------------------- #

def figure_3d(family: str) -> list[Path]:
    curve = pd.read_csv(REPO / "results/wittig_transfer/transfer_curve.csv")
    if set(curve["AF"]) != set(ORDER):
        raise ValueError("transfer curve does not carry all five methods")
    base_style(family, 7)
    fig, ax = plt.subplots(figsize=(3.3, 2.30))
    fig.subplots_adjust(left=0.17, right=0.975, bottom=0.20, top=0.965)
    ax.axvspan(2, 12, color="#8899AA", alpha=0.14, zorder=0)
    for method in ORDER:
        group = curve[curve["AF"].eq(method)].sort_values("acquisition_fraction")
        x = group["acquisition_fraction"].to_numpy(float) * 100
        y = group["mean_r2"].to_numpy(float)
        sem = group["sem_r2"].to_numpy(float)
        ax.plot(x, y, color=COLORS[method], lw=1.35, zorder=3)
        ax.fill_between(x, y - sem, y + sem, color=COLORS[method], alpha=0.14, lw=0, zorder=2)
    # the q = 0.8 efficiency target of this task: R0 + 0.8 (Rmax,30 - R0), as used by efficiency_endpoint.py
    eff = pd.read_csv(REPO / "results/efficiency/efficiency_per_task.csv")
    target = eff[eff["dataset"].eq("Wittig") & eff["split_family"].eq("paper_defined")]["target_score"].unique()
    if len(target) != 1:
        raise ValueError("expected one efficiency target for the Wittig transfer task")
    ax.axhline(float(target[0]), color=TARGET_LINE_COLOR, lw=1.2, ls=":", zorder=4)
    handles = [plt.Line2D([], [], color=COLORS[m], lw=1.6) for m in ORDER]
    handles.append(plt.Line2D([], [], color=TARGET_LINE_COLOR, lw=1.2, ls=":"))
    ax.legend(handles, [DISPLAY[m] for m in ORDER] + [TARGET_LINE_LABEL], loc="lower center", bbox_to_anchor=(0.5, 0.012),
              ncol=3, frameon=False, fontsize=6.1, handlelength=1.25, handletextpad=0.35,
              columnspacing=0.65, labelspacing=0.12, borderaxespad=0.12)
    # y-limits: at least -0.41 .. 0.31, widened on the 0.05 / 0.1 grid as far as the mean +/- SEM bands need
    lo = float((curve["mean_r2"] - curve["sem_r2"]).min())
    hi = float((curve["mean_r2"] + curve["sem_r2"]).max())
    ylim = (min(-0.41, float(np.floor(lo * 20) / 20)), max(0.31, float(np.ceil(hi * 10) / 10 + 0.01)))
    ax.set_xlim(0, 30)
    ax.set_ylim(*ylim)
    ax.set_yticks(np.round(np.arange(np.ceil(ylim[0] * 10) / 10, ylim[1] + 1e-9, 0.1), 1))
    ax.set_xlabel("CAS reactions acquired (% of pool)")
    ax.set_ylabel("Target $R^2$")
    ax.grid(False)
    return save(fig, "fig3d_wittig_published_split", pdf_kwargs={"metadata": {"CreationDate": None}}, dpi=300)


# --------------------------------------------------------------------------- #
#  Fig 4C -- highest mean and statistically competitive, stacked
# --------------------------------------------------------------------------- #

SPLIT_SCOPES = ["Published", "Chemical family", "Cluster", "Random"]
DATASET_SCOPES = ["BH", "Rh", "SAD", "Wittig"]
SPLIT_LABELS = ["Published", "Chemical\nFamily", "Cluster", "Random"]
DATASET_LABELS = ["Buchwald-\nHartwig", "Rh asymmetric\nhydrogenation", "Sharpless\ndihydroxylation", "Wittig\nolefination"]
SHADES = {"BH": "#FCE8E6", "Rh": "#E6F4EA", "SAD": "#E8F0FE", "Wittig": "#F5EFDC"}
PALE, SLOT = 0.32, 0.15
TITLE_SIZE, LABEL_SIZE, KEY_GRAY = 8.0, 7.5, "#4D4D4D"
M_LEFT, M_RIGHT, M_GAP = 0.44, 0.03, 0.14
TITLE_IN, KEY_IN, LABEL_LINE_IN, LABEL_PAD_IN = 0.17, 0.19, 0.135, 0.055
SPLIT_SHARE = 0.435   # the split panel's labels are short, so it gives width to the dataset panel


def counts_frame(kind: str, stem: str, scopes: list[str]) -> pd.DataFrame:
    df = pd.read_csv(REPO / f"supporting_information/tables/competitiveness/{kind}_by_{stem}.csv")
    rows = []
    for scope in scopes:
        for method in ORDER:
            cell = df[(df["scope"] == scope) & (df["acquisition_function"] == DISPLAY[method])]["2-12"].iloc[0]
            n, big_n, pct = map(int, re.fullmatch(r"(\d+)/(\d+) \((\d+)%\)", cell).groups())
            rows.append({"scope": scope, "method": method, "n": n, "N": big_n, "percent": pct})
    return pd.DataFrame(rows)


def panel_frame(stem: str, scopes: list[str]) -> pd.DataFrame:
    hi = counts_frame("highest_mean", stem, scopes).rename(columns={"n": "n_hi", "percent": "pct_hi"})
    co = counts_frame("competitive", stem, scopes).rename(columns={"n": "n_co", "percent": "pct_co"})
    df = hi.merge(co[["scope", "method", "n_co", "pct_co"]], on=["scope", "method"])
    if (df["n_co"] < df["n_hi"]).any():
        raise ValueError("competitive count below highest-mean count; the classifications should nest")
    return df


def draw_stacked(ax, frame: pd.DataFrame, scopes: list[str], slot: float) -> None:
    for j, method in enumerate(ORDER):
        x = np.arange(len(scopes)) + (j - 2) * slot
        sub = frame[frame["method"] == method].set_index("scope").loc[scopes]
        co, hi = sub["pct_co"].to_numpy(float), sub["pct_hi"].to_numpy(float)
        ax.bar(x, hi, slot * 0.9, color=COLORS[method], linewidth=0, zorder=3)
        ax.bar(x, co - hi, slot * 0.9, bottom=hi, color=COLORS[method], alpha=PALE, linewidth=0.4,
               edgecolor="white", zorder=2)
        for xi, c in zip(x, co):
            if c == 0.0:  # a measured zero must be visible, not just an absent bar
                ax.hlines(0.4, xi - slot * 0.45, xi + slot * 0.45, color=COLORS[method], linewidth=1.5, zorder=5)


def frame_axes(ax, scopes, labels, ylabel) -> None:
    ax.set_xlim(-0.5, len(scopes) - 0.5)
    ax.set_ylim(0, 104)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_xticks(range(len(scopes)))
    ax.set_xticklabels(labels, fontsize=LABEL_SIZE)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=TITLE_SIZE)
    ax.tick_params(axis="both", which="major", length=2.5, width=0.8, pad=2)


def figure_4c(family: str) -> list[Path]:
    split, dataset = panel_frame("split_family", SPLIT_SCOPES), panel_frame("dataset", DATASET_SCOPES)
    for frame in (split, dataset):  # each partition covers the 118 tasks exactly once
        if not frame.groupby("method")["N"].sum().eq(118).all():
            raise ValueError("a Panel C partition does not cover 118 tasks")
    base_style(family, 8, **{"axes.titlesize": 9, "axes.grid": False, "savefig.dpi": 300})
    width, ax_h = 6.0, 1.20
    deepest = max(l.count("\n") + 1 for l in DATASET_LABELS + SPLIT_LABELS)
    bottom = LABEL_PAD_IN + deepest * LABEL_LINE_IN
    span = width - M_LEFT - M_RIGHT - M_GAP
    w_left, w_right = span * SPLIT_SHARE, span * (1.0 - SPLIT_SHARE)
    height = KEY_IN + TITLE_IN + ax_h + bottom
    fig = plt.figure(figsize=(width, height))
    lefts = [M_LEFT, M_LEFT + w_left + M_GAP]
    axes = [fig.add_axes([l / width, bottom / height, w / width, ax_h / height]) for l, w in zip(lefts, (w_left, w_right))]
    for ax, ax_w, frame, scopes, labels, title, ylab, shade in (
        (axes[0], w_left, split, SPLIT_SCOPES, SPLIT_LABELS, "Results by split family", "Tasks (%)", False),
        (axes[1], w_right, dataset, DATASET_SCOPES, DATASET_LABELS, "Results by dataset", None, True),
    ):
        if shade:
            for i, scope in enumerate(scopes):
                ax.axvspan(i - 0.5, i + 0.5, color=SHADES[scope], linewidth=0, zorder=-2)
        # bars stay physically identical in both panels; only the group spacing differs
        draw_stacked(ax, frame, scopes, SLOT * w_left / ax_w)
        frame_axes(ax, scopes, labels, ylab)
        ax.set_title(title, loc="left", fontsize=TITLE_SIZE, fontweight="bold",
                     fontfamily=plt.rcParams["font.family"], pad=3)
    axes[1].tick_params(labelleft=False)
    keys = [Patch(facecolor=KEY_GRAY, edgecolor="none", label="Highest mean AUC"),
            Patch(facecolor=(0.30, 0.30, 0.30, PALE), edgecolor=KEY_GRAY, linewidth=0.6, label="Statistically competitive")]
    fig.legend(handles=keys, loc="upper left", bbox_to_anchor=(M_LEFT / width, 1.0), ncol=2, frameon=False,
               fontsize=7, handlelength=1.15, handletextpad=0.45, columnspacing=1.5, borderpad=0.0, borderaxespad=0.15)
    return save(fig, "fig4c_benchmark_performance", pdf_kwargs={"metadata": {"CreationDate": None}, "transparent": True},
                png_kwargs={"transparent": True}, dpi=600)


def save(fig, stem: str, *, pdf_kwargs: dict | None = None, png_kwargs: dict | None = None, dpi: int = 300) -> list[Path]:
    OUT.mkdir(parents=True, exist_ok=True)
    paths = [OUT / f"{stem}.pdf", OUT / f"{stem}.png"]
    fig.savefig(paths[0], **(pdf_kwargs or {}))
    fig.savefig(paths[1], dpi=dpi, **(png_kwargs or {}))
    plt.close(fig)
    return paths


def main() -> int:
    family = register_fonts()
    written = figure_3d(family) + figure_4c(family)
    print(f"font family: {family}")
    for path in written:
        print(f"  {path.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
