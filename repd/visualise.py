"""Matplotlib visualisation helpers for REPD data.

All functions return the matplotlib Figure so callers can save, display, or
further customise the output.

Dark-theme defaults match the project's aesthetic, but every palette value is
an overrideable parameter.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import geopandas as gpd
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from matplotlib.figure import Figure

_BG = "#13100d"
_EDGE_DARK = "#332418"
_TEXT = "#f5f0eb"
_MUTED = "#8a8278"
_LABEL = "#b8b0a6"


def _tech_palette(df: pd.DataFrame) -> dict[str, tuple[float, float, float]]:
    """HSV-based colour palette keyed by Technology Type."""
    techs = sorted(df["Technology Type"].dropna().unique())
    n = len(techs)
    return {t: mcolors.hsv_to_rgb((i / n, 0.85, 0.92)) for i, t in enumerate(techs)}


def _style_ax(ax: plt.Axes, bg: str = _BG) -> None:
    ax.set_facecolor(bg)
    ax.tick_params(colors=_MUTED)
    for spine in ax.spines.values():
        spine.set_edgecolor(_EDGE_DARK)
    ax.set_axisbelow(True)


def plot_status_map(
    gdf: gpd.GeoDataFrame,
    lauth_path: str | Path,
    title: str,
    marker: str = "x",
    size: int = 28,
    linewidths: float = 0.9,
    alpha: float = 0.75,
    figsize: tuple[int, int] = (10, 14),
    credit: str = "",
) -> Figure:
    """Plot REPD project points on a UK local authority basemap.

    Args:
        gdf: GeoDataFrame of projects to plot (already filtered to the desired
            status group). Must have a valid ``geometry`` column.
        lauth_path: Path to the ``localauth.json`` GeoJSON boundary file.
        title: Plot title.
        marker: Matplotlib marker style (e.g. ``"x"``, ``"2"``, ``"o"``).
        size: Marker size (``s`` parameter).
        linewidths: Marker line width.
        alpha: Marker alpha.
        figsize: Figure size in inches.
        credit: Optional attribution string rendered in the bottom-right corner.

    Returns:
        Matplotlib Figure.
    """
    lauth = gpd.read_file(lauth_path)
    colors = _tech_palette(gdf)

    fig, ax = plt.subplots(figsize=figsize, facecolor=_BG)
    _style_ax(ax)

    lauth.plot(
        ax=ax, facecolor="none", edgecolor="#6b5040",
        linewidth=0.3, alpha=0.2, zorder=1,
    )

    for tech, group in gdf.groupby("Technology Type"):
        ax.scatter(
            group.geometry.x, group.geometry.y,
            c=[colors[tech]], marker=marker, s=size,
            linewidths=linewidths, alpha=alpha, label=tech, zorder=3,
        )

    ax.grid(color="#ffffff", alpha=0.12, linewidth=0.5, linestyle="--")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    ax.set_title(
        title, fontsize=16, fontweight="bold", color=_TEXT, pad=16,
        path_effects=[pe.withStroke(linewidth=3, foreground=_BG)],
    )
    legend = ax.legend(
        title="Technology Type", loc="lower left", fontsize=7, title_fontsize=8,
        framealpha=0.5, facecolor="#1d1810", edgecolor=_EDGE_DARK, labelcolor=_LABEL,
    )
    legend.get_title().set_color("#a09890")

    if credit:
        fig.text(0.99, 0.01, credit, ha="right", va="bottom",
                 fontsize=7, color="#54483e", alpha=0.7)

    plt.tight_layout()
    return fig


def plot_outcome_summary(
    df: pd.DataFrame,
    year: int | None = None,
    figsize: tuple[int, int] = (15, 7),
    credit: str = "",
) -> Figure:
    """Stacked bar chart + donut chart showing project outcomes by technology.

    Args:
        df: Processed REPD DataFrame (with datetime columns parsed).
        year: If given, restrict to rows whose ``Record Last Updated`` year
            matches. Pass ``None`` to use the full dataset.
        figsize: Figure size in inches.
        credit: Optional attribution string.

    Returns:
        Matplotlib Figure.
    """
    from repd.constants import (
        SUCCESSFUL_DEVELOPMENT_TYPES,
        UNSUCCESSFUL_DEVELOPMENT_TYPES,
    )

    subset = (
        df[df["Record Last Updated (dd/mm/yyyy)"].dt.year == year].copy()
        if year is not None
        else df.copy()
    )

    categories = {
        "Successful": subset[subset["Development Status (short)"].isin(SUCCESSFUL_DEVELOPMENT_TYPES)],
        "Unsuccessful": subset[subset["Development Status (short)"].isin(UNSUCCESSFUL_DEVELOPMENT_TYPES)],
        "Awaiting Construction": subset[subset["Development Status (short)"] == "Awaiting Construction"],
        "Application Submitted": subset[subset["Development Status (short)"] == "Application Submitted"],
    }
    counts = {k: len(v) for k, v in categories.items()}
    total = sum(counts.values())

    palette = {
        "Successful": "#38a870",
        "Unsuccessful": "#d44030",
        "Awaiting Construction": "#d4a830",
        "Application Submitted": "#c8601e",
    }

    all_techs = sorted(df["Technology Type"].dropna().unique())
    n_t = len(all_techs)
    tech_colors = {
        t: mcolors.hsv_to_rgb((i / n_t, 0.85, 0.92)) for i, t in enumerate(all_techs)
    }

    fig, axes = plt.subplots(1, 2, figsize=figsize, facecolor=_BG)

    # ── Stacked bar ───────────────────────────────────────────────────────────
    ax1 = axes[0]
    _style_ax(ax1)
    x = np.arange(4)
    bottom = np.zeros(4)
    dfs_ordered = list(categories.values())

    for tech in all_techs:
        vals = np.array([len(d[d["Technology Type"] == tech]) for d in dfs_ordered])
        ax1.bar(x, vals, 0.5, bottom=bottom, color=tech_colors[tech], label=tech, alpha=0.85)
        bottom += vals

    ax1.set_xticks(x)
    ax1.set_xticklabels(list(counts.keys()), color=_LABEL, fontsize=10, rotation=12, ha="right")
    ax1.set_ylabel("Number of Projects", color=_MUTED, fontsize=10)
    title_suffix = f" ({year})" if year else ""
    ax1.set_title(f"Projects by Outcome & Technology{title_suffix}", color=_TEXT, fontsize=13, fontweight="bold")
    ax1.grid(axis="y", color="#ffffff", alpha=0.05, linewidth=0.5)
    ax1.legend(fontsize=6, loc="upper right", ncol=2, framealpha=0.3,
               facecolor="#1d1810", edgecolor=_EDGE_DARK, labelcolor=_LABEL)
    for xi, cnt in enumerate(counts.values()):
        ax1.text(xi, bottom[xi] + 3, str(cnt), ha="center", color=_TEXT, fontsize=11, fontweight="bold")

    # ── Donut ─────────────────────────────────────────────────────────────────
    ax2 = axes[1]
    ax2.set_facecolor(_BG)
    wedge_colors = [palette[k] for k in counts]
    ax2.pie(
        list(counts.values()), colors=wedge_colors, startangle=90,
        wedgeprops=dict(width=0.55, edgecolor=_BG, linewidth=2),
    )
    ax2.text(0, 0, f"{total}\nprojects", ha="center", va="center",
             fontsize=14, color=_TEXT, fontweight="bold")

    handles = [
        mpatches.Patch(color=wedge_colors[i], label=f"{k}  {v} ({v / total * 100:.1f}%)")
        for i, (k, v) in enumerate(counts.items())
    ]
    ax2.legend(handles=handles, loc="lower center", fontsize=9,
               framealpha=0.3, facecolor="#1d1810", edgecolor=_EDGE_DARK, labelcolor=_LABEL)
    ax2.set_title(f"Outcome Distribution{title_suffix}", color=_TEXT, fontsize=13, fontweight="bold")

    if credit:
        fig.text(0.99, 0.01, credit, ha="right", va="bottom",
                 fontsize=7, color="#54483e", alpha=0.7)

    plt.tight_layout()
    return fig


def plot_delay_distribution(
    df: pd.DataFrame,
    from_col: str,
    to_col: str,
    title: str,
    bar_color: str = "#c8601e",
    figsize: tuple[int, int] = (16, 7),
    min_tech_count: int = 3,
    credit: str = "",
) -> Figure:
    """Histogram + horizontal box-by-technology for a planning delay pair.

    Computes delay as ``to_col - from_col`` in years for rows where both
    columns are non-null and the result is non-negative.

    Args:
        df: Processed REPD DataFrame with parsed datetime columns.
        from_col: Start date column (e.g. ``"Planning Permission Granted"``).
        to_col: End date column (e.g. ``"Under Construction"``).
        title: Shared chart title prefix.
        bar_color: Histogram fill colour.
        figsize: Figure size in inches.
        min_tech_count: Minimum projects per technology to include in box plot.
        credit: Optional attribution string.

    Returns:
        Matplotlib Figure.
    """
    work = df[df[from_col].notna() & df[to_col].notna()].copy()
    work["delay_days"] = (work[to_col] - work[from_col]).dt.days
    work = work[work["delay_days"] >= 0].copy()
    work["delay_years"] = work["delay_days"] / 365.25

    mean_y = work["delay_years"].mean()
    median_y = work["delay_years"].median()
    print(f"{title}:  n={len(work):,}  mean={mean_y:.1f} yrs  median={median_y:.1f} yrs")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, facecolor=_BG)

    # ── Histogram ─────────────────────────────────────────────────────────────
    _style_ax(ax1)
    bins = np.arange(0, work["delay_years"].max() + 1, 0.5)
    ax1.hist(work["delay_years"], bins=bins, color=bar_color, alpha=0.82,
             edgecolor=_BG, linewidth=0.5)
    ax1.axvline(mean_y, color="#d4a830", linewidth=2, linestyle="--",
                label=f"Mean   {mean_y:.1f} yrs")
    ax1.axvline(median_y, color="#38a870", linewidth=2, linestyle=":",
                label=f"Median {median_y:.1f} yrs")
    ax1.set_title(title, color=_TEXT, fontsize=13, fontweight="bold")
    ax1.set_xlabel("Years", color=_MUTED, fontsize=10)
    ax1.set_ylabel("Number of Projects", color=_MUTED, fontsize=10)
    ax1.grid(axis="y", color="#ffffff", alpha=0.05, linewidth=0.5)
    ax1.legend(fontsize=9, framealpha=0.4, facecolor="#1d1810",
               edgecolor=_EDGE_DARK, labelcolor=_LABEL)

    # ── Box plot by technology ─────────────────────────────────────────────────
    _style_ax(ax2)
    tech_order = (
        work.groupby("Technology Type")["delay_years"]
        .median()
        .sort_values(ascending=False)
        .index.tolist()
    )
    tech_order = [t for t in tech_order if len(work[work["Technology Type"] == t]) >= min_tech_count]
    data_groups = [work[work["Technology Type"] == t]["delay_years"].to_numpy() for t in tech_order]
    tech_cmap = plt.get_cmap("tab20", len(tech_order))

    bp = ax2.boxplot(
        data_groups, vert=False, patch_artist=True, widths=0.55,
        medianprops=dict(color="#d4a830", linewidth=2.5),
        whiskerprops=dict(color=_MUTED, linewidth=1),
        capprops=dict(color=_MUTED, linewidth=1.5),
        flierprops=dict(marker="x", color="#d44030", markersize=4, alpha=0.6),
        boxprops=dict(linewidth=0.8),
    )
    for patch, idx in zip(bp["boxes"], range(len(tech_order))):
        patch.set_facecolor(tech_cmap(idx))
        patch.set_alpha(0.75)

    ax2.set_yticks(range(1, len(tech_order) + 1))
    ax2.set_yticklabels(tech_order, color=_LABEL, fontsize=8)
    ax2.set_xlabel("Years", color=_MUTED, fontsize=10)
    ax2.set_title(f"By Technology  ({title})", color=_TEXT, fontsize=13, fontweight="bold")
    ax2.grid(axis="x", color="#ffffff", alpha=0.05, linewidth=0.5)

    if credit:
        fig.text(0.99, 0.01, credit, ha="right", va="bottom",
                 fontsize=7, color="#54483e", alpha=0.7)

    plt.tight_layout()
    return fig


def plot_yearly_status_stacked(
    counts: pd.DataFrame,
    year_range: tuple[int, int] | None = None,
    palette: dict[str, str] | None = None,
    figsize: tuple[int, int] = (16, 7),
    credit: str = "",
) -> Figure:
    """Stacked bar chart of projects per year, broken down by status.

    X axis: calendar year.
    Y axis: number of projects, stacked by status (Operational, Denied,
    Under Construction, Awaiting Construction).

    Args:
        counts: DataFrame from :meth:`REPDProcessor.yearly_status_counts`
            — integer year index, one column per status label.
        year_range: Optional ``(start, end)`` to clip sparse years
            (e.g. ``(2000, 2025)``).
        palette: Optional ``{label: hex_colour}`` overrides.
        figsize: Figure size in inches.
        credit: Optional attribution string rendered bottom-right.

    Returns:
        Matplotlib Figure.
    """
    _DEFAULT_PALETTE: dict[str, str] = {
        "Operational": "#38a870",
        "Denied": "#d44030",
        "Under Construction": "#c8601e",
        "Awaiting Construction": "#d4a830",
    }
    color_map = {**_DEFAULT_PALETTE, **(palette or {})}

    work = counts.copy()
    if year_range is not None:
        work = work.loc[year_range[0]: year_range[1]]

    years = work.index.tolist()
    x = np.arange(len(years))
    labels = work.columns.tolist()

    fallback = plt.get_cmap("tab20", len(labels))
    bar_colors = [
        color_map.get(lbl, mcolors.to_hex(fallback(i)))
        for i, lbl in enumerate(labels)
    ]

    fig, ax = plt.subplots(figsize=figsize, facecolor=_BG)
    _style_ax(ax)

    bottom = np.zeros(len(years))
    for lbl, color in zip(labels, bar_colors):
        vals = work[lbl].to_numpy()
        ax.bar(x, vals, bottom=bottom, color=color, label=lbl,
               alpha=0.88, edgecolor=_BG, linewidth=0.4)
        bottom += vals

    step = max(1, len(years) // 20)
    ax.set_xticks(x[::step])
    ax.set_xticklabels(
        [str(y) for y in years[::step]],
        color=_LABEL, fontsize=9, rotation=45, ha="right",
    )
    ax.set_xlabel("Year", color=_MUTED, fontsize=10)
    ax.set_ylabel("Number of projects", color=_MUTED, fontsize=10)
    ax.set_title(
        "UK Renewable Energy Projects by Status per Year",
        color=_TEXT, fontsize=14, fontweight="bold", pad=14,
        path_effects=[pe.withStroke(linewidth=3, foreground=_BG)],
    )
    ax.grid(axis="y", color="#ffffff", alpha=0.05, linewidth=0.5)
    ax.legend(
        loc="upper left", fontsize=9, framealpha=0.4,
        facecolor="#1d1810", edgecolor=_EDGE_DARK, labelcolor=_LABEL,
    )

    if credit:
        fig.text(0.99, 0.01, credit, ha="right", va="bottom",
                 fontsize=7, color="#54483e", alpha=0.7)

    plt.tight_layout()
    return fig


def plot_stacked_bars(
    data: pd.DataFrame,
    title: str,
    ylabel: str,
    figsize: tuple[int, int] = (16, 8),
    credit: str = "",
    colors: dict[str, tuple[float, float, float]] | None = None,
) -> Figure:
    """One stacked bar chart, e.g. applications or MW per year by technology.

    Args:
        data: Index = x categories (e.g. years), columns = stack series
            (e.g. technologies). Column order sets stack order (first column
            at the bottom) and legend order.
        title: Panel title.
        ylabel: Y-axis label.
        figsize: Figure size in inches.
        credit: Optional attribution string.
        colors: Optional colour per column; pass the same dict to several
            charts so a technology keeps its colour across them. Missing
            columns get an HSV wheel colour.

    Returns:
        Matplotlib Figure.
    """
    x = np.arange(len(data.index))
    n = len(data.columns)
    colors = {c: (colors or {}).get(c, mcolors.hsv_to_rgb((i / n, 0.85, 0.92))) for i, c in enumerate(data.columns)}

    fig, ax = plt.subplots(figsize=figsize, facecolor=_BG)
    _style_ax(ax)
    base = np.zeros(len(x))
    for col in data.columns:
        vals = data[col].to_numpy(dtype=float)
        ax.bar(x, vals, bottom=base, color=colors[col], label=col, alpha=0.88, edgecolor=_BG, linewidth=0.4)
        base += vals
    ax.set_ylabel(ylabel, color=_MUTED, fontsize=11)
    ax.set_title(title, color=_TEXT, fontsize=15, fontweight="bold", pad=10, loc="left")
    ax.grid(axis="y", color="#ffffff", alpha=0.05, linewidth=0.5)
    for spine in ("right", "top"):
        ax.spines[spine].set_visible(False)
    ax.legend(loc="upper left", fontsize=10, framealpha=0.4,
              facecolor="#1d1810", edgecolor=_EDGE_DARK, labelcolor=_LABEL)
    ax.set_xticks(x, [str(i) for i in data.index], color=_LABEL, fontsize=10, rotation=45, ha="right")

    if credit:
        fig.text(0.99, 0.01, credit, ha="right", va="bottom", fontsize=7, color="#54483e", alpha=0.7)

    plt.tight_layout()
    return fig


def plot_choropleth(
    df: pd.DataFrame,
    lauth_path: str | Path,
    col: str,
    group_col: str,
    title: str,
    cmap: str = "Blues",
    top_n_labels: int = 5,
    figsize: tuple[int, int] = (12, 14),
    credit: str = "",
    authority_aliases: dict[str, str] | None = None,
) -> Figure:
    """Choropleth of project counts per local authority.

    Args:
        df: Processed REPD DataFrame.
        lauth_path: Path to the ``localauth.json`` boundary GeoJSON.
        col: Column name to use for the count (will be created by groupby).
        group_col: Column on the REPD side to match to ``LAD24NM``
            (defaults to ``"Planning Authority"``).
        title: Plot title.
        cmap: Matplotlib colormap name.
        top_n_labels: Number of top authorities to annotate on the map.
        figsize: Figure size in inches.
        credit: Optional attribution string.
        authority_aliases: Optional dict mapping REPD authority names to their
            ``LAD24NM`` equivalents where they diverge.

    Returns:
        Matplotlib Figure.
    """
    lauth = gpd.read_file(lauth_path)

    work = df.copy()
    work["_auth"] = work[group_col].str.strip()
    if authority_aliases:
        work["_auth"] = work["_auth"].replace(authority_aliases)

    counts = work.groupby("_auth").size().reset_index(name=col)
    geo = lauth.merge(counts, left_on="LAD24NM", right_on="_auth", how="left")
    geo[col] = geo[col].fillna(0).astype(int)

    fig, ax = plt.subplots(figsize=figsize, facecolor="#0f0f1a")
    ax.set_facecolor("#0f0f1a")
    vmax = max(geo[col].max(), 1)

    geo.plot(
        column=col, ax=ax, cmap=cmap, vmin=0, vmax=vmax,
        missing_kwds=dict(color="#1c1c2e", edgecolor="#334466", linewidth=0.25),
        edgecolor="#334466", linewidth=0.25,
    )

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=vmax))
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.02)
    cb.ax.tick_params(colors="#888899", labelsize=7)
    cb.set_label("Project count", color="#888899", fontsize=8)

    top = geo.nlargest(top_n_labels, col).copy()
    top["centroid"] = top.geometry.centroid
    for _, row in top.iterrows():
        ax.annotate(
            f"{row['LAD24NM']}\n({int(row[col])})",
            xy=(row["centroid"].x, row["centroid"].y),
            fontsize=5.5, ha="center", va="center", color="#f0f0f0",
            fontweight="bold",
            path_effects=[pe.withStroke(linewidth=1.5, foreground="#0f0f1a")],
        )

    ax.set_title(title, fontsize=14, fontweight="bold", color="#e8e8f0", pad=12,
                 path_effects=[pe.withStroke(linewidth=3, foreground="#0f0f1a")])
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    for spine in ax.spines.values():
        spine.set_edgecolor("#333355")

    if credit:
        fig.text(0.99, 0.01, credit, ha="right", va="bottom",
                 fontsize=7, color="#555577", alpha=0.7)

    plt.tight_layout()
    return fig


def plot_time_sankey(
    stats: pd.DataFrame,
    title: str,
    node_colors: dict[str, str],
    order: list[str] | None = None,
    legend: dict[str, str] | None = None,
    xlabel: str = "Months since application submitted (cumulative mean)",
    figsize: tuple[int, int] = (18, 9),
    credit: str = "",
) -> Figure:
    """Sankey whose x-axis is time: link length = mean months spent on that step.

    Each node sits at the cumulative mean months from the root, so the
    horizontal distance between two stages is the average time taken. Links
    are labelled with mean ± SD and median months; thin links into a leaf put
    the label beside the leaf instead.

    Args:
        stats: One row per link with a ``(from, to)`` MultiIndex and columns
            ``projects``, ``mean``, ``sd``, ``median`` (months). Optional bool
            column ``open`` marks links timed to the window end (projects still
            waiting). Links must form a tree (each node has one parent).
        title: Plot title.
        node_colors: Colour for every node name.
        order: Node names in top-to-bottom order among siblings.
        legend: Optional ``{label: colour}`` legend entries.
        xlabel: X-axis label.
        figsize: Figure size in inches.
        credit: Optional attribution string.

    Returns:
        Matplotlib Figure.
    """
    links = stats.reset_index()
    links.columns = ["from", "to", *links.columns[2:]]
    assert links["to"].is_unique, "links must form a tree"
    row = links.set_index("to")
    rank = {n: i for i, n in enumerate(order or [])}
    children = {
        p: sorted(g["to"], key=lambda n: rank.get(n, len(rank)))
        for p, g in links.groupby("from", sort=False)
    }
    root = next(n for n in children if n not in row.index)
    total = int(links.loc[links["from"] == root, "projects"].sum())

    # X: cumulative mean months. Untimed links borrow the longest sibling step.
    x = {root: 0.0}
    stack = [root]
    while stack:
        p = stack.pop()
        kids = children.get(p, [])
        fallback = row.loc[kids, "mean"].max()
        for c in kids:
            step = row.at[c, "mean"]
            x[c] = x[p] + (step if pd.notna(step) else fallback if pd.notna(fallback) else 1.0)
            stack.append(c)
    xmax = max(x.values())

    # Y: leaves stacked top-down with gaps; parents centred over their children.
    # shortcut: tiny leaves get a minimum drawn height, so ribbon widths are not exact for n < ~0.6% of the cohort.
    min_h, gap = total * 0.006, total * 0.025

    def size(n: str) -> float:
        return sum(size(c) for c in children[n]) if n in children else max(row.at[n, "projects"], min_h)

    band: dict[str, tuple[float, float]] = {}
    cursor = [0.0]

    def place(n: str) -> tuple[float, float]:
        if n not in children:
            band[n] = (cursor[0], cursor[0] - size(n))
            cursor[0] -= size(n) + gap
            return band[n]
        spans = [place(c) for c in children[n]]
        mid = (spans[0][0] + spans[-1][1]) / 2
        band[n] = (mid + size(n) / 2, mid - size(n) / 2)
        return band[n]

    place(root)

    def fmt(r: pd.Series) -> str:
        if pd.isna(r["mean"]):
            return "not timed"
        prefix = "waiting " if r.get("open", False) else ""
        return f"{prefix}{r['mean']:.1f} ± {r['sd']:.1f} mo, median {r['median']:.1f}"

    timing = {n: fmt(r) for n, r in row.iterrows()}
    thin = total * 0.08  # ribbons narrower than this can't hold a label inside

    fig, ax = plt.subplots(figsize=figsize, facecolor=_BG)
    _style_ax(ax)
    w = xmax * 0.008
    t = np.linspace(0, 1, 60)
    s = t * t * (3 - 2 * t)
    halo = [pe.withStroke(linewidth=3, foreground=_BG)]

    for p, kids in children.items():
        src_top = band[p][0]
        for c in kids:
            h = size(c)
            a_top, a_bot = src_top, src_top - h
            b_top, b_bot = band[c]
            src_top -= h
            x0, x1 = x[p] + w / 2, x[c] - w / 2
            xs = x0 + (x1 - x0) * t
            ax.fill_between(xs, a_bot + (b_bot - a_bot) * s, a_top + (b_top - a_top) * s,
                            color=node_colors[c], alpha=0.4, linewidth=0, zorder=1)

            text = timing[c]
            if h >= thin:
                ax.text((x0 + x1) / 2, (a_top + a_bot + b_top + b_bot) / 4, text,
                        ha="center", va="center", fontsize=8.5, color=_TEXT,
                        path_effects=halo, zorder=4)
            elif c in children:  # thin ribbon into a mid stage: label above it
                ax.text((x0 + x1) / 2, (a_top + b_top) / 2 + gap * 0.1, text,
                        ha="center", va="bottom", fontsize=8.5, color=_TEXT,
                        path_effects=halo, zorder=4)
            # thin ribbons into leaves carry their timing on the leaf label instead

    for n, (top, bot) in band.items():
        ax.add_patch(mpatches.Rectangle((x[n] - w / 2, bot), w, top - bot,
                                        color=node_colors[n], linewidth=0, zorder=2))
        count = total if n == root else int(row.at[n, "projects"])
        label = f"{n}\n{count:,}"
        if n == root:
            ax.text(x[n] - w, (top + bot) / 2, label, ha="right", va="center",
                    fontsize=10, color=_TEXT, fontweight="bold", zorder=4)
        elif n in children:
            ax.text(x[n], top + gap * 0.15, label, ha="center", va="bottom",
                    fontsize=10, color=_TEXT, fontweight="bold", path_effects=halo, zorder=4)
        else:
            label = f"{n}  {count:,}"
            if row.at[n, "projects"] < thin:
                label += f"\n{timing[n]}"
            ax.text(x[n] + w, (top + bot) / 2, label, ha="left", va="center",
                    fontsize=9, color=_LABEL, path_effects=halo, zorder=4)

    ax.set_xlim(-xmax * 0.1, xmax * 1.22)
    ax.set_ylim(cursor[0] + gap * 0.5, gap * 1.2)
    ax.set_yticks([])
    ax.set_xlabel(xlabel, color=_MUTED, fontsize=10)
    ax.grid(axis="x", color="#ffffff", alpha=0.06, linewidth=0.5)
    for spine in ("left", "right", "top"):
        ax.spines[spine].set_visible(False)
    ax.set_title(title, fontsize=15, fontweight="bold", color=_TEXT, pad=14, loc="left")

    if legend:
        handles = [mpatches.Patch(color=c, label=l) for l, c in legend.items()]
        ax.legend(handles=handles, loc="lower right", fontsize=9, framealpha=0.4,
                  facecolor="#1d1810", edgecolor=_EDGE_DARK, labelcolor=_LABEL)

    if credit:
        fig.text(0.99, 0.01, credit, ha="right", va="bottom",
                 fontsize=7, color="#54483e", alpha=0.7)

    plt.tight_layout()
    return fig


def plot_alluvial(
    states: pd.DataFrame,
    order: list[str],
    colors: dict[str, str],
    title: str,
    origin: pd.Timestamp | None = None,
    figsize: tuple[int, int] = (18, 9),
    credit: str = "",
    label_first: bool = False,
) -> Figure:
    """Alluvial timeline: stacked state columns per snapshot, ribbons between.

    Args:
        states: One row per project, one column per snapshot date (ascending),
            values are state names, or NaN where the project does not exist yet.
            New arrivals fan in from the top of the previous column.
        order: State names, top to bottom.
        colors: Colour for every state.
        title: Plot title.
        origin: Date the first arrivals fan in from (defaults to one snapshot
            spacing before the first column).
        figsize: Figure size in inches.
        credit: Optional attribution string.
        label_first: Also label the first column's states on its left (for two-column before/after charts).

    Returns:
        Matplotlib Figure.
    """
    import matplotlib.dates as mdates

    dates = list(states.columns)
    xs = mdates.date2num(dates)
    x_origin = mdates.date2num(origin) if origin is not None else xs[0] - (xs[1] - xs[0])
    rank = {s: i for i, s in enumerate(order)}
    gap = states.notna().sum().max() * 0.03
    w = (xs[-1] - x_origin) * 0.006

    def layout(col: pd.Series) -> dict[str, tuple[float, float]]:
        """Top/bottom y for each present state, stacked top-down from 0."""
        counts = col.value_counts()
        out, y = {}, 0.0
        for s in order:
            if counts.get(s, 0):
                out[s] = (y, y - counts[s])
                y -= counts[s] + gap
        return out

    bands = [layout(states[d]) for d in dates]

    fig, ax = plt.subplots(figsize=figsize, facecolor=_BG)
    _style_ax(ax)
    t = np.linspace(0, 1, 50)
    s_curve = t * t * (3 - 2 * t)

    for j, d in enumerate(dates):
        prev = states[dates[j - 1]] if j else pd.Series(np.nan, index=states.index)
        pairs = pd.DataFrame({"a": prev, "b": states[d]}).dropna(subset=["b"])
        flows = pairs.fillna({"a": "__new"}).value_counts(["a", "b"])
        a_rank = lambda a: -1 if a == "__new" else rank[a]
        # Source ends ordered by target, target ends by source: fewest crossings.
        dst_cursor = {b: bands[j][b][0] for b in bands[j]}
        x0 = xs[j - 1] + w / 2 if j else x_origin
        x1 = xs[j] - w / 2
        src_cursor = {a: bands[j - 1][a][0] for a in bands[j - 1]} if j else {}
        seg_src = {}
        for (a, b), n in sorted(flows.items(), key=lambda kv: (a_rank(kv[0][0]), rank[kv[0][1]])):
            if a == "__new":
                seg_src[(a, b)] = (gap * 0.6, gap * 0.6)
            else:
                seg_src[(a, b)] = (src_cursor[a], src_cursor[a] - n)
                src_cursor[a] -= n
        for (a, b), n in sorted(flows.items(), key=lambda kv: (rank[kv[0][1]], a_rank(kv[0][0]))):
            a_top, a_bot = seg_src[(a, b)]
            b_top = dst_cursor[b]
            b_bot = b_top - n
            dst_cursor[b] = b_bot
            xx = x0 + (x1 - x0) * t
            ax.fill_between(xx, a_bot + (b_bot - a_bot) * s_curve, a_top + (b_top - a_top) * s_curve,
                            color=colors[b], alpha=0.45 if a == b else 0.75, linewidth=0, zorder=1)

        for s, (top, bot) in bands[j].items():
            ax.add_patch(mpatches.Rectangle((xs[j] - w / 2, bot), w, top - bot,
                                            color=colors[s], linewidth=0, zorder=2))
        new = int(prev.isna().sum() - states[d].isna().sum())
        header = f"{states[d].notna().sum():,}" if label_first else f"{states[d].notna().sum():,}\n+{new:,} new"
        ax.text(xs[j], gap * 0.9, header, ha="center", va="bottom", fontsize=8.5, color=_LABEL, zorder=4)

    halo = [pe.withStroke(linewidth=3, foreground=_BG)]
    last = states[dates[-1]].value_counts()
    for s, (top, bot) in bands[-1].items():
        ax.text(xs[-1] + w, (top + bot) / 2, f"{s}  {last[s]:,}", ha="left", va="center",
                fontsize=9.5, color=_TEXT, path_effects=halo, zorder=4)

    if label_first:
        first = states[dates[0]].value_counts()
        for s, (top, bot) in bands[0].items():
            ax.text(xs[0] - w, (top + bot) / 2, f"{first[s]:,}  {s}", ha="right", va="center",
                    fontsize=9.5, color=_TEXT, path_effects=halo, zorder=4)

    lowest = min(bot for b in bands for _, bot in b.values())
    left_pad = (xs[-1] - x_origin) * (0.17 if label_first else 0.0) + w * 2
    ax.set_xlim(x_origin - left_pad, xs[-1] + (xs[-1] - x_origin) * 0.17)
    ax.set_ylim(lowest - gap, gap * 4)
    ax.set_xticks(xs, [f"{d:%d %b %Y}" for d in dates], fontsize=8.5, color=_MUTED)
    ax.set_yticks([])
    ax.grid(axis="x", color="#ffffff", alpha=0.05, linewidth=0.5)
    for spine in ("left", "right", "top"):
        ax.spines[spine].set_visible(False)
    ax.set_title(title, fontsize=15, fontweight="bold", color=_TEXT, pad=14, loc="left")

    if credit:
        fig.text(0.99, 0.01, credit, ha="right", va="bottom",
                 fontsize=7, color="#54483e", alpha=0.7)

    plt.tight_layout()
    return fig
