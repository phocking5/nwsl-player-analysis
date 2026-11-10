"""Shared chart styling and helpers."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from adjustText import adjust_text

# Colors (validated categorical palette, light mode)
SURFACE = '#fcfcfb'
TEXT = '#0b0b0b'
TEXT_2 = '#52514e'
MUTED = '#c3c2b7'
GRID = '#e8e7e2'
MID = '#2a78d6'      # midfielders: blue
FWD = '#eb6834'      # forwards: orange
CATEGORY_COLORS = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100']  # pizza-slice categories
GROUP_COLORS = {'Midfielder': MID, 'Forward': FWD}
# six goals-added action types, fixed order (palette slots 1-6)
ACTION_COLORS = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300']


def pizza(df: pd.DataFrame, row: pd.Series, categories: dict, labels: dict, title: str,
          subtitle_text: str, path=None):
    """Percentile pizza chart for one player vs the rest of `df`.

    categories: {category name: (weight, [metric cols])}; labels: {metric col: display label}
    """
    from mplsoccer import PyPizza

    cols, colors = [], []
    for i, (_, (_, metrics)) in enumerate(categories.items()):
        cols += metrics
        colors += [CATEGORY_COLORS[i]] * len(metrics)
    values = [round((df[c] <= row[c]).mean() * 100) for c in cols]
    baker = PyPizza(params=[labels[c] for c in cols], background_color=SURFACE,
                    straight_line_color=GRID, straight_line_lw=1, last_circle_lw=0,
                    other_circle_lw=0, inner_circle_size=18)
    fig, ax = baker.make_pizza(
        values, figsize=(6.5, 7), color_blank_space='same', blank_alpha=0.15, slice_colors=colors,
        value_colors=['#ffffff'] * len(cols), value_bck_colors=colors, param_location=120,
        kwargs_slices=dict(edgecolor=SURFACE, zorder=2, linewidth=2),
        kwargs_params=dict(color=TEXT, fontsize=9),
        kwargs_values=dict(color='#ffffff', fontsize=9, zorder=3,
                           bbox=dict(edgecolor=SURFACE, boxstyle='round,pad=0.25', lw=1)))
    fig.subplots_adjust(top=0.84, bottom=0.10)  # headroom so outer labels clear the title block
    fig.text(0.05, 0.97, title, fontsize=14, fontweight='bold', color=TEXT)
    fig.text(0.05, 0.94, subtitle_text, fontsize=9, color=TEXT_2)
    handles = [plt.Rectangle((0, 0), 1, 1, color=CATEGORY_COLORS[i]) for i in range(len(categories))]
    fig.legend(handles, list(categories), loc='lower center', ncol=4, fontsize=9, frameon=False)
    if path:
        fig.savefig(path)
    return fig


def set_style() -> None:
    plt.rcParams.update({
        'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
        'axes.edgecolor': MUTED, 'axes.labelcolor': TEXT_2, 'axes.titlecolor': TEXT,
        'axes.titlesize': 13, 'axes.titleweight': 'bold', 'axes.titlelocation': 'left',
        'axes.titlepad': 22,  # leaves room for the subtitle line
        'axes.labelsize': 10, 'axes.spines.top': False, 'axes.spines.right': False,
        'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.8, 'axes.axisbelow': True,
        'xtick.color': TEXT_2, 'ytick.color': TEXT_2, 'xtick.labelsize': 9, 'ytick.labelsize': 9,
        'legend.frameon': False, 'legend.fontsize': 9, 'font.size': 10,
        'figure.dpi': 110, 'savefig.dpi': 160, 'savefig.bbox': 'tight',
    })


def subtitle(ax, text: str) -> None:
    """Grey one-line subtitle under the left-aligned title."""
    ax.text(0, 1.02, text, transform=ax.transAxes, color=TEXT_2, fontsize=9, va='bottom')


def labeled_scatter(ax, df: pd.DataFrame, x: str, y: str, label_mask: pd.Series,
                    color: str, label_col: str = 'display_name', size: float = 40) -> None:
    """Scatter all players in grey-ish context, highlight + label a subset."""
    rest = df[~label_mask]
    hi = df[label_mask]
    ax.scatter(rest[x], rest[y], s=size, color=color, alpha=0.30, edgecolor=SURFACE, linewidth=1)
    ax.scatter(hi[x], hi[y], s=size * 1.3, color=color, edgecolor=SURFACE, linewidth=1.5, zorder=3)
    # Players sitting on (almost) the same point share one label, e.g. "A / B"
    x_span = (df[x].max() - df[x].min()) or 1
    y_span = (df[y].max() - df[y].min()) or 1
    keys = list(zip((hi[x] / x_span * 100).round(), (hi[y] / y_span * 100).round()))
    groups = hi.assign(_key=keys).groupby('_key', sort=False)
    texts = [ax.text(g[x].mean(), g[y].mean(), ' / '.join(g[label_col]), fontsize=8, color=TEXT)
             for _, g in groups]
    adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle='-', color=MUTED, lw=0.6))


def mean_lines(ax, df: pd.DataFrame, x: str, y: str) -> None:
    ax.axvline(df[x].mean(), color=MUTED, lw=1, ls='--', zorder=1)
    ax.axhline(df[y].mean(), color=MUTED, lw=1, ls='--', zorder=1)


def leaderboard(ax, df: pd.DataFrame, value: str, color: str, n: int = 15,
                actual: str | None = None, label_col: str = 'display_name') -> None:
    """Horizontal bar leaderboard; optional dot for an 'actual' comparison metric."""
    top = df.nlargest(n, value).iloc[::-1]
    ypos = np.arange(len(top))
    ax.barh(ypos, top[value], color=color, height=0.65, edgecolor=SURFACE, linewidth=2)
    if actual:
        ax.scatter(top[actual], ypos, color=TEXT, s=22, zorder=3, label=actual)
    ax.set_yticks(ypos, [f'{r[label_col]}  ·  {r["team_short"]}' for _, r in top.iterrows()])
    ax.grid(axis='y', visible=False)
    ax.tick_params(axis='y', length=0)


TEAM_SHORT = {
    'Angel City': 'LA', 'Chicago Red Stars': 'CHI', 'Houston Dash': 'HOU', 'KC Current': 'KC',
    'NJ/NY Gotham FC': 'NJY', 'North Carolina Courage': 'NC', 'OL Reign': 'SEA',
    'Seattle Reign': 'SEA', 'Orlando Pride': 'ORL', 'Portland Thorns': 'POR',
    'Racing Louisville FC': 'LOU', 'San Diego Wave': 'SD', 'Washington Spirit': 'WAS',
}
