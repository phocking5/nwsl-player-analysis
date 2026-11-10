"""Per-90 player metrics from American Soccer Analysis season tables."""
from __future__ import annotations

import numpy as np
import pandas as pd

POSITION_GROUPS = {
    'DM': 'Midfielder', 'CM': 'Midfielder', 'AM': 'Midfielder',
    'W': 'Forward', 'ST': 'Forward',
    'FB': 'Defender', 'CB': 'Defender', 'GK': 'Goalkeeper',
}
POSITION_NAMES = {'DM': 'Defensive mid', 'CM': 'Central mid', 'AM': 'Attacking mid',
                  'W': 'Winger', 'ST': 'Striker', 'FB': 'Fullback', 'CB': 'Center back', 'GK': 'Goalkeeper'}

# goals added (g+) action types, in a fixed order used for colors everywhere
GA_ACTIONS = ['passing', 'receiving', 'dribbling', 'shooting', 'interrupting', 'fouling']
GA_LABELS = {a: a.capitalize() for a in GA_ACTIONS}

# season totals that become per-90 rates
TOTAL_COLS = {
    'np_shots': 'shots', 'shots_on_target': 'shots_on_target', 'np_goals': 'np_goals',
    'np_xg': 'np_xg', 'key_passes': 'key_passes', 'primary_assists': 'assists',
    'xassists': 'xa', 'attempted_passes': 'passes',
    **{f'ga_{a}': f'ga_{a}' for a in GA_ACTIONS},
    **{f'n_{a}': f'n_{a}' for a in GA_ACTIONS},
}


def per90_table(raw: pd.DataFrame, min_minutes: int = 900) -> pd.DataFrame:
    """Filter to qualified players and add per-90 rates and ratio stats."""
    df = raw[raw['minutes_played'] >= min_minutes].copy()
    df = df.rename(columns={'minutes_played': 'minutes', 'player_name': 'display_name'})
    df['position_group'] = df['general_position'].map(POSITION_GROUPS)
    df['position'] = df['general_position'].map(POSITION_NAMES)
    df['team_short'] = df['team_abbreviation']

    # A few players have no goals-added row; treat as zero rather than drop them
    ga_cols = [c for c in df.columns if c.startswith(('ga_', 'n_'))]
    df[ga_cols] = df[ga_cols].fillna(0)

    for src, name in TOTAL_COLS.items():
        df[f'{name}_p90'] = df[src] / df['minutes'] * 90
    df['ga_total'] = df[[f'ga_{a}' for a in GA_ACTIONS]].sum(axis=1)
    df['ga_total_p90'] = df['ga_total'] / df['minutes'] * 90
    df['g_plus_a_p90'] = (df['np_goals'] + df['primary_assists']) / df['minutes'] * 90
    df['npxg_plus_xa_p90'] = (df['np_xg'] + df['xassists']) / df['minutes'] * 90
    df['xg_per_shot'] = df['np_xg'] / df['np_shots'].replace(0, np.nan)
    df['goals_minus_xg'] = df['np_goals'] - df['np_xg']
    df['pass_pct'] = df['pass_completion_percentage'] * 100
    df['pass_over_exp'] = df['passes_completed_over_expected_p100']
    df['vertical_yds'] = df['avg_vertical_distance_yds']
    df['touch_share'] = df['share_team_touches'] * 100
    return df.reset_index(drop=True)


def composite(df: pd.DataFrame, categories: dict) -> pd.DataFrame:
    """Weighted average of within-group percentile categories.

    categories: {name: (weight, [metric columns])}
    """
    out = df.copy()
    out['score'] = 0.0
    for cat, (weight, cols) in categories.items():
        out[cat] = out[cols].rank(pct=True).mean(axis=1) * 100
        out['score'] += weight * out[cat]
    out['composite_rank'] = out['score'].rank(ascending=False, method='min').astype(int)
    return out


def variance_share(df: pd.DataFrame, parts: list[str], total: str) -> pd.Series:
    """Share of the spread in `total` contributed by each additive part.

    Because total = sum(parts), var(total) = sum(cov(part, total)), so
    cov(part, total) / var(total) splits the variance exactly and sums to 1.
    """
    var_total = df[total].var()
    return pd.Series({p: df[p].cov(df[total]) / var_total for p in parts})


def separation(df: pd.DataFrame, metrics: dict[str, str], n_top: int = 10,
               rank_col: str = 'ga_total_p90') -> pd.DataFrame:
    """Standardized difference between the top-n players (by rank_col) and the rest."""
    ordered = df.sort_values(rank_col, ascending=False)
    top, rest = ordered.head(n_top), ordered.iloc[n_top:]
    rows = []
    for col, label in metrics.items():
        pooled = np.sqrt((top[col].var() + rest[col].var()) / 2)
        rows.append({'metric': label, 'top10': top[col].mean(), 'rest': rest[col].mean(),
                     'effect_size': (top[col].mean() - rest[col].mean()) / pooled})
    return pd.DataFrame(rows).sort_values('effect_size')
