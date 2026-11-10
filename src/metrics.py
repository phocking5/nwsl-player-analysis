"""Player-season metrics built from StatsBomb event data.

StatsBomb coordinates: pitch is 120 x 80 yards and every event is recorded
from the acting team's perspective, attacking toward x = 120.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

PITCH_X, PITCH_Y = 120, 80
GOAL = np.array([120, 40])

MIDFIELD_POSITIONS = {
    'Left Defensive Midfield', 'Right Defensive Midfield', 'Center Defensive Midfield',
    'Left Center Midfield', 'Right Center Midfield', 'Center Midfield',
    'Center Attacking Midfield', 'Left Attacking Midfield', 'Right Attacking Midfield',
    'Left Midfield', 'Right Midfield',
}
FORWARD_POSITIONS = {
    'Center Forward', 'Left Center Forward', 'Right Center Forward',
    'Left Wing', 'Right Wing', 'Secondary Striker',
}
SET_PIECE_PASSES = {'Corner', 'Free Kick', 'Throw-in', 'Goal Kick', 'Kick Off'}
WON_OUTCOMES = {'Won', 'Success', 'Success In Play', 'Success Out'}

# Players commonly known by a single name (Brazilian convention)
DISPLAY_NAME_OVERRIDES = {
    'Kerolin Ferraz': 'Kerolin',
    'Adriana Silva': 'Adriana',
}


def position_group(position: str) -> str:
    if position in MIDFIELD_POSITIONS:
        return 'Midfielder'
    if position in FORWARD_POSITIONS:
        return 'Forward'
    if position == 'Goalkeeper':
        return 'Goalkeeper'
    return 'Defender'


# --------------------------------------------------------------------------------------
# Minutes and positions
# --------------------------------------------------------------------------------------
def player_seasons(minutes: pd.DataFrame) -> pd.DataFrame:
    """One row per player: total minutes, main team, primary position and group."""
    minutes = minutes.assign(position_group=minutes['position_name'].map(position_group))
    total = minutes.groupby('player_id')['minutes'].sum().rename('minutes')

    def top(col: str) -> pd.Series:
        by = minutes.groupby(['player_id', col])['minutes'].sum().reset_index()
        return by.sort_values('minutes').drop_duplicates('player_id', keep='last').set_index('player_id')[col]

    names = minutes.drop_duplicates('player_id').set_index('player_id')
    # Prefer StatsBomb's nickname; otherwise shorten long legal names to first + last
    short = names['player_name'].str.split().apply(lambda p: f'{p[0]} {p[-1]}' if len(p) > 2 else ' '.join(p))
    display = names['player_nickname'].fillna(short).replace(DISPLAY_NAME_OVERRIDES)
    return pd.DataFrame({
        'player_name': names['player_name'],
        'display_name': display,
        'team': top('team_name'),
        'primary_position': top('position_name'),
        'position_group': top('position_group'),
        'matches': minutes.groupby('player_id')['match_id'].nunique(),
        'minutes': total,
    }).reset_index()


# --------------------------------------------------------------------------------------
# Expected threat (xT), after Karun Singh (2018)
# --------------------------------------------------------------------------------------
def _zone(x: pd.Series, y: pd.Series, nx: int, ny: int) -> tuple[np.ndarray, np.ndarray]:
    zx = np.clip((x / PITCH_X * nx).astype(int), 0, nx - 1)
    zy = np.clip((y / PITCH_Y * ny).astype(int), 0, ny - 1)
    return zx.to_numpy(), zy.to_numpy()


def fit_xt(events: pd.DataFrame, nx: int = 16, ny: int = 12, n_iter: int = 10) -> np.ndarray:
    """Fit an xT grid: the probability that possession of the ball in a zone
    leads to a goal within the next few actions.

    xT(z) = P(shoot|z) * P(goal|shot,z) + P(move|z) * sum_z' T(z -> z') * xT(z')
    """
    shots = events[(events['type_name'] == 'Shot') & (events['shot_type_name'] != 'Penalty')]
    passes = events[(events['type_name'] == 'Pass') & events['pass_outcome_name'].isna()
                    & ~events['pass_type_name'].isin(SET_PIECE_PASSES)]
    carries = events[events['type_name'] == 'Carry']

    moves = pd.DataFrame({
        'x': pd.concat([passes['x'], carries['x']]),
        'y': pd.concat([passes['y'], carries['y']]),
        'end_x': pd.concat([passes['pass_end_x'], carries['carry_end_x']]),
        'end_y': pd.concat([passes['pass_end_y'], carries['carry_end_y']]),
    }).dropna()

    shot_count = np.zeros((ny, nx))
    goal_count = np.zeros((ny, nx))
    move_count = np.zeros((ny, nx))
    sx, sy = _zone(shots['x'], shots['y'], nx, ny)
    np.add.at(shot_count, (sy, sx), 1)
    is_goal = (shots['shot_outcome_name'] == 'Goal').to_numpy()
    np.add.at(goal_count, (sy[is_goal], sx[is_goal]), 1)
    mx, my = _zone(moves['x'], moves['y'], nx, ny)
    ex, ey = _zone(moves['end_x'], moves['end_y'], nx, ny)
    np.add.at(move_count, (my, mx), 1)

    # Transition matrix between zones, from successful moves
    n_zones = nx * ny
    start_idx = my * nx + mx
    end_idx = ey * nx + ex
    transitions = np.zeros((n_zones, n_zones))
    np.add.at(transitions, (start_idx, end_idx), 1)
    transitions /= np.maximum(transitions.sum(axis=1, keepdims=True), 1)

    total = shot_count + move_count
    p_shoot = np.divide(shot_count, total, out=np.zeros_like(total), where=total > 0)
    p_move = np.divide(move_count, total, out=np.zeros_like(total), where=total > 0)
    p_goal = np.divide(goal_count, shot_count, out=np.zeros_like(total), where=shot_count > 0)

    xt = np.zeros((ny, nx))
    for _ in range(n_iter):
        future = (transitions @ xt.reshape(-1)).reshape(ny, nx)
        xt = p_shoot * p_goal + p_move * future
    return xt


def add_xt(events: pd.DataFrame, xt: np.ndarray) -> pd.DataFrame:
    """Add an `xt_added` column: xT(end zone) - xT(start zone) for successful
    open-play passes and carries. Other events get 0."""
    ny, nx = xt.shape
    events = events.copy()
    is_pass = (events['type_name'] == 'Pass') & events['pass_outcome_name'].isna() \
        & ~events['pass_type_name'].isin(SET_PIECE_PASSES)
    is_carry = events['type_name'] == 'Carry'
    end_x = np.where(is_pass, events['pass_end_x'], events['carry_end_x'])
    end_y = np.where(is_pass, events['pass_end_y'], events['carry_end_y'])
    valid = (is_pass | is_carry) & events['x'].notna() & pd.notna(end_x)

    sx, sy = _zone(events.loc[valid, 'x'], events.loc[valid, 'y'], nx, ny)
    ex, ey = _zone(pd.Series(end_x[valid]), pd.Series(end_y[valid]), nx, ny)
    events['xt_added'] = 0.0
    events.loc[valid, 'xt_added'] = xt[ey, ex] - xt[sy, sx]
    return events


# --------------------------------------------------------------------------------------
# Per-player counting stats
# --------------------------------------------------------------------------------------
def _dist_to_goal(x, y) -> np.ndarray:
    return np.sqrt((GOAL[0] - x) ** 2 + (GOAL[1] - y) ** 2)


def _in_box(x, y) -> pd.Series:
    return (x >= 102) & (y >= 18) & (y <= 62)


def player_totals(events: pd.DataFrame) -> pd.DataFrame:
    """Season totals of on-ball actions for every player."""
    e = events[events['player_id'].notna()].copy()
    t = e['type_name']

    passes = t == 'Pass'
    complete = passes & e['pass_outcome_name'].isna()
    open_play = ~e['pass_type_name'].isin(SET_PIECE_PASSES)
    shots = t == 'Shot'
    penalty = e['shot_type_name'] == 'Penalty'
    goal = shots & (e['shot_outcome_name'] == 'Goal')
    carries = t == 'Carry'

    # Progressive: moves the ball >= 25% closer to goal and ends in the attacking 60%
    start_d = _dist_to_goal(e['x'], e['y'])
    prog_pass = complete & open_play & (_dist_to_goal(e['pass_end_x'], e['pass_end_y']) <= 0.75 * start_d) \
        & (e['pass_end_x'] >= 48)
    carry_len = np.hypot(e['carry_end_x'] - e['x'], e['carry_end_y'] - e['y'])
    prog_carry = carries & (carry_len >= 5) \
        & (_dist_to_goal(e['carry_end_x'], e['carry_end_y']) <= 0.75 * start_d) & (e['carry_end_x'] >= 48)

    # xA: the xG of the shot each pass assisted
    shot_xg = e.loc[shots, ['id', 'shot_statsbomb_xg']].set_index('id')['shot_statsbomb_xg']
    e['xa'] = e['pass_assisted_shot_id'].map(shot_xg).fillna(0.0)

    tackle = (t == 'Duel') & (e['duel_type_name'] == 'Tackle')
    flags = pd.DataFrame({
        'player_id': e['player_id'],
        # Shooting
        'shots': shots & ~penalty,
        'goals': goal,
        'np_goals': goal & ~penalty,
        'np_xg': np.where(shots & ~penalty, e['shot_statsbomb_xg'], 0.0),
        'shots_on_target': shots & ~penalty & e['shot_outcome_name'].isin(['Goal', 'Saved', 'Saved To Post']),
        # Creation
        'assists': passes & (e['pass_goal_assist'] == True),
        'xa': e['xa'],
        'key_passes': passes & ((e['pass_shot_assist'] == True) | (e['pass_goal_assist'] == True)),
        'through_balls': complete & (e['pass_through_ball'] == True),
        'passes_into_box': complete & open_play & _in_box(e['pass_end_x'], e['pass_end_y']) & ~_in_box(e['x'], e['y']),
        'crosses': passes & (e['pass_cross'] == True),
        # Passing / progression
        'passes': passes,
        'passes_completed': complete,
        'progressive_passes': prog_pass,
        'progressive_carries': prog_carry,
        'carries_into_box': carries & _in_box(e['carry_end_x'], e['carry_end_y']) & ~_in_box(e['x'], e['y']),
        'xt_pass': np.where(passes, e['xt_added'], 0.0),
        'xt_carry': np.where(carries, e['xt_added'], 0.0),
        'dribbles': t == 'Dribble',
        'dribbles_completed': (t == 'Dribble') & (e['dribble_outcome_name'] == 'Complete'),
        # Defending
        'tackles': tackle,
        'tackles_won': tackle & e['duel_outcome_name'].isin(WON_OUTCOMES),
        'interceptions': (t == 'Interception') & e['interception_outcome_name'].isin(WON_OUTCOMES),
        'pressures': t == 'Pressure',
        'counterpressures': (t == 'Pressure') & (e['counterpress'] == True),
        'ball_recoveries': (t == 'Ball Recovery') & (e['ball_recovery_recovery_failure'] != True),
        'blocks': t == 'Block',
        # Discipline / ball security
        'fouls_committed': t == 'Foul Committed',
        'fouls_won': t == 'Foul Won',
        'turnovers': t.isin(['Dispossessed', 'Miscontrol']),
    })
    totals = flags.groupby('player_id').sum(numeric_only=True)
    totals['xt'] = totals['xt_pass'] + totals['xt_carry']
    return totals.reset_index()


COUNT_COLS = [
    'shots', 'goals', 'np_goals', 'np_xg', 'shots_on_target', 'assists', 'xa', 'key_passes',
    'through_balls', 'passes_into_box', 'crosses', 'passes', 'passes_completed',
    'progressive_passes', 'progressive_carries', 'carries_into_box', 'xt', 'xt_pass', 'xt_carry',
    'dribbles', 'dribbles_completed', 'tackles', 'tackles_won', 'interceptions', 'pressures',
    'counterpressures', 'ball_recoveries', 'blocks', 'fouls_committed', 'fouls_won', 'turnovers',
]


def per90_table(seasons: pd.DataFrame, totals: pd.DataFrame, min_minutes: int = 900) -> pd.DataFrame:
    """Join minutes to totals, add per-90 rates and ratio stats, filter by minutes."""
    df = seasons.merge(totals, on='player_id', how='left').fillna({c: 0 for c in COUNT_COLS})
    df = df[df['minutes'] >= min_minutes].copy()
    for col in COUNT_COLS:
        df[f'{col}_p90'] = df[col] / df['minutes'] * 90
    df['g_plus_a_p90'] = (df['np_goals'] + df['assists']) / df['minutes'] * 90
    df['npxg_plus_xa_p90'] = (df['np_xg'] + df['xa']) / df['minutes'] * 90
    df['tackles_ints_p90'] = (df['tackles_won'] + df['interceptions']) / df['minutes'] * 90
    df['pass_pct'] = df['passes_completed'] / df['passes'].replace(0, np.nan) * 100
    df['dribble_pct'] = df['dribbles_completed'] / df['dribbles'].replace(0, np.nan) * 100
    df['xg_per_shot'] = df['np_xg'] / df['shots'].replace(0, np.nan)
    df['goals_minus_xg'] = df['np_goals'] - df['np_xg']
    return df.reset_index(drop=True)


def percentile_ranks(df: pd.DataFrame, cols: list[str], group_col: str = 'position_group') -> pd.DataFrame:
    """Percentile rank (0-100) of each metric within the player's position group."""
    out = df.copy()
    for col in cols:
        out[f'{col}_pct'] = df.groupby(group_col)[col].rank(pct=True) * 100
    return out
