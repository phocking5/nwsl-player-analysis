"""Download and flatten StatsBomb open data for the NWSL 2023 season.

Outputs (in data/processed/):
    matches.parquet         one row per match
    events.parquet          one row per on-ball event (passes, shots, duels, fouls, ...)
    player_minutes.parquet  minutes played per player per match, with position

Raw JSON is cached in data/raw/ so re-running is fast.

Usage:
    python src/load_data.py
"""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests

COMPETITION_ID = 49   # NWSL
SEASON_ID = 107       # 2023
BASE_URL = 'https://raw.githubusercontent.com/statsbomb/open-data/master/data'

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / 'data' / 'raw'
OUT_DIR = ROOT / 'data' / 'processed'

# Columns we keep from the (very wide) flattened event data
EVENT_COLS = [
    'id', 'match_id', 'index', 'period', 'minute', 'second', 'possession',
    'type_name', 'play_pattern_name', 'team_name', 'player_id', 'player_name',
    'position_name', 'location', 'under_pressure',
    # passes
    'pass_end_location', 'pass_length', 'pass_recipient_name', 'pass_outcome_name',
    'pass_type_name', 'pass_height_name', 'pass_cross', 'pass_through_ball',
    'pass_switch', 'pass_goal_assist', 'pass_shot_assist', 'pass_assisted_shot_id',
    # carries / dribbles
    'carry_end_location', 'dribble_outcome_name',
    # shots
    'shot_statsbomb_xg', 'shot_outcome_name', 'shot_type_name', 'shot_body_part_name',
    'shot_end_location', 'shot_key_pass_id',
    # defending
    'duel_type_name', 'duel_outcome_name', 'interception_outcome_name',
    'ball_recovery_recovery_failure', 'counterpress',
    # fouls / cards
    'foul_committed_card_name', 'bad_behaviour_card_name',
]


def fetch_json(rel_path: str) -> list | dict:
    """Return JSON from the StatsBomb repo, caching it under data/raw/."""
    cache = RAW_DIR / rel_path
    if cache.exists():
        return json.loads(cache.read_text())
    for attempt in range(4):  # GitHub raw occasionally times out; retry a few times
        try:
            resp = requests.get(f'{BASE_URL}/{rel_path}', timeout=60)
            resp.raise_for_status()
            break
        except requests.RequestException:
            if attempt == 3:
                raise
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(resp.text)
    return resp.json()


def load_matches() -> pd.DataFrame:
    matches = pd.json_normalize(fetch_json(f'matches/{COMPETITION_ID}/{SEASON_ID}.json'), sep='_')
    keep = ['match_id', 'match_date', 'home_team_home_team_name', 'away_team_away_team_name',
            'home_score', 'away_score', 'competition_stage_name']
    matches = matches[keep].rename(columns={
        'home_team_home_team_name': 'home_team',
        'away_team_away_team_name': 'away_team',
        'competition_stage_name': 'stage',
    })
    return matches.sort_values('match_date').reset_index(drop=True)


def load_events(match_id: int) -> pd.DataFrame:
    events = pd.json_normalize(fetch_json(f'events/{match_id}.json'), sep='_')
    events['match_id'] = match_id
    for col in EVENT_COLS:
        if col not in events.columns:
            events[col] = pd.NA
    return events[EVENT_COLS]


def _clock_to_minutes(clock: str | None) -> float | None:
    if clock is None:
        return None
    mins, secs = clock.split(':')
    return int(mins) + int(secs) / 60


def load_minutes(match_id: int, match_end: float) -> pd.DataFrame:
    """Minutes per player per position spell, from the lineup file.

    StatsBomb lineups list each position a player held with a from/to match clock.
    A missing 'to' means the player was on the pitch at the final whistle.
    """
    rows = []
    for team in fetch_json(f'lineups/{match_id}.json'):
        for player in team['lineup']:
            for spell in player['positions']:
                start = _clock_to_minutes(spell['from'])
                end = _clock_to_minutes(spell['to']) or match_end
                rows.append({
                    'match_id': match_id,
                    'team_name': team['team_name'],
                    'player_id': player['player_id'],
                    'player_name': player['player_name'],
                    'player_nickname': player.get('player_nickname'),
                    'position_name': spell['position'],
                    'minutes': max(end - start, 0),
                })
    return pd.DataFrame(rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    matches = load_matches()
    match_ids = matches['match_id'].tolist()
    print(f'{len(match_ids)} matches')

    # Download in parallel; files are cached after the first run
    with ThreadPoolExecutor(max_workers=8) as pool:
        events_list = list(pool.map(load_events, match_ids))
    events = pd.concat(events_list, ignore_index=True)
    print(f'{len(events):,} events')

    # Match end = last event clock in each match (includes stoppage time)
    clock = events['minute'] + events['second'] / 60
    match_end = clock.groupby(events['match_id']).max()
    with ThreadPoolExecutor(max_workers=8) as pool:
        minutes_list = list(pool.map(lambda m: load_minutes(m, match_end[m]), match_ids))
    minutes = pd.concat(minutes_list, ignore_index=True)

    # Split [x, y] lists into numeric columns for easier filtering
    for col, prefix in [('location', ''), ('pass_end_location', 'pass_end_'),
                        ('carry_end_location', 'carry_end_')]:
        xy = events[col].apply(lambda v: v if isinstance(v, list) else [None, None])
        events[f'{prefix}x'] = xy.str[0].astype(float)
        events[f'{prefix}y'] = xy.str[1].astype(float)
    events = events.drop(columns=['location', 'pass_end_location', 'carry_end_location', 'shot_end_location'])

    matches.to_parquet(OUT_DIR / 'matches.parquet', index=False)
    events.to_parquet(OUT_DIR / 'events.parquet', index=False)
    minutes.to_parquet(OUT_DIR / 'player_minutes.parquet', index=False)
    print(f'Saved to {OUT_DIR}')


if __name__ == '__main__':
    main()
