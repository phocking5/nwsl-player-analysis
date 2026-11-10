"""Download NWSL player-season data from the American Soccer Analysis (ASA) API.

ASA publishes free NWSL models built on full event data:
    xgoals      shots, goals, xG, key passes, assists, xA
    xpass       passing volume, completion vs expected completion, directness
    goals-added g+: the value every action adds to a team's chance of scoring or conceding,
                split into six action types (Dribbling, Fouling, Interrupting, Passing,
                Receiving, Shooting) with a count of actions for each

Output: data/processed/asa_players_<season>.parquet, one row per player
(regular season only, so playoff teams don't get extra minutes).

Usage:
    python src/load_asa.py            # 2025
    python src/load_asa.py 2024
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd
import requests

BASE_URL = 'https://app.americansocceranalysis.com/api/v1/nwsl'
ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / 'data' / 'raw' / 'asa'
OUT_DIR = ROOT / 'data' / 'processed'
STAGE = 'Regular Season'


def fetch(endpoint: str, **params) -> list[dict]:
    """GET an ASA endpoint, caching the JSON response under data/raw/asa/."""
    key = endpoint.replace('/', '_') + ''.join(f'__{k}-{v}' for k, v in sorted(params.items()))
    cache = RAW_DIR / f'{key.replace(" ", "_")}.json'
    if cache.exists():
        return json.loads(cache.read_text())
    for attempt in range(4):  # the API is occasionally slow; retry a few times
        try:
            resp = requests.get(f'{BASE_URL}/{endpoint}', params=params, timeout=120)
            resp.raise_for_status()
            break
        except requests.RequestException:
            if attempt == 3:
                raise
            time.sleep(5)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(resp.text)
    return resp.json()


def load_season(season: str) -> pd.DataFrame:
    common = dict(season_name=season, stage_name=STAGE)

    # team_id is a list for traded players; the main team is resolved below instead
    xg = pd.DataFrame(fetch('players/xgoals', **common)).drop(columns=['team_id'])
    pens = pd.DataFrame(fetch('players/xgoals', shot_pattern='Penalty', **common))
    pens = pens.set_index('player_id')[['shots', 'goals', 'xgoals']].add_prefix('pen_')
    xg = xg.join(pens, on='player_id').fillna({'pen_shots': 0, 'pen_goals': 0, 'pen_xgoals': 0})
    xg['np_shots'] = xg['shots'] - xg['pen_shots']
    xg['np_goals'] = xg['goals'] - xg['pen_goals']
    xg['np_xg'] = xg['xgoals'] - xg['pen_xgoals']

    xpass = pd.DataFrame(fetch('players/xpass', **common)).drop(
        columns=['team_id', 'general_position', 'minutes_played'])

    # goals added: explode the nested per-action-type list into wide columns
    ga_rows = []
    for rec in fetch('players/goals-added', **common):
        row = {'player_id': rec['player_id']}
        for action in rec['data']:
            name = action['action_type'].lower()
            row[f'ga_{name}'] = action['goals_added_above_avg']
            row[f'ga_raw_{name}'] = action['goals_added_raw']
            row[f'n_{name}'] = action['count_actions']
        ga_rows.append(row)
    ga = pd.DataFrame(ga_rows)

    # Main team = the team a player logged the most minutes for (handles mid-season trades)
    by_team = pd.DataFrame(fetch('players/xgoals', split_by_teams='true', **common))
    main_team = (by_team.sort_values('minutes_played')
                 .drop_duplicates('player_id', keep='last')
                 .set_index('player_id')['team_id'].rename('main_team_id'))

    players = pd.DataFrame(fetch('players'))[['player_id', 'player_name', 'birth_date', 'nationality']]
    teams = pd.DataFrame(fetch('teams'))[['team_id', 'team_name', 'team_abbreviation']]

    df = (xg.join(main_team, on='player_id')
            .merge(xpass, on='player_id', how='left')
            .merge(ga, on='player_id', how='left')
            .merge(players, on='player_id', how='left')
            .merge(teams, left_on='main_team_id', right_on='team_id', how='left'))
    df['season'] = season
    return df.drop(columns=['main_team_id'])


def main() -> None:
    season = sys.argv[1] if len(sys.argv) > 1 else '2025'
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_season(season)
    games = fetch('games', season_name=season)
    regular = sum(not g['knockout_game'] for g in games)
    out = OUT_DIR / f'asa_players_{season}.parquet'
    df.to_parquet(out, index=False)
    print(f'NWSL {season}: {regular} regular-season games, {len(df)} players, '
          f'{df["goals"].sum():.0f} goals ({df["pen_goals"].sum():.0f} pens), {df["xgoals"].sum():.1f} xG')
    print(f'Saved {out}')


if __name__ == '__main__':
    main()
