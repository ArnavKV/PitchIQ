"""PitchIQ: build independently attributed player-season datasets.

Sources: dcaribou/transfermarkt-datasets and vibedatascience/understat_players_aggregated.
This does NOT join identities across providers or overwrite the existing player data.
"""
from pathlib import Path
from urllib.request import Request, urlopen
import ssl
import certifi
import shutil
import pandas as pd

ROOT = Path(__file__).resolve().parent
SOURCES = ROOT / 'data' / 'player_sources'
OUTPUT = ROOT / 'data' / 'player_comparison'
SOURCES.mkdir(parents=True, exist_ok=True)
OUTPUT.mkdir(parents=True, exist_ok=True)

URLS = {
    'appearances.csv.gz': 'https://pub-e682421888d945d684bcae8890b0ec20.r2.dev/data/appearances.csv.gz',
    'games.csv.gz': 'https://pub-e682421888d945d684bcae8890b0ec20.r2.dev/data/games.csv.gz',
    'understat.csv': 'https://raw.githubusercontent.com/vibedatascience/understat_players_aggregated/main/understat_players_aggregated_2014_td.csv',
}


def download(name, url):
    destination = SOURCES / name
    if destination.exists() and destination.stat().st_size > 100:
        print(f'Reusing {name}')
        return destination
    temporary = destination.with_suffix(destination.suffix + '.partial')
    print(f'Downloading {name} ...')
    try:
        request = Request(url, headers={'User-Agent': 'PitchIQ/1.0 (educational analytics)'})
        context = ssl.create_default_context(cafile=certifi.where())
        with urlopen(request, timeout=90, context=context) as response, open(temporary, 'wb') as out:
            shutil.copyfileobj(response, out)
        temporary.replace(destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return destination


def require_columns(frame, columns, source):
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f'{source} missing columns: {missing}; found: {list(frame.columns)}')


def transfermarkt():
    appearances = pd.read_csv(download('appearances.csv.gz', URLS['appearances.csv.gz']), low_memory=False)
    games = pd.read_csv(download('games.csv.gz', URLS['games.csv.gz']), low_memory=False)
    require_columns(appearances, ['game_id', 'player_id', 'player_name', 'player_club_id', 'goals', 'assists', 'minutes_played', 'date', 'competition_id'], 'appearances')
    require_columns(games, ['game_id', 'season'], 'games')

    # Source appearances are one row per player in a game. Use the game season
    # instead of deriving it from calendar dates (which mislabels summer games).
    appearance_cols = ['game_id', 'player_id', 'player_name', 'player_club_id', 'goals', 'assists', 'minutes_played', 'competition_id']
    a = appearances[appearance_cols].copy()
    a = a.dropna(subset=['game_id', 'player_id'])
    a = a.drop_duplicates(subset=['game_id', 'player_id'], keep='first')
    game_cols = ['game_id', 'season']
    a = a.merge(games[game_cols].drop_duplicates('game_id'), on='game_id', how='inner', validate='many_to_one')
    for col in ['goals', 'assists', 'minutes_played']:
        a[col] = pd.to_numeric(a[col], errors='coerce')
    a = a[a['minutes_played'].between(0, 130) & a['goals'].ge(0) & a['assists'].ge(0)].copy()
    a['player_id'] = a['player_id'].astype(int)
    a['player_club_id'] = a['player_club_id'].astype('Int64')

    # Group across clubs only within SAME competition/season. Club IDs should
    # remain separately available in future for transferred-player views.
    result = a.groupby(['player_id', 'season', 'competition_id'], as_index=False).agg(
        Player=('player_name', 'first'),
        Games=('game_id', 'nunique'),
        Goals=('goals', 'sum'),
        Assists=('assists', 'sum'),
        Minutes=('minutes_played', 'sum'),
        Clubs=('player_club_id', 'nunique'),
    )
    result['GoalsPer90'] = (result['Goals'] * 90 / result['Minutes'].replace(0, float('nan'))).round(3)
    result['AssistsPer90'] = (result['Assists'] * 90 / result['Minutes'].replace(0, float('nan'))).round(3)
    result['Source'] = 'Transfermarkt community dataset'
    result['Coverage'] = 'Matches present in the source dataset; not guaranteed complete season'
    result.to_csv(OUTPUT / 'transfermarkt_player_seasons.csv', index=False)
    print(f'Transfermarkt: {len(result):,} player-season-competition rows')
    return result


def understat():
    df = pd.read_csv(download('understat.csv', URLS['understat.csv']), low_memory=False)
    required = ['id', 'player_name', 'team_title', 'league', 'year', 'goals', 'assists', 'xG', 'xA', 'shots', 'games', 'time']
    require_columns(df, required, 'understat')
    df = df.copy()
    for field in ['goals', 'assists', 'xG', 'xA', 'shots', 'games', 'time']:
        df[field] = pd.to_numeric(df[field], errors='coerce')
    df = df.dropna(subset=['id', 'year', 'league', 'goals', 'assists', 'xG', 'xA', 'shots', 'time'])
    df = df[
        df['goals'].ge(0) & df['assists'].ge(0) & df['xG'].ge(0)
        & df['xA'].ge(0) & df['shots'].ge(0) & df['time'].ge(0)
    ].copy()
    # Source documents transferred players as separate team rows. Aggregate
    # numerical metrics but retain an explicit club list for transparency.
    df['id'] = df['id'].astype(int)
    df['year'] = df['year'].astype(int)
    result = df.groupby(['id', 'year', 'league'], as_index=False).agg(
        Player=('player_name', 'first'),
        Teams=('team_title', lambda values: ' / '.join(sorted(set(map(str, values))))),
        Goals=('goals', 'sum'),
        Assists=('assists', 'sum'),
        xG=('xG', 'sum'),
        xA=('xA', 'sum'),
        Shots=('shots', 'sum'),
        Games=('games', 'sum'),
        Minutes=('time', 'sum'),
    )
    result['Season'] = result['year'].astype(str) + '-' + ((result['year'] + 1) % 100).astype(str).str.zfill(2)
    result['GoalsPer90'] = (result['Goals'] * 90 / result['Minutes'].replace(0, float('nan'))).round(3)
    result['AssistsPer90'] = (result['Assists'] * 90 / result['Minutes'].replace(0, float('nan'))).round(3)
    result['Source'] = 'Understat community aggregation'
    result['Coverage'] = 'Source-reported league season; validate coverage for current/incomplete seasons'
    result.to_csv(OUTPUT / 'understat_player_seasons.csv', index=False)
    print(f'Understat: {len(result):,} player-season-league rows')
    return result


def print_examples(label, frame):
    print('\n' + label)
    for name in ['Messi', 'Ronaldo', 'Haaland', 'Mbapp', 'Yamal']:
        found = frame[frame['Player'].str.contains(name, case=False, na=False, regex=False)]
        print(f'  {name}: {len(found)} season/competition rows')
        if name in ('Messi', 'Ronaldo') and not found.empty:
            print('    Names:', ', '.join(found['Player'].drop_duplicates().head(4).tolist()))


def main():
    print('PitchIQ Player Data Preparation')
    print('Outputs will be separate by provider; existing players.csv is untouched.\n')
    try:
        tm = transfermarkt()
        print_examples('Transfermarkt coverage', tm)
    except Exception as exc:
        print(f'Transfermarkt source unavailable or incompatible: {exc}')
    try:
        us = understat()
        print_examples('Understat coverage', us)
    except Exception as exc:
        print(f'Understat source unavailable or incompatible: {exc}')
    print('\nCompleted available sources. Review results before replacing Player Comparison.')
    print('Do not present different provider metrics as measured on the same matches.')
    print('Check upstream publication and redistribution terms before shipping raw CSVs.')


if __name__ == '__main__':
    main()
