"""PitchIQ: audit and clean downloaded player-season tables WITHOUT merging providers.

Run after: python prepare_players.py
Requires: pandas
Outputs to data/player_comparison/cleaned/; original files untouched.
"""
from pathlib import Path
import re
import pandas as pd

ROOT = Path(__file__).resolve().parent
INPUT = ROOT / 'data' / 'player_comparison'
OUTPUT = INPUT / 'cleaned'
OUTPUT.mkdir(parents=True, exist_ok=True)

SOURCES = {
    'transfermarkt': {
        'file': 'transfermarkt_player_seasons.csv',
        'id': 'player_id', 'season': 'season', 'competition': 'competition_id',
        'integer': ['Games', 'Goals', 'Assists', 'Minutes', 'Clubs'],
        'required': ['player_id', 'Player', 'season', 'competition_id', 'Games', 'Goals', 'Assists', 'Minutes'],
    },
    'understat': {
        'file': 'understat_player_seasons.csv',
        'id': 'id', 'season': 'Season', 'competition': 'league',
        'integer': ['Games', 'Goals', 'Assists', 'Shots', 'Minutes'],
        'required': ['id', 'Player', 'Season', 'league', 'Games', 'Goals', 'Assists', 'Minutes', 'Shots', 'xG', 'xA'],
    },
}

# Known identity checks, using *full names* to prevent including namesakes.
# IDs are resolved from the provider's table and displayed for verification;
# they are never matched across data providers automatically.
WATCH_NAMES = {
    'Lionel Messi': ['Lionel Messi', 'Lionel Andrés Messi Cuccittini'],
    'Cristiano Ronaldo': ['Cristiano Ronaldo', 'Cristiano Ronaldo dos Santos Aveiro'],
    'Erling Haaland': ['Erling Haaland', 'Erling Braut Haaland'],
    'Kylian Mbappé': ['Kylian Mbappé', 'Kylian Mbappe'],
    'Lamine Yamal': ['Lamine Yamal', 'Lamine Yamal Nasraoui Ebana'],
}


def normalize_name(value):
    return re.sub(r'\s+', ' ', str(value).strip()).casefold()


def clean_source(source, spec):
    path = INPUT / spec['file']
    if not path.exists():
        print(f'[{source}] Missing {path}. Run python prepare_players.py first.')
        return

    df = pd.read_csv(path, low_memory=False)
    count_in = len(df)
    missing = sorted(set(spec['required']) - set(df.columns))
    if missing:
        print(f'[{source}] Missing required columns: {missing}. File unchanged.')
        return

    # Keep a human-readable audit trail of rejected rows.
    rejected = []
    def reject(mask, reason):
        nonlocal df
        mask = mask.fillna(True)
        bad = df.loc[mask].copy()
        if len(bad):
            bad['RejectionReason'] = reason
            rejected.append(bad)
            df = df.loc[~mask].copy()

    id_col = spec['id']
    season_col = spec['season']
    comp_col = spec['competition']
    numbers = list(dict.fromkeys(spec['integer'] + (['xG', 'xA'] if source == 'understat' else [])))
    for col in numbers + [id_col]:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    reject(
        df[id_col].isna() | (df[id_col] <= 0)
        | df['Player'].isna() | df['Player'].astype(str).str.strip().eq('')
        | df[season_col].isna() | df[comp_col].isna(),
        'Missing player identity, season, or competition',
    )
    reject(df[numbers].isna().any(axis=1), 'Missing or nonnumeric statistics')
    reject(df[numbers].lt(0).any(axis=1), 'Negative statistics')
    reject(
        (df['Goals'] > df['Games'] * 10)
        | (df['Assists'] > df['Games'] * 10)
        | (df['Minutes'] > df['Games'] * 130)
        | ((df['Games'] == 0) & (df[['Goals', 'Assists', 'Minutes']].sum(axis=1) > 0)),
        'Implausible statistics for recorded games',
    )
    if source == 'understat':
        reject(df['Goals'] > df['Shots'], 'Goals exceed recorded shots')
        reject(df['xG'] > df['Shots'] + 1e-6, 'xG exceeds number of shots')

    # Duplicate identity+season+competition implies risk of double counting.
    # Do not arbitrarily keep one conflicting row: quarantine all of them.
    key = [id_col, season_col, comp_col]
    df = df.drop_duplicates().copy()
    reject(df.duplicated(key, keep=False), 'Ambiguous duplicate player-season-competition')

    # Retain genuine zero-minute entries but don't fabricate per-90 values.
    df[id_col] = df[id_col].astype('int64')
    for col in spec['integer']:
        df[col] = df[col].astype('int64')
    df['Player'] = df['Player'].astype(str).str.strip().str.replace(r'\s+', ' ', regex=True)
    df['GoalsPer90'] = (90 * df['Goals'] / df['Minutes'].where(df['Minutes'] > 0)).round(3)
    df['AssistsPer90'] = (90 * df['Assists'] / df['Minutes'].where(df['Minutes'] > 0)).round(3)
    if source == 'understat':
        df['xGPer90'] = (90 * df['xG'] / df['Minutes'].where(df['Minutes'] > 0)).round(3)
        df['xAPer90'] = (90 * df['xA'] / df['Minutes'].where(df['Minutes'] > 0)).round(3)

    # PlayerID is provider-scoped; names can differ across seasons.
    df['ProviderPlayerID'] = source + ':' + df[id_col].astype(str)
    df = df.sort_values(['Player', season_col, comp_col], kind='stable').reset_index(drop=True)
    clean_path = OUTPUT / f'{source}_clean.csv'
    df.to_csv(clean_path, index=False)
    if rejected:
        pd.concat(rejected, ignore_index=True).to_csv(OUTPUT / f'{source}_rejected.csv', index=False)
    print(f'\n[{source}] Rows: {count_in:,} input -> {len(df):,} clean; '
          f'{sum(map(len, rejected)):,} rejected')
    print(f'Saved: {clean_path}')
    print(f'Unique player IDs: {df[id_col].nunique():,}')

    print('Identity checks (exact normalized names, not substring matches):')
    normalized = df['Player'].map(normalize_name)
    for player, aliases in WATCH_NAMES.items():
        found = df[normalized.isin([normalize_name(name) for name in aliases])]
        if found.empty:
            print(f'  {player}: not found under the expected names; manual ID review needed')
        else:
            ids = sorted(found['ProviderPlayerID'].unique())
            print(f'  {player}: {len(found)} rows; IDs={ids[:8]}')
            if len(ids) > 1:
                print('    WARNING: multiple IDs: review before treating as one person')

    # A separate list helps review inconsistent naming for the same ID.
    names_by_id = (df.groupby('ProviderPlayerID')['Player'].nunique())
    multiple = names_by_id[names_by_id > 1]
    if len(multiple):
        examples = df[df['ProviderPlayerID'].isin(multiple.index)][['ProviderPlayerID', 'Player']].drop_duplicates()
        examples.to_csv(OUTPUT / f'{source}_name_variants.csv', index=False)
        print(f'Name variants: {len(multiple)} IDs; exported for review')


def main():
    print('PitchIQ Player Data Cleanup')
    print('Provider datasets remain separate. Originals are never overwritten.')
    for source, spec in SOURCES.items():
        clean_source(source, spec)
    print('\nNext: verify player IDs and compare selected season totals before replacing the UI.')
    print('Do not publish raw third-party datasets without checking reuse permissions.')


if __name__ == '__main__':
    main()
