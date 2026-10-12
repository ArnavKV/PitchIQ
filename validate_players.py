"""PitchIQ player-data audit (run after clean_players.py).

Reads cleaned Transfermarkt and Understat aggregates, checks five notable
player seasons, and writes reports. It does NOT automatically certify that
statistics match official records, join provider IDs, or change the app.

Run: python validate_players.py
Requires: pandas
"""
from pathlib import Path
import unicodedata
import pandas as pd

ROOT = Path(__file__).resolve().parent
CLEAN = ROOT / 'data' / 'player_comparison' / 'cleaned'
OUT = ROOT / 'data' / 'player_comparison' / 'validation'
OUT.mkdir(parents=True, exist_ok=True)

# IDs confirmed as distinct in the existing cleanup output.
# Understat Mbappe is intentionally unresolved until inspected.
PLAYERS = [
    {'name':'Lionel Messi', 'tm':28003, 'us':2097, 'season':2011, 'tm_league':'ES1', 'us_league':'La_liga'},
    {'name':'Cristiano Ronaldo', 'tm':8198, 'us':2371, 'season':2014, 'tm_league':'ES1', 'us_league':'La_liga'},
    {'name':'Erling Haaland', 'tm':418560, 'us':8260, 'season':2022, 'tm_league':'GB1', 'us_league':'EPL'},
    {'name':'Kylian Mbappe', 'tm':342229, 'us':None, 'season':2023, 'tm_league':'FR1', 'us_league':'Ligue_1'},
    {'name':'Lamine Yamal', 'tm':937958, 'us':11527, 'season':2024, 'tm_league':'ES1', 'us_league':'La_liga'},
]

METRICS = ['Games','Goals','Assists','Minutes']

def normalized(s):
    s = unicodedata.normalize('NFKD', str(s))
    return ''.join(c for c in s if not unicodedata.combining(c)).casefold()

def load(name, required):
    path = CLEAN / f'{name}_clean.csv'
    if not path.exists():
        raise FileNotFoundError(f'Missing {path}; run python clean_players.py first.')
    df = pd.read_csv(path, low_memory=False)
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f'{path.name}: missing columns {sorted(missing)}')
    return df

def choose(frame, player_id_col, player_id, season_col, season, competition_col, competition):
    numeric_id = pd.to_numeric(frame[player_id_col], errors='coerce')
    numeric_season = pd.to_numeric(frame[season_col], errors='coerce')
    return frame.loc[
        numeric_id.eq(player_id) & numeric_season.eq(season)
        & frame[competition_col].astype(str).str.casefold().eq(competition.casefold())
    ]

def summarize(row):
    if len(row) != 1:
        return {}
    return {metric:row.iloc[0][metric] for metric in METRICS}

def print_row(label, row):
    if len(row) == 0:
        print(f'  {label}: MISSING (source may not cover this competition/season)')
    elif len(row) > 1:
        print(f'  {label}: WARNING - {len(row)} ambiguous records')
    else:
        r = row.iloc[0]
        print(f"  {label}: {r['Player']} | {int(r['Games'])} apps, {int(r['Goals'])} goals, "
              f"{int(r['Assists'])} assists, {int(r['Minutes'])} min"
              + (f" | xG {r['xG']:.2f}, xA {r['xA']:.2f}" if 'xG' in r.index else ''))

def main():
    tm = load('transfermarkt', ['player_id','season','competition_id','Player',*METRICS])
    us = load('understat', ['id','year','league','Player',*METRICS,'xG','xA'])
    print('PitchIQ: Player Dataset Validation')
    print('Provider counts: Transfermarkt {:,}, Understat {:,}'.format(len(tm),len(us)))
    print('Warning: structural cleanup is not an official-statistics verification.')

    report = []
    for entry in PLAYERS:
        season = entry['season']
        print(f"\n{entry['name']} | {season}-{str((season+1)%100).zfill(2)}")
        a = choose(tm,'player_id',entry['tm'],'season',season,'competition_id',entry['tm_league'])
        b = choose(us,'id',entry['us'],'year',season,'league',entry['us_league']) if entry['us'] is not None else us.iloc[0:0]
        print_row('Transfermarkt',a)
        print_row('Understat',b)
        record = {'Player':entry['name'],'SeasonStart':season,'TransfermarktID':entry['tm'],
                  'UnderstatID':entry['us'],'TransfermarktCompetition':entry['tm_league'],
                  'UnderstatLeague':entry['us_league'], 'TMRowCount':len(a),'USRowCount':len(b)}
        for prefix, rows in [('TM',a),('US',b)]:
            for m, val in summarize(rows).items():
                record[f'{prefix}_{m}'] = val
        if len(a)==1 and len(b)==1:
            for m in METRICS:
                diff = int(a.iloc[0][m]) - int(b.iloc[0][m])
                record[f'{m}_TM_minus_US'] = diff
            print('  Difference (TM minus Understat):',
                  ', '.join(f"{m}={record[m+'_TM_minus_US']:+}" for m in METRICS))
            print('  Note: provider definitions and match coverage may differ.')
        else:
            print('  Comparison incomplete: missing or ambiguous source row.')
        report.append(record)

    comparison = pd.DataFrame(report)
    comparison_path = OUT / 'featured_season_comparison.csv'
    comparison.to_csv(comparison_path,index=False)

    # Surface plausible Mbappe name variants without pretending to have
    # established the correct identity. An ID must be confirmed manually.
    names = us['Player'].astype(str)
    candidate_mask = names.map(normalized).str.contains(r'mbapp|mbape',regex=True,na=False)
    candidates = us.loc[candidate_mask].copy()
    cols = [col for col in ['id','Player','year','league','Games','Goals','Assists','Minutes','xG','xA']
            if col in candidates.columns]
    candidates[cols].sort_values(['Player','year']).to_csv(
        OUT / 'mbappe_understat_candidates.csv',index=False)
    print('\nUnderstat Mbappe name candidates:')
    if candidates.empty:
        print('  No matching names; manually inspect source before linking IDs.')
    else:
        for (pid,name), group in candidates.groupby(['id','Player']):
            print(f"  ID {pid}: {name}; seasons {sorted(group['year'].astype(int).unique().tolist())}")
    
    # Optional manually sourced reference table: never pretend this check was
    # performed unless user has entered references and their citations.
    ref_path = OUT / 'official_reference.csv'
    if ref_path.exists():
        ref = pd.read_csv(ref_path)
        need = {'Player','SeasonStart','Competition','Goals','Assists','Games','ReferenceURL'}
        if need.issubset(ref.columns):
            print('\nFound manual reference table. Review against the report:')
            print(ref[['Player','SeasonStart','Competition','Goals','Assists','Games','ReferenceURL']].to_string(index=False))
        else:
            print('\nManual reference CSV exists but lacks columns:',sorted(need-set(ref.columns)))
    else:
        template = pd.DataFrame([{
            'Player':p['name'],'SeasonStart':p['season'],'Competition':p['tm_league'],
            'Goals':'','Assists':'','Games':'','ReferenceURL':''} for p in PLAYERS])
        template.to_csv(OUT / 'official_reference_template.csv',index=False)
        print('\nCreated blank official_reference_template.csv for sourced manual checks.')

    print(f'\nReports saved in {OUT}')
    print('Next: inspect missing/mismatched seasons and confirm Mbappe ID manually.')
    print('Do not overwrite the Player Comparison data yet.')

if __name__ == '__main__':
    main()
