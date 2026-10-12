from pathlib import Path
import math
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data' / 'shots' / 'shots.csv'
OUT = ROOT / 'models' / 'xg_predictor.joblib'
FEATURES = ['ShotDistance', 'ShotAngle', 'BodyPart', 'ShotType']
MIN_GK_SHOTS = 50
GK_PRIOR = 35.0


def geometry(x, y):
    dx, dy = 120.0 - float(x), 40.0 - float(y)
    left, right = 36.0 - float(y), 44.0 - float(y)
    return (math.hypot(dx, dy), math.degrees(math.atan2(abs(dx * (right - left)), dx * dx + left * right)))


def adjust_probability(p, multiplier):
    odds = np.asarray(p) / np.maximum(1 - np.asarray(p), 1e-12)
    result = odds * np.asarray(multiplier)
    return result / (1 + result)


def main():
    if not DATA.exists():
        raise FileNotFoundError(f'Missing {DATA}')
    df = pd.read_csv(DATA)
    required = ['MatchID', 'X', 'Y', 'Goal', 'BodyPart', 'ShotType', 'ShooterID', 'Shooter', 'StatsBomb_xG', 'Goalkeeper']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f'Missing columns: {missing}. If Goalkeeper is missing, rerun prepare_shots.py (cached events will be reused).')

    for col in ['X', 'Y', 'Goal', 'StatsBomb_xG']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=['MatchID', 'X', 'Y', 'Goal']).copy()
    df = df[df['X'].between(0, 120) & df['Y'].between(0, 80) & df['Goal'].isin([0, 1])].copy()
    for col in ['BodyPart', 'ShotType']:
        df[col] = df[col].fillna('Unknown').astype(str)
    geom = [geometry(x, y) for x, y in zip(df.X, df.Y)]
    df['ShotDistance'] = [g[0] for g in geom]
    df['ShotAngle'] = [g[1] for g in geom]
    df['PlayerKey'] = df.ShooterID.apply(lambda v: str(int(v)) if pd.notna(v) else None)
    df['KeeperKey'] = df.Goalkeeper.fillna('').astype(str).str.strip()
    df.loc[df.KeeperKey.str.lower().isin(['', 'unknown', 'nan', 'none']), 'KeeperKey'] = ''

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(splitter.split(df, df.Goal, groups=df.MatchID))
    train, test = df.iloc[train_idx].copy(), df.iloc[test_idx].copy()
    preprocessor = ColumnTransformer([
        ('numeric', StandardScaler(), ['ShotDistance', 'ShotAngle']),
        ('categorical', OneHotEncoder(handle_unknown='ignore'), ['BodyPart', 'ShotType']),
    ])
    model = Pipeline([('preprocessor', preprocessor), ('classifier', LogisticRegression(max_iter=2000))])
    model.fit(train[FEATURES], train.Goal.astype(int))
    base = model.predict_proba(test[FEATURES])[:, 1]

    valid_train = train[train.PlayerKey.notna() & train.StatsBomb_xG.between(0, 1)].copy()
    grouped = valid_train.groupby('PlayerKey').agg(shots=('Goal','size'), goals=('Goal','sum'), expected=('StatsBomb_xG','sum'), name=('Shooter','first'))
    players = {}
    for key, r in grouped.iterrows():
        if r.shots < 30:
            continue
        factor = np.clip((float(r.goals) + 20.) / (float(r.expected) + 20.), 0.70, 1.30)
        players[key] = {'name':str(r['name']), 'shots':int(r.shots), 'goals':int(r.goals),
                        'reference_xg':round(float(r.expected),3), 'multiplier':float(factor)}

    foot_train = valid_train[valid_train.BodyPart.isin(['Left Foot','Right Foot'])]
    foot_grouped = foot_train.groupby(['PlayerKey','BodyPart']).agg(shots=('Goal','size'), goals=('Goal','sum'), expected=('StatsBomb_xG','sum'))
    feet = {}
    for (key, foot), r in foot_grouped.iterrows():
        if r.shots < 35:
            continue
        factor = np.clip((float(r.goals) + 30.)/(float(r.expected) + 30.), .75, 1.25)
        feet[f'{key}|{foot}'] = {'shots':int(r.shots), 'multiplier':float(factor)}

    # Only shots with an explicitly identified opponent keeper are eligible.
    # Restrict to non-penalties: pre-shot xG is a poor standalone measure of keeper ability
    # for penalties; penalty placement / tactics are not modeled here.
    keeper_train = train[(train.KeeperKey != '') & train.StatsBomb_xG.between(0,1) & (train.ShotType != 'Penalty')].copy()
    keeper_grouped = keeper_train.groupby('KeeperKey').agg(shots=('Goal','size'), goals=('Goal','sum'), expected=('StatsBomb_xG','sum'))
    keepers = {}
    for name, r in keeper_grouped.iterrows():
        if r.shots < MIN_GK_SHOTS:
            continue
        # >1 means conceding more than reference xG, <1 fewer.
        factor = np.clip((float(r.goals) + GK_PRIOR)/(float(r.expected) + GK_PRIOR), .80, 1.20)
        keepers[name] = {'shots':int(r.shots), 'goals_conceded':int(r.goals),
                         'reference_xg_against':round(float(r.expected),3), 'multiplier':float(factor)}

    def shooter_factor(key, foot):
        if key is None:
            return 1.
        item = feet.get(f'{key}|{foot}')
        return item['multiplier'] if item else players.get(key, {}).get('multiplier',1.)

    p_mult = np.array([shooter_factor(k, f) for k, f in zip(test.PlayerKey, test.BodyPart)])
    player_adjusted = adjust_probability(base, p_mult)
    g_mult = np.array([keepers.get(k,{}).get('multiplier',1.) if t != 'Penalty' else 1.
                       for k, t in zip(test.KeeperKey, test.ShotType)])
    combined = adjust_probability(base, p_mult*g_mult)
    y = test.Goal.astype(int)

    print('=== PitchIQ Player + Goalkeeper xG ===')
    print(f'Training shots: {len(train):,}; testing shots: {len(test):,}')
    print(f'Train/test matches: {train.MatchID.nunique():,} / {test.MatchID.nunique():,}')
    print(f'Players with sufficient shots: {len(players):,}; player-foot profiles: {len(feet):,}')
    print(f'Identified goalkeeper shots: {int((df.KeeperKey != "").sum()):,} / {len(df):,}')
    print(f'Goalkeepers with at least {MIN_GK_SHOTS} non-penalty TRAIN shots: {len(keepers)}')
    print(f'Test shots with trained goalkeeper: {sum((test.KeeperKey.isin(keepers)) & (test.ShotType != "Penalty")):,}')
    for label, probs in [('Standard', base), ('Player adjusted', player_adjusted), ('Player + goalkeeper', combined)]:
        print(f'{label}: Brier={brier_score_loss(y,probs):.5f}, log loss={log_loss(y,probs):.5f}, ROC AUC={roc_auc_score(y,probs):.4f}')
    print('Top goalie samples by training shots:')
    for name, info in sorted(keepers.items(), key=lambda v: -v[1]['shots'])[:12]:
        print(f"  {name}: {info['shots']} shots, goals={info['goals_conceded']}, reference xG={info['reference_xg_against']:.1f}, factor={info['multiplier']:.3f}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        'model':model, 'features':FEATURES, 'training_shots':len(train), 'testing_shots':len(test),
        'training_matches':train.MatchID.nunique(), 'testing_matches':test.MatchID.nunique(),
        'brier_score':brier_score_loss(y,base), 'player_brier_score':brier_score_loss(y,player_adjusted),
        'combined_brier_score':brier_score_loss(y,combined),
        'player_factors':players, 'player_foot_factors':feet, 'goalkeeper_factors':keepers,
        'goalkeeper_min_shots':MIN_GK_SHOTS, 'goalkeeper_prior':GK_PRIOR,
        'goalkeeper_method':'Training-match identified keeper, non-penalty goals against/reference xG, strongly shrunk and capped odds factor',
    }, OUT)
    print(f'Saved: {OUT}')
    if brier_score_loss(y,combined) >= brier_score_loss(y,player_adjusted):
        print('CAUTION: Goalkeeper adjustment did not improve global held-out Brier; treat as an experiment.')
    print('Goalkeepers not identified on a shot use neutral factor 1.0.')

if __name__ == '__main__':
    main()
