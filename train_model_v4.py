"""PitchIQ V4: validated Dixon-Coles adjustment + logistic/Poisson blend.

Run: python train_model_v4.py
Needs train_model_v2.py, train_model_v3.py, and data/matches/all_matches.csv
in the same project. Neither the active Match Predictor nor V2/V3 are changed.

2020-21/2021-22 -> model fitting for validation
2022-23           -> choose rho and blend weight
2020-21 to 2022-23 -> refit models
2023-24           -> ONE final held-out evaluation
"""
from pathlib import Path
import joblib
import numpy as np
from scipy.stats import poisson
from sklearn.metrics import log_loss, accuracy_score, confusion_matrix

from train_model_v2 import make_examples
from train_model_v3 import (
    load_matches, add_goal_labels, make_poisson_models,
    poisson_outcomes, train_logistic, logistic_outcomes,
    CLASS_ORDER,
)

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / 'models' / 'match_predictor_v4.joblib'
VAL_SEASON = '2022-23'
TEST_SEASON = '2023-24'

# Negative rho modestly increases low-score draws (0-0 and 1-1).
RHO_VALUES = [-0.15, -0.10, -0.05, 0.0, 0.05, 0.10]
# Blend = weight * logistic + (1-weight) * Poisson/DC.
BLEND_WEIGHTS = [0.0, 0.25, 0.50, 0.75, 1.0]


def dixon_coles_probabilities(base, home_mu, away_mu, rho):
    """Apply Dixon-Coles low-score adjustments to [away, draw, home].

    The joint Poisson pmf is changed for 0-0, 0-1, 1-0, 1-1.
    These corrections preserve total probability algebraically.
    """
    h = np.asarray(home_mu, dtype=float)
    a = np.asarray(away_mu, dtype=float)
    p00 = poisson.pmf(0, h) * poisson.pmf(0, a)
    p01 = poisson.pmf(0, h) * poisson.pmf(1, a)
    p10 = poisson.pmf(1, h) * poisson.pmf(0, a)
    p11 = poisson.pmf(1, h) * poisson.pmf(1, a)

    # tau00=1-h*a*rho, tau01=1+h*rho,
    # tau10=1+a*rho, tau11=1-rho
    tau00 = 1 - h * a * rho
    tau01 = 1 + h * rho
    tau10 = 1 + a * rho
    tau11 = 1 - rho
    if np.any(np.minimum.reduce([tau00, tau01, tau10, np.full_like(h, tau11)]) <= 0):
        raise ValueError('rho produces a negative low-score probability')

    corrected = np.asarray(base, dtype=float).copy()
    corrected[:, 0] += p01 * (tau01 - 1)           # Away win 0-1
    corrected[:, 1] += p00 * (tau00 - 1) + p11 * (tau11 - 1)  # Draws
    corrected[:, 2] += p10 * (tau10 - 1)           # Home win 1-0
    corrected = np.clip(corrected, 1e-12, None)
    corrected /= corrected.sum(axis=1, keepdims=True)
    return corrected


def metrics(y, probs):
    predicted = np.asarray(CLASS_ORDER)[np.argmax(probs, axis=1)]
    truth = np.asarray(y)
    mask = truth == 'D'
    # Draw probability calibration error (binary Brier score)
    draw_brier = np.mean((probs[:, 1] - mask.astype(float)) ** 2)
    return {
        'accuracy': float(accuracy_score(truth, predicted)),
        'log_loss': float(log_loss(truth, probs, labels=CLASS_ORDER)),
        'draw_recall': float(np.mean(predicted[mask] == 'D')) if mask.any() else float('nan'),
        'draw_brier': float(draw_brier),
    }


def show(label, stats):
    print(f"  {label:<25} accuracy={stats['accuracy']:.2%}  log loss={stats['log_loss']:.4f}  "
          f"draw recall={stats['draw_recall']:.2%}  draw Brier={stats['draw_brier']:.4f}")


def main():
    print('=== PitchIQ Match Predictor V4: Dixon-Coles + Blend ===')
    matches = load_matches()
    X, meta = make_examples(matches)
    goals = add_goal_labels(matches, meta)
    seasons = sorted(meta.Season.unique())
    print(f'Matches: {len(matches):,}; eligible examples: {len(X):,}')
    print('Seasons:', ', '.join(seasons))
    if VAL_SEASON not in seasons or TEST_SEASON not in seasons:
        raise ValueError('Expected 2022-23 validation and 2023-24 test seasons.')
    early = (meta.Season < VAL_SEASON).to_numpy()
    val = (meta.Season == VAL_SEASON).to_numpy()
    final_train = (meta.Season < TEST_SEASON).to_numpy()
    test = (meta.Season == TEST_SEASON).to_numpy()
    if min(early.sum(), val.sum(), test.sum()) == 0:
        raise ValueError('Training, validation, and test must all contain matches.')
    y = meta.Result
    print(f'Validation fit: {early.sum():,} matches -> {val.sum():,} validation matches')

    logistic = train_logistic(X.loc[early], y.loc[early])
    poisson_models = make_poisson_models(X.loc[early], goals[early])
    val_log = logistic_outcomes(logistic, X.loc[val])
    val_pois, val_h, val_a = poisson_outcomes(poisson_models, X.loc[val])
    print('\nValidation benchmark:')
    show('Logistic', metrics(y.loc[val], val_log))
    show('Uncorrected Poisson', metrics(y.loc[val], val_pois))

    candidates = []
    for rho in RHO_VALUES:
        try:
            dc = dixon_coles_probabilities(val_pois, val_h, val_a, rho)
        except ValueError:
            print(f'  skipping rho={rho:+.2f} (invalid correction)')
            continue
        for w in BLEND_WEIGHTS:
            combined = w * val_log + (1 - w) * dc
            loss = log_loss(y.loc[val], combined, labels=CLASS_ORDER)
            candidates.append((float(loss), rho, w))
    if not candidates:
        raise ValueError('No valid adjustment candidates')
    best_val_loss, best_rho, best_weight = min(candidates, key=lambda r: r[0])
    print(f'\nSelected on validation only: rho={best_rho:+.2f}, logistic weight={best_weight:.2f}, '
          f'validation log loss={best_val_loss:.4f}')
    chosen_val = best_weight * val_log + (1 - best_weight) * dixon_coles_probabilities(
        val_pois, val_h, val_a, best_rho)
    show('V4 selected (validation)', metrics(y.loc[val], chosen_val))

    print(f'\nRefitting on {final_train.sum():,} earlier matches; testing on {test.sum():,} held-out matches')
    logistic = train_logistic(X.loc[final_train], y.loc[final_train])
    poisson_models = make_poisson_models(X.loc[final_train], goals[final_train])
    test_log = logistic_outcomes(logistic, X.loc[test])
    test_pois, test_h, test_a = poisson_outcomes(poisson_models, X.loc[test])
    test_dc = dixon_coles_probabilities(test_pois, test_h, test_a, best_rho)
    test_v4 = best_weight * test_log + (1 - best_weight) * test_dc
    print('\nFinal held-out test (not used to pick parameters):')
    for name, probs in [
        ('V2-style Logistic', test_log),
        ('V3 Poisson', test_pois),
        ('Dixon-Coles Poisson', test_dc),
        ('V4 validated blend', test_v4),
    ]:
        show(name, metrics(y.loc[test], probs))
    pred = np.asarray(CLASS_ORDER)[np.argmax(test_v4, axis=1)]
    print('\nV4 confusion matrix: actual rows and predicted columns H/D/A')
    print(confusion_matrix(y.loc[test], pred, labels=['H', 'D', 'A']))
    print('\nDraw probability check (bins show predictions vs observed draws):')
    observed = (y.loc[test].to_numpy() == 'D').astype(int)
    for low, high in [(0, .15), (.15, .25), (.25, .35), (.35, .50), (.50, 1.01)]:
        mask = (test_v4[:, 1] >= low) & (test_v4[:, 1] < high)
        if mask.any():
            print(f'  {low:.0%}-{min(high, 1):.0%}: n={mask.sum():4d}, '
                  f'avg predicted={test_v4[mask, 1].mean():.1%}, actual={observed[mask].mean():.1%}')

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        'selected_model': 'validated_blend',
        'logistic_model': logistic,
        'poisson_home_model': poisson_models[0],
        'poisson_away_model': poisson_models[1],
        'rho': best_rho,
        'logistic_weight': best_weight,
        'classes': CLASS_ORDER,
        'features': list(X.columns),
        'validation_season': VAL_SEASON,
        'test_season': TEST_SEASON,
        'final_test_metrics': metrics(y.loc[test], test_v4),
        'notes': 'Experimental model. Uses V2 prematch features. Requires updated inference; old Streamlit page is unchanged.',
    }, MODEL_PATH)
    print(f'\nSaved experimental model: {MODEL_PATH}')
    print('V1/V2/V3 and the current Streamlit predictor were not modified.')
    print('Do not tune parameters further on this same final test set.')


if __name__ == '__main__':
    main()
