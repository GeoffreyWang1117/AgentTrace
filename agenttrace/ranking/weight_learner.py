"""
Weight Learning for CausalScorer.

Replaces grid-searched weights with weights learned from data using:
1. Logistic Regression on sub-score features
2. (Optional) LambdaMART learning-to-rank via LightGBM
"""

import json
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from agenttrace.ranking.scorer import CausalScorer, NodeFeatures


@dataclass
class WeightLearningResult:
    """Result of weight learning."""

    method: str
    weights: dict[str, float]
    cross_val_hit_at_1: float
    cross_val_mrr: float
    feature_names: list[str]


WEIGHT_KEYS = ["position", "structure", "content", "flow", "confidence"]


def prepare_training_data(
    all_features: list[list[NodeFeatures]],
    all_ground_truths: list[str],
) -> tuple[np.ndarray, np.ndarray, list[list[int]]]:
    """
    Prepare training data from feature lists.

    Args:
        all_features: List of per-scenario feature lists (one NodeFeatures per candidate)
        all_ground_truths: Root cause node ID for each scenario

    Returns:
        X: Feature matrix (n_candidates x 5)
        y: Labels (1 for root cause, 0 otherwise)
        groups: Query groups for LTR
    """
    X_rows = []
    y_rows = []
    groups = []

    for features_list, gt_id in zip(all_features, all_ground_truths):
        group_size = 0
        for features in features_list:
            vec = CausalScorer.extract_feature_vector(features)
            X_rows.append(vec)
            y_rows.append(1.0 if features.node_id == gt_id else 0.0)
            group_size += 1
        groups.append(group_size)

    return np.array(X_rows), np.array(y_rows), groups


def train_logistic_regression(
    X: np.ndarray,
    y: np.ndarray,
    groups: list[int],
    n_folds: int = 5,
) -> WeightLearningResult:
    """
    Train logistic regression on sub-scores and derive weights.

    Uses 5-fold cross-validation at the scenario level.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    # Build scenario-level fold indices
    n_scenarios = len(groups)
    fold_size = max(1, n_scenarios // n_folds)
    _scenario_indices = list(range(n_scenarios))

    # Scenario-level cross-validation
    hit_at_1_scores = []
    mrr_scores = []
    all_coefs = []

    for fold in range(n_folds):
        test_start = fold * fold_size
        test_end = min(test_start + fold_size, n_scenarios)
        test_scenarios = set(range(test_start, test_end))

        # Build train/test splits by scenario
        train_idx = []
        test_idx = []
        offset = 0
        for s_idx, g_size in enumerate(groups):
            indices = list(range(offset, offset + g_size))
            if s_idx in test_scenarios:
                test_idx.extend(indices)
            else:
                train_idx.extend(indices)
            offset += g_size

        if not train_idx or not test_idx:
            continue

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        model = LogisticRegression(max_iter=1000, C=1.0)
        model.fit(X_train_s, y_train)
        all_coefs.append(model.coef_[0])

        # Evaluate: compute Hit@1 and MRR per test scenario
        probs = model.predict_proba(X_test_s)[:, 1]
        test_offset = 0
        for s_idx in sorted(test_scenarios):
            if s_idx >= len(groups):
                break
            g_size = groups[s_idx]
            group_probs = probs[test_offset : test_offset + g_size]
            group_labels = y_test[test_offset : test_offset + g_size]
            test_offset += g_size

            if len(group_probs) == 0:
                continue

            ranked_indices = np.argsort(-group_probs)
            gt_positions = np.where(group_labels[ranked_indices] == 1.0)[0]

            if len(gt_positions) > 0:
                rank = gt_positions[0] + 1
                hit_at_1_scores.append(1.0 if rank == 1 else 0.0)
                mrr_scores.append(1.0 / rank)
            else:
                hit_at_1_scores.append(0.0)
                mrr_scores.append(0.0)

    # Average coefficients across folds -> weights
    avg_coefs = np.mean(all_coefs, axis=0) if all_coefs else np.ones(5) / 5
    # Normalize to positive weights that sum to 1
    abs_coefs = np.abs(avg_coefs)
    weights_arr = abs_coefs / abs_coefs.sum()

    weights = {k: float(w) for k, w in zip(WEIGHT_KEYS, weights_arr)}

    return WeightLearningResult(
        method="logistic_regression",
        weights=weights,
        cross_val_hit_at_1=np.mean(hit_at_1_scores) if hit_at_1_scores else 0.0,
        cross_val_mrr=np.mean(mrr_scores) if mrr_scores else 0.0,
        feature_names=WEIGHT_KEYS,
    )


def train_lambdamart(
    X: np.ndarray,
    y: np.ndarray,
    groups: list[int],
    n_folds: int = 5,
) -> Optional[WeightLearningResult]:
    """
    Train LambdaMART via LightGBM for learning-to-rank.

    Returns None if lightgbm is not installed.
    """
    try:
        import lightgbm as lgb
    except ImportError:
        return None

    n_scenarios = len(groups)
    fold_size = max(1, n_scenarios // n_folds)

    hit_at_1_scores = []
    mrr_scores = []
    importances = []

    for fold in range(n_folds):
        test_start = fold * fold_size
        test_end = min(test_start + fold_size, n_scenarios)
        test_scenarios = set(range(test_start, test_end))

        train_idx = []
        test_idx = []
        train_groups = []
        test_groups = []
        offset = 0
        for s_idx, g_size in enumerate(groups):
            indices = list(range(offset, offset + g_size))
            if s_idx in test_scenarios:
                test_idx.extend(indices)
                test_groups.append(g_size)
            else:
                train_idx.extend(indices)
                train_groups.append(g_size)
            offset += g_size

        if not train_idx or not test_idx:
            continue

        train_ds = lgb.Dataset(X[train_idx], label=y[train_idx], group=train_groups)
        params = {
            "objective": "lambdarank",
            "metric": "ndcg",
            "ndcg_eval_at": [1, 3, 5],
            "num_leaves": 15,
            "learning_rate": 0.1,
            "verbose": -1,
        }
        model = lgb.train(params, train_ds, num_boost_round=100)
        importances.append(model.feature_importance("gain"))

        # Evaluate
        preds = model.predict(X[test_idx])
        test_offset = 0
        for g_size in test_groups:
            group_preds = preds[test_offset : test_offset + g_size]
            group_labels = y[test_idx[test_offset : test_offset + g_size]]
            test_offset += g_size

            ranked_indices = np.argsort(-group_preds)
            gt_positions = np.where(group_labels[ranked_indices] == 1.0)[0]
            if len(gt_positions) > 0:
                rank = gt_positions[0] + 1
                hit_at_1_scores.append(1.0 if rank == 1 else 0.0)
                mrr_scores.append(1.0 / rank)
            else:
                hit_at_1_scores.append(0.0)
                mrr_scores.append(0.0)

    # Derive weights from feature importance
    avg_imp = np.mean(importances, axis=0) if importances else np.ones(5) / 5
    weights_arr = avg_imp / avg_imp.sum()
    weights = {k: float(w) for k, w in zip(WEIGHT_KEYS, weights_arr)}

    return WeightLearningResult(
        method="lambdamart",
        weights=weights,
        cross_val_hit_at_1=np.mean(hit_at_1_scores) if hit_at_1_scores else 0.0,
        cross_val_mrr=np.mean(mrr_scores) if mrr_scores else 0.0,
        feature_names=WEIGHT_KEYS,
    )


def save_learned_weights(result: WeightLearningResult, path: Path):
    """Save learned weights to JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(
            {
                "method": result.method,
                "weights": result.weights,
                "cross_val_hit_at_1": result.cross_val_hit_at_1,
                "cross_val_mrr": result.cross_val_mrr,
            },
            f,
            indent=2,
        )
