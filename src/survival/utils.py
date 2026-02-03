import numpy as np
import pandas as pd
import polars as pl
from sksurv.metrics import (
    brier_score,
    concordance_index_censored,
    cumulative_dynamic_auc,
    integrated_brier_score,
)
from sksurv.util import Surv


def get_survival_at_t(surv_fns, t):
    """Standardizes survival probability extraction across libraries."""
    if isinstance(surv_fns, pd.DataFrame):  # lifelines
        return np.array(
            [np.interp(t, surv_fns.index, surv_fns[col]) for col in surv_fns.columns]
        )
    else:  # sksurv (StepFunction objects)
        return np.array([fn(t) for fn in surv_fns])


def evaluate_survival_model(
    y_train: pl.DataFrame, y_test: pl.DataFrame, risk_scores, surv_fns=None
) -> pl.DataFrame:
    """Calcule les métriques d'évaluation pour un modèle de survie."""

    # Convertir les DataFrames polars en structures sksurv
    y_train = Surv.from_dataframe("event", "time", y_train.to_pandas())
    y_test = Surv.from_dataframe("event", "time", y_test.to_pandas())

    metrics = dict()

    # 1. C-Index
    metrics["C-index"] = concordance_index_censored(
        y_test["event"], y_test["time"], risk_scores
    )[0]

    # 2. td-AUC
    safe_limit = y_train["time"].max() * 0.95
    times_auc = np.quantile(y_test["time"][y_test["event"] == 1], [0.25, 0.5, 0.75])
    times_auc = times_auc[times_auc < safe_limit]

    mask = y_test["time"] < safe_limit
    _, mean_auc = cumulative_dynamic_auc(
        y_train, y_test[mask], risk_scores[mask], times_auc
    )
    metrics["Mean td-AUC"] = mean_auc

    # Integrated Brier Score and Brier Score at t_median
    if surv_fns is not None:
        # 1. Define evaluation times
        test_times = np.percentile(y_test["time"], np.linspace(10, 90, 15))

        if isinstance(surv_fns, pd.DataFrame):
            preds_at_times = np.array(
                [
                    np.interp(test_times, surv_fns.index, surv_fns[col])
                    for col in surv_fns.columns
                ]
            )
        else:
            # Standard sksurv StepFunction handling
            preds_at_times = np.array([f(test_times) for f in surv_fns])

        # 3. Integrated Brier Score
        metrics["IBS"] = integrated_brier_score(
            y_train, y_test, preds_at_times, test_times
        )

        # 4. Specific Brier at Median
        t_med = np.median(y_train["time"][y_train["event"]])

        if isinstance(surv_fns, pd.DataFrame):
            s_at_t_med = np.array(
                [
                    np.interp(t_med, surv_fns.index, surv_fns[col])
                    for col in surv_fns.columns
                ]
            )
        else:
            s_at_t_med = np.array([fn(t_med) for fn in surv_fns])

        _, brier_med = brier_score(y_train, y_test, s_at_t_med, t_med)
        metrics[f"Brier (t={t_med})"] = brier_med[0]

    return pl.DataFrame(metrics)
