"""Metrics to evaluate performance of regression models."""

from typing import Literal

import altair as alt
import numpy as np
import pandas as pd
import polars as pl
from polars._typing import PythonLiteral

IntoExpr = PythonLiteral | pl.Expr | pl.Series


def parse_into_expression(expr: IntoExpr) -> pl.Expr:
    """Parse a literal or expression into a Polars expression."""
    if isinstance(expr, str):
        return pl.col(expr)
    if isinstance(expr, pl.Expr):
        return expr
    return pl.lit(expr)


def coverage(
    lower_bound: IntoExpr,
    upper_bound: IntoExpr,
    value: IntoExpr = "True Price",
) -> pl.Expr:
    """Return an expression to compute the coverage of an interval over values."""

    value = parse_into_expression(value)
    lower_bound = parse_into_expression(lower_bound)
    upper_bound = parse_into_expression(upper_bound)

    return (
        value.is_between(lower_bound, upper_bound)
        .mean()
        .mul(100)
        .round(1)
        .cast(pl.Utf8)
        + pl.lit("%")
    ).alias("Coverage")


def pinball_loss(y_true: IntoExpr, y_pred: IntoExpr, alpha: float) -> pl.Expr:
    """Return an expression to compute the Pinball loss."""

    y_true = parse_into_expression(y_true)
    y_pred = parse_into_expression(y_pred)

    residual = y_true - y_pred
    loss = pl.max_horizontal(alpha * residual, (alpha - 1) * residual)
    return loss.mean().cast(pl.Int32).alias(f"Pinball q_{alpha}")


alt.data_transformers.enable("vegafusion")


def plot_correlation(
    data: pd.DataFrame | pl.DataFrame,
    corr_types: list[Literal["pearson", "kendall", "spearman"]],
    mark: Literal["circle", "square", "tick", "point"] = "circle",
) -> alt.ConcatChart:
    """Plot the pairwise correlations between columns.

    Args:
        data (pd.DataFrame | pl.DataFrame): The input data.
        corr_types (list[Literal["pearson", "kendall", "spearman"]]): The types of correlations to compute and plot.
        mark (Literal["circle", "square", "tick", "point"], optional): The mark type to use for the correlation plot, by default "circle".

    Returns:
        alt.ConcatChart: An Altair chart showing the pairwise correlations between columns for each specified correlation type.
    """
    # Convert Polars to pandas if needed
    if isinstance(data, pl.DataFrame):
        data = data.to_pandas()

    subplot_row = []
    for num, corr_type in enumerate(corr_types):
        yaxis = alt.Axis() if num == 0 else alt.Axis(labels=False)
        corr_df = data.select_dtypes(["number", "boolean"]).corr(corr_type)  # type: ignore
        mask = np.zeros_like(corr_df, dtype=bool)
        mask[np.triu_indices_from(mask)] = True
        corr_df[mask] = np.nan

        corr2 = (
            corr_df.reset_index()
            .melt(id_vars="index")
            .dropna()
            .sort_values("variable", ascending=False)
        )
        var_sort = corr2["variable"].value_counts().index.tolist()
        ind_sort = corr2["index"].value_counts().index.tolist()

        subplot_row.append(
            alt.Chart(corr2, mark=mark, title=f"{corr_type.capitalize()} correlations")
            .transform_calculate(abs_value="abs(datum.value)")
            .encode(
                alt.X("index", sort=ind_sort, title=""),
                alt.Y("variable", sort=var_sort[::-1], title="", axis=yaxis),
                alt.Color(
                    "value",
                    title="",
                    scale=alt.Scale(domain=[-1, 1], scheme="blueorange"),
                ),
                alt.Size("abs_value:Q", scale=alt.Scale(domain=[0, 1]), legend=None),
                [
                    alt.Tooltip("value", format=".2f").title("corr"),
                    alt.Tooltip("index").title("x"),
                    alt.Tooltip("variable").title("y"),
                ],
            )
        )

    return (
        alt.concat(*subplot_row).resolve_axis(y="shared").configure_view(strokeWidth=0)
    )
