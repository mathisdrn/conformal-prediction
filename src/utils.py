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
    data: pd.DataFrame | pl.DataFrame, corr_types=["pearson", "spearman"], mark="circle"
):
    """
    Plot the pairwise correlations between columns.

    Parameters
    ----------
    data : DataFrame
        pandas or polars DataFrame with input data.
    corr_types: list of (str or function)
        Which correlations to calculate.
        Anything that is accepted by DataFrame.corr.
    mark: str
        Shape of the points. Passed to Chart.
        One of "circle", "square", "tick", or "point".

    Returns
    -------
    ConcatChart
        Concatenated Chart of the correlation plots laid out in a single row.
    """

    # Convert Polars to pandas if needed
    if isinstance(data, pl.DataFrame):
        data = data.to_pandas()

    subplot_row = []
    for num, corr_type in enumerate(corr_types):
        if num > 0:
            yaxis = alt.Axis(labels=False)
        else:
            yaxis = alt.Axis()
        corr_df = data.select_dtypes(["number", "boolean"]).corr(corr_type)
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
            alt.Chart(corr2, mark=mark, title=f"{corr_type.capitalize()} correlations")  # type: ignore
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
