import marimo

__generated_with = "0.14.13"
app = marimo.App(width="full")


@app.cell
def _():
    from collections import deque

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go
    import statsmodels.api as sm
    from plotly.subplots import make_subplots
    from scipy.linalg import inv as scipy_inv

    return deque, go, make_subplots, mo, np, pd, plt, px, scipy_inv, sm


@app.cell
def _(mo):
    mo.md(
        r"""
    # Bayesian Policy Maker (No Constant)
    How much information should a policy maker take from pre-2000 inflation as regards persistence in the post-2000 era.

    This note replicates the methdology outlined in [Kiley](https://www.ijcb.org/journal/ijcb24q1a6.pdf), extending it to UK data, but **without using a constant term** in the Phillips Curve estimation.
    """
    )
    return


@app.cell
def _(mo):
    mo.md(r"""## Define shared functions and configure start / end dates""")
    return


@app.cell
def _(np):
    def prepare_data(
        df_input, cpi_col_name="core_cpi", unemployment_col_name="unemployment_rate"
    ):
        """
        Prepares the data for Phillips Curve estimation.
        Calculates inflation, lagged unemployment, and average lagged inflation.
        """
        df = df_input.copy()
        df = df.sort_index()  # Ensure data is sorted by date

        # Calculate Core CPI inflation: 12-month log difference, as a percentage
        # delta_p(t) = (ln(CPI_t) - ln(CPI_{t-12})) * 100
        df["core_cpi_log"] = np.log(df[cpi_col_name])
        df["delta_p"] = (
            df["core_cpi_log"].diff(1) * 100
        )  # 1 vs 12 - monthly vs annual rate

        # Create lagged unemployment rate: u(t-1)
        df["unemployment_rate_lag1"] = df[unemployment_col_name].shift(1)

        # Create average lagged inflation: sum_{j=1 to 12} delta_p(t-j) / 12
        # This means taking delta_p, shifting it by 1 (to get t-1),
        # then taking a 12-period rolling mean.
        df["avg_lag_inflation"] = df["delta_p"].shift(1).rolling(window=12).mean()

        # Dependent variable is current inflation
        df["Y"] = df["delta_p"]

        # Independent variables for regression at time t:
        # X1: avg_lag_inflation(t)
        # X2: unemployment_rate_lag1(t)
        df_reg = df[["Y", "avg_lag_inflation", "unemployment_rate_lag1"]].copy()

        # Drop rows with NaNs created by diffs, shifts, and rolling windows
        df_reg = df_reg.dropna()
        return df_reg

    return (prepare_data,)


@app.cell
def _(sm):
    def run_ols(
        df_sample, y_col="Y", x_cols=["avg_lag_inflation", "unemployment_rate_lag1"]
    ):
        """
        Runs OLS regression and returns key statistics.
        Model: Y = b1*X1 + b2*X2 (NO constant term, as specified)
        """
        Y = df_sample[y_col]
        X = df_sample[x_cols]
        # Do NOT add constant for this version
        # X = sm.add_constant(X, has_constant="skip", prepend=False)

        model = sm.OLS(Y, X)
        results = model.fit()

        coeffs = results.params
        # Standard errors from OLS
        std_errs = results.bse
        # Variance-covariance matrix of coefficients
        vcov_coeffs = results.cov_params()
        # Sigma squared (variance of residuals)
        sigma_sq_error = results.mse_resid

        return (
            coeffs,
            std_errs,
            vcov_coeffs,
            sigma_sq_error,
            X,
            Y,
            results.nobs,
            results,
        )

    return (run_ols,)


@app.cell
def _(np, scipy_inv):
    def run_bayesian_estimation(
        gamma_prior, V_prior, gamma_ls, X_post, sigma_sq_post, w
    ):
        """
        Performs Bayesian estimation based on Equation 4 of Kiley (2022).
        This version is refactored for simplicity to more closely mirror the paper's formula.

        Args:
            gamma_prior (np.ndarray): The prior mean vector (Γ_tilde).
            V_prior (np.ndarray): The prior variance-covariance matrix (V).
            gamma_ls (np.ndarray): The least-squares estimate from the data (Γ_LS).
            X_post (np.ndarray): The design matrix from the data (X).
            sigma_sq_post (float): The variance of the error term from the data (σ^2).
            w (float): The weight on the prior information.

        Returns:
            tuple: A tuple containing:
                - np.ndarray: The posterior mean coefficients (Γ_hat).
                - np.ndarray: The posterior standard errors.
        """
        print(f"Calculating posterior for {w}")
        # --- Components of Equation (4) ---
        # Prior precision (V^-1)
        inv_V_prior = scipy_inv(V_prior)

        # Data precision (σ^-2 * X'X)
        data_precision = (1 / sigma_sq_post) * (X_post.T @ X_post)

        # --- Calculation based on Equation (4) ---
        # Posterior variance-covariance matrix: (w*V^-1 + (1-w)*σ^-2*X'X)^-1
        posterior_vcov = scipy_inv((w * inv_V_prior) + ((1 - w) * data_precision))

        # Posterior mean: post_vcov @ (w*V^-1*Γ_prior + (1-w)*σ^-2*X'X*Γ_LS)
        prior_component = w * inv_V_prior @ gamma_prior
        data_component = (1 - w) * data_precision @ gamma_ls
        posterior_mean = posterior_vcov @ (prior_component + data_component)

        # Standard errors are the square root of the diagonal of the posterior variance
        posterior_std_errs = np.sqrt(np.diag(posterior_vcov))

        return posterior_mean, posterior_std_errs

    return (run_bayesian_estimation,)


@app.cell
def _(deque, np, pd):
    def iterative_forecast(coeffs, initial_inf_lags, unemployment_lags):
        """
        Generates multi-step ahead forecasts for inflation as a pandas Series.
        Modified for no constant term version.

        Args:
            coeffs (pd.Series): Model coefficients (no constant).
            initial_inf_lags (list or np.array): The 12 most recent actual inflation values.
            unemployment_lags (pd.Series): Assumed unemployment rates for the forecast horizon,
                                           with a DatetimeIndex.

        Returns:
            pd.Series: A Series of forecasted inflation values with a DatetimeIndex.
        """
        num_lags = len(initial_inf_lags)
        inflation_lags = deque(initial_inf_lags, maxlen=num_lags)
        forecasts = {}

        inf_coeff, unemp_coeff = coeffs  # Only 2 coefficients, no constant

        for date, u_lag in unemployment_lags.items():
            avg_inf_lag = np.mean(inflation_lags)
            forecast = (inf_coeff * avg_inf_lag) + (
                unemp_coeff * u_lag
            )  # No constant term

            forecasts[date] = forecast
            inflation_lags.append(forecast)

        return pd.Series(forecasts, name="inflation_forecast")

    return (iterative_forecast,)


@app.cell
def _(np):
    def process_forecast(forecast_series, base_df, initial_lags_end):
        """
        Processes a single forecast series to rebase it.
        """
        df = base_df.copy()

        # Rename the forecast series to join on it
        forecast_series.name = "delta_p_fcst"

        # Combine the historical data with the forecast
        df = forecast_series.to_frame().combine_first(df)

        # Convert from log-differences back to level
        df["delta_p_fcst"] = df["delta_p_fcst"] / 100
        df["delta_p_fcst"] = df["delta_p_fcst"].fillna(df["core_cpi_log"])
        df["delta_p_fcst"] = df["delta_p_fcst"].cumsum()
        df["delta_p_fcst"] = np.exp(df["delta_p_fcst"])

        # Bring into 12 change
        df["delta_p_fcst"] = df["delta_p_fcst"].diff(12)  # vs diff(12)

        # Return only the forecasted portion
        return df["delta_p_fcst"].loc[df.index > initial_lags_end]

    return (process_forecast,)


@app.cell
def _(pd):
    first_date = pd.Timestamp("1958-01-01")
    last_date = pd.Timestamp("2019-12-31")
    pre_2000_end = pd.Timestamp("1999-12-31")
    post_2000_start = pd.Timestamp("2000-01-01")
    return first_date, last_date, post_2000_start, pre_2000_end


@app.cell
def _(mo):
    mo.md(r"""# US Version""")
    return


@app.cell
def _(mo):
    mo.md(r"""## Load US data""")
    return


@app.cell
def _(pd):
    us_unemployment = pd.read_csv("data/Unemployment Rate UNRATE.csv")
    us_unemployment["observation_date"] = pd.to_datetime(
        us_unemployment["observation_date"]
    )
    us_core_cpi = pd.read_csv("data/CPI Less Food and Energy.csv")
    us_core_cpi["observation_date"] = pd.to_datetime(us_core_cpi["observation_date"])

    us_data = pd.merge(us_unemployment, us_core_cpi, on="observation_date", how="left")
    us_data = us_data.rename(
        columns={"UNRATE": "unemployment_rate", "CPILFESL": "core_cpi"}
    )
    # us_data = us_data.dropna()
    us_data.set_index("observation_date", inplace=True)
    us_data.info()
    return (us_data,)


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Validate Input Data
    Table 1 and Figure 1
    """
    )
    return


@app.cell(hide_code=True)
def _(go, make_subplots, us_data):
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=("Core CPI Prices", "Unemployment Rate"),
    )

    fig.add_trace(
        go.Scatter(
            x=us_data.index,
            y=us_data["core_cpi"].pct_change(12),
            name="Core CPI",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=us_data.index,
            y=us_data["unemployment_rate"],
            name="Unemployment Rate",
        ),
        row=2,
        col=1,
    )

    fig.update_yaxes(
        title_text="Percent change from<br>12 months earlier", row=1, col=1
    )
    fig.update_yaxes(title_text="Percent", row=2, col=1)

    fig.update_layout(
        height=750,
        showlegend=False,
        title_text="Figure 1. CPI Inflation and the Civilian Unemployment Rate",
    )

    fig.show()
    return


@app.cell(hide_code=True)
def _(prepare_data, px, us_data):
    regression_df = prepare_data(us_data)

    fig_input = px.line(
        regression_df,
        y=["Y", "avg_lag_inflation", "unemployment_rate_lag1"],
        title="Input Data for regression",
    )

    fig_input.update_layout(
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return (regression_df,)


@app.cell(hide_code=True)
def _(first_date, last_date, pd, post_2000_start, pre_2000_end, us_data):
    df = us_data.copy()
    df["core_cpi_1m_change"] = df["core_cpi"].diff(periods=1)
    df["core_cpi_12m_change"] = df["core_cpi"].diff(periods=12)

    full_sample = df.loc[first_date:last_date]
    pre_2000_sample = df.loc[first_date:pre_2000_end]
    post_1999_sample = df.loc[post_2000_start:last_date]

    summary_rows = [
        # Full Sample
        {
            "Sample": "Full sample",
            "Variable": "CPI inflation (percent)",
            "Observations": full_sample["core_cpi_1m_change"].count(),
            "Mean (annual rate)": round(full_sample["core_cpi_12m_change"].mean(), 3),
            "Std. Deviation": round(full_sample["core_cpi_1m_change"].std(), 3),
            "Auto-Correlation": round(
                full_sample["core_cpi_1m_change"].autocorr(lag=1), 3
            ),
        },
        {
            "Sample": "Full sample",
            "Variable": "Unemployment rate (percent)",
            "Observations": "-",
            "Mean (annual rate)": round(full_sample["unemployment_rate"].mean(), 3),
            "Std. Deviation": round(full_sample["unemployment_rate"].std(), 3),
            "Auto-Correlation": round(
                full_sample["unemployment_rate"].autocorr(lag=1), 3
            ),
        },
        # Pre-2000 Sample
        {
            "Sample": "Pre-2000 sample",
            "Variable": "CPI inflation (percent)",
            "Observations": pre_2000_sample["core_cpi_1m_change"].count(),
            "Mean (annual rate)": round(
                pre_2000_sample["core_cpi_12m_change"].mean(), 3
            ),
            "Std. Deviation": round(pre_2000_sample["core_cpi_1m_change"].std(), 3),
            "Auto-Correlation": round(
                pre_2000_sample["core_cpi_1m_change"].autocorr(lag=1), 3
            ),
        },
        {
            "Sample": "Pre-2000 sample",
            "Variable": "Unemployment rate (percent)",
            "Observations": "-",
            "Mean (annual rate)": round(pre_2000_sample["unemployment_rate"].mean(), 3),
            "Std. Deviation": round(pre_2000_sample["unemployment_rate"].std(), 3),
            "Auto-Correlation": round(
                pre_2000_sample["unemployment_rate"].autocorr(lag=1), 3
            ),
        },
        # Post-1999 Sample
        {
            "Sample": "Post-1999 sample",
            "Variable": "CPI inflation (percent)",
            "Observations": post_1999_sample["core_cpi_1m_change"].count(),
            "Mean (annual rate)": round(
                post_1999_sample["core_cpi_12m_change"].mean(), 3
            ),
            "Std. Deviation": round(post_1999_sample["core_cpi_1m_change"].std(), 3),
            "Auto-Correlation": round(
                post_1999_sample["core_cpi_1m_change"].autocorr(lag=1), 3
            ),
        },
        {
            "Sample": "Post-1999 sample",
            "Variable": "Unemployment rate (percent)",
            "Observations": "-",
            "Mean (annual rate)": round(
                post_1999_sample["unemployment_rate"].mean(), 3
            ),
            "Std. Deviation": round(post_1999_sample["unemployment_rate"].std(), 3),
            "Auto-Correlation": round(
                post_1999_sample["unemployment_rate"].autocorr(lag=1), 3
            ),
        },
    ]

    summary_df = pd.DataFrame(summary_rows)

    final_rows = [
        # Full Sample Section
        {
            "Variable": "Full sample",
            "Observations": "",
            "Mean (annual rate)": "",
            "Std. Deviation": "",
            "Auto-Correlation": "",
        },
        *summary_df[summary_df["Sample"] == "Full sample"]
        .drop("Sample", axis=1)
        .to_dict("records"),
        # Pre-2000 Section
        {
            "Variable": "Pre-2000 sample",
            "Observations": "",
            "Mean (annual rate)": "",
            "Std. Deviation": "",
            "Auto-Correlation": "",
        },
        *summary_df[summary_df["Sample"] == "Pre-2000 sample"]
        .drop("Sample", axis=1)
        .to_dict("records"),
        # Post-1999 Section
        {
            "Variable": "Post-1999 sample",
            "Observations": "",
            "Mean (annual rate)": "",
            "Std. Deviation": "",
            "Auto-Correlation": "",
        },
        *summary_df[summary_df["Sample"] == "Post-1999 sample"]
        .drop("Sample", axis=1)
        .to_dict("records"),
    ]

    final_table = pd.DataFrame(final_rows)
    final_table
    return


@app.cell
def _(mo):
    mo.md(r"""## Perform least-squares regression""")
    return


@app.cell
def _(first_date, last_date, post_2000_start, pre_2000_end, regression_df):
    df_pre_2000 = regression_df.loc[first_date:pre_2000_end]
    df_post_2000 = regression_df.loc[post_2000_start:last_date]
    df_full_sample = regression_df.loc[first_date:last_date]

    results_summary = {}
    return df_full_sample, df_post_2000, df_pre_2000, results_summary


@app.cell
def _(mo):
    mo.md(r"""### Prior estimation""")
    return


@app.cell
def _(df_pre_2000, mo, results_summary, run_ols):
    (
        prior_mean_coeffs,
        prior_std_errs,
        prior_vcov_coeffs,
        prior_sigma_sq_error,
        X_pre,
        Y_pre,
        prior_nobs,
        prior_model,
    ) = run_ols(df_pre_2000)

    results_summary["OLS Pre-2000 (Prior)"] = {
        "b(1) Coeff": prior_mean_coeffs["avg_lag_inflation"],
        "b(1) SE": prior_std_errs["avg_lag_inflation"],
        "a Coeff": prior_mean_coeffs["unemployment_rate_lag1"],
        "a SE": prior_std_errs["unemployment_rate_lag1"],
        "N": prior_nobs,
    }
    # results_summary['OLS Pre-2000 (Prior)']
    mo.md(prior_model.summary().tables[1].as_html())
    return prior_mean_coeffs, prior_vcov_coeffs


@app.cell
def _(mo):
    mo.md(r"""### Post-2000 estimation""")
    return


@app.cell
def _(df_post_2000, mo, results_summary, run_ols):
    # OLS for Likelihood (Post-2000 data) - also used for "Uninformative Prior"
    (
        ls_coeffs_post,
        ls_std_errs_post,
        _,
        sigma_sq_error_post,
        X_post,
        Y_post,
        post_nobs,
        post_model,
    ) = run_ols(df_post_2000)
    results_summary["OLS Post-2000 (Uninf. Prior)"] = {
        "b(1) Coeff": ls_coeffs_post["avg_lag_inflation"],
        "b(1) SE": ls_std_errs_post["avg_lag_inflation"],
        "a Coeff": ls_coeffs_post["unemployment_rate_lag1"],
        "a SE": ls_std_errs_post["unemployment_rate_lag1"],
        "N": post_nobs,
    }
    # results_summary['OLS Post-2000 (Uninf. Prior)']
    mo.md(post_model.summary().tables[1].as_html())
    return X_post, ls_coeffs_post, post_nobs, sigma_sq_error_post


@app.cell
def _(mo):
    mo.md(r"""### Full estimation""")
    return


@app.cell
def _(df_full_sample, mo, results_summary, run_ols):
    # OLS for Full Sample
    (
        full_coeffs,
        full_std_errs,
        _,
        sigma_sq_error_full,
        X_full,
        Y_full,
        full_nobs,
        full_model,
    ) = run_ols(df_full_sample)
    results_summary["OLS Full Sample"] = {
        "b(1) Coeff": full_coeffs["avg_lag_inflation"],
        "b(1) SE": full_std_errs["avg_lag_inflation"],
        "a Coeff": full_coeffs["unemployment_rate_lag1"],
        "a SE": full_std_errs["unemployment_rate_lag1"],
        "N": full_nobs,
    }
    # results_summary['OLS Full Sample']
    mo.md(full_model.summary().tables[1].as_html())
    return


@app.cell
def _(mo):
    mo.md(r"""## Bayesian 'estimation'""")
    return


@app.cell
def _(
    X_post,
    ls_coeffs_post,
    post_nobs,
    prior_mean_coeffs,
    prior_vcov_coeffs,
    results_summary,
    run_bayesian_estimation,
    sigma_sq_error_post,
):
    # 4. Bayesian Estimations for different weights
    weights_on_prior = [0.5, 0.2, 0.05, 0.0]  # As in Kiley (2022) Table 2
    # weights_on_prior = [0.0]

    for w_val in weights_on_prior:
        bayes_coeffs, bayes_std_errs = run_bayesian_estimation(
            prior_mean_coeffs,
            prior_vcov_coeffs,
            ls_coeffs_post,
            X_post.values,
            sigma_sq_error_post,
            # full_coeffs,
            # X_full.values,
            # sigma_sq_error_full,
            w_val,
        )

        results_summary[f"Bayesian (w={w_val})"] = {
            "b(1) Coeff": bayes_coeffs[
                0
            ],  # Assuming order: avg_lag_inflation, unemployment_rate_lag1
            "b(1) SE": bayes_std_errs[0],
            "a Coeff": bayes_coeffs[1],
            "a SE": bayes_std_errs[1],
            "N": post_nobs,
        }

    return (weights_on_prior,)


@app.cell
def _(mo):
    mo.md(r"""## Present results""")
    return


@app.cell
def _(mo):
    mo.md(r"""### Table of results""")
    return


@app.cell
def _(df_post_2000, df_pre_2000, first_date, last_date, pd, results_summary):
    # 5. Results Presentation
    results_df = pd.DataFrame.from_dict(results_summary, orient="index")
    column_order = [
        "b(1) Coeff",
        "b(1) SE",
        "a Coeff",
        "a SE",
        "N",
    ]
    # No constant coefficients in this version
    results_df = results_df.reindex(columns=column_order)  # Use reindex for columns

    row_order = [
        "OLS Pre-2000 (Prior)",
        "OLS Post-2000 (Uninf. Prior)",
        "OLS Full Sample",
        "Bayesian (w=0.5)",
        "Bayesian (w=0.2)",
        "Bayesian (w=0.05)",
        "Bayesian (w=0.0)",
    ]
    existing_row_order = [row for row in row_order if row in results_df.index]
    results_df = results_df.reindex(index=existing_row_order)  # Use reindex for rows

    print(
        "Estimated Phillips Curve Coefficients and Standard Errors (Monthly Data Model WITHOUT Constant)"
    )
    print(
        "Model: delta_p(t) = b(1)*avg_lag_inflation(t) + a*unemployment_rate_lag1(t) + e(t)"
    )

    # ---

    print(
        f"Data range after processing: {first_date.strftime('%Y-%m-%d')} to {last_date.strftime('%Y-%m-%d')}"
    )
    print(
        f"Pre-2000 sample: {df_pre_2000.index.min().strftime('%Y-%m-%d')} to {df_pre_2000.index.max().strftime('%Y-%m-%d')}"
    )
    print(
        f"Post-1999 sample: {df_post_2000.index.min().strftime('%Y-%m-%d')} to {df_post_2000.index.max().strftime('%Y-%m-%d')}"
    )
    print("\n--- Results Summary ---")
    print(results_df.to_string(float_format="%.3f"))
    return column_order, results_df, row_order


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Estimated distributions

    Is it even useful to plot the estimated distributions?

    > Adrian says No
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Forecast
    Create the forecasts, and the plot
    """
    )
    return


@app.cell
def _(results_df):
    results_df
    return


@app.cell
def _(iterative_forecast, pd, regression_df, results_df, weights_on_prior):
    # Jan 2022. Then May 2023, then full data.
    end_of_history = pd.to_datetime("2021-12")
    forecast_steps = 24

    initial_lags_end = end_of_history
    initial_lags_start = end_of_history - pd.DateOffset(months=11)
    initial_inf_lags = regression_df.loc[initial_lags_start:initial_lags_end, "Y"]

    # Set our unemployment assumption (constant at last value)
    last_known_unemployment = regression_df.loc[
        end_of_history, "unemployment_rate_lag1"
    ]

    # Create the date range for the 12-month forecast
    forecast_dates = pd.date_range(
        start=end_of_history, periods=forecast_steps + 1, freq="MS"
    )[1:]

    # Create the unemployment Series for the forecast period
    unemployment_assumption = pd.Series(
        data=last_known_unemployment, index=forecast_dates
    )

    forecasts = {}
    for w_val2 in weights_on_prior:
        key = f"Bayesian (w={w_val2})"
        coeffs_to_use = results_df.loc[key][
            ["b(1) Coeff", "a Coeff"]  # No constant coefficient
        ].values
        fcst = iterative_forecast(
            coeffs_to_use, initial_inf_lags.values, unemployment_assumption
        )
        forecasts[key] = fcst

    return forecasts, initial_lags_end


@app.cell
def _(forecasts, initial_lags_end, np, pd, process_forecast, us_data):
    # Empty df to hold data
    all_forecasts_df = pd.DataFrame(
        index=us_data.index[us_data.index > initial_lags_end]
    )

    # Create the initial DataFrame with historical data
    base_df = pd.DataFrame(index=us_data.index)
    base_df["core_cpi"] = us_data["core_cpi"].loc[:initial_lags_end]
    base_df["core_cpi_log"] = np.log(base_df["core_cpi"])
    base_df["delta_p"] = base_df["core_cpi_log"].diff(1) * 100
    base_df["delta_p_fcst"] = base_df["delta_p"]

    # Fill in the rebased forecasts using the base data
    for bayes_key in forecasts:
        rebased_series = process_forecast(
            forecasts[bayes_key], base_df, initial_lags_end
        )
        all_forecasts_df[bayes_key] = rebased_series

    historical_data_for_plot = (
        us_data["core_cpi"].loc[pd.Timestamp("2018-01-01") : initial_lags_end].diff(12)
    )

    historical_data_for_plot.name = "core_cpi_12m"
    return all_forecasts_df, historical_data_for_plot


@app.cell
def _(all_forecasts_df, go, historical_data_for_plot, initial_lags_end, pd):
    fig_fcts = go.Figure()

    # 1. Add historical data
    fig_fcts.add_trace(
        go.Scatter(
            x=historical_data_for_plot.index,
            y=historical_data_for_plot.values,
            name="Historical Data",
            mode="lines",
        )
    )

    # 2. Add trace for each forecast_series
    for col_name in all_forecasts_df.columns:
        fig_fcts.add_trace(
            go.Scatter(
                x=all_forecasts_df.index,
                y=all_forecasts_df[col_name].values,
                name=col_name,
                mode="lines",
                line=dict(width=2.5, dash="dot"),
            )
        )

        # 3. Update layout
        fig_fcts.update_layout(
            title=dict(
                text="<b> Forecast for Core CPI Inflation using estimated Phillips curve by Bayesian Policymaker (No Constant)</b>",
                x=0.5,
                font=dict(size=22, family="Times New Roman"),
            ),
            yaxis_title="Year-over-Year Percent Change",
            yaxis_ticksuffix="%",
            template="plotly_white",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.0,
                xanchor="right",
                x=1,
                font=dict(size=12, family="Times New Roman"),
            ),
            hovermode="x unified",
        )

    fig_fcts.add_vrect(
        x0=initial_lags_end + pd.DateOffset(months=1),
        x1=max(all_forecasts_df[s].index[-1] for s in all_forecasts_df.columns),
        # x1=pd.Timestamp("2023-01-01"),
        annotation_text="Forecast Period",
        annotation_position="top right",
        fillcolor="green",
        opacity=0.1,
        line_width=0,
    )

    fig_fcts.show()

    return


@app.cell
def _(mo):
    mo.md(
        r"""
    # UK Version

    Replicate the same - but for the UK
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Load input data

    ### CPI
    Monthly index, 2015=100, non-seasonally adjusted, non-food non-energy, source: [OECD Data Explorer](https://data-explorer.oecd.org/vis?lc=en&pg=0&bp=true&snb=20&df%5Bds%5D=dsDisseminateFinalDMZ&df%5Bid%5D=DSD_PRICES@DF_PRICES_ALL&df%5Bag%5D=OECD.SDD.TPS&df%5Bvs%5D=1.0&tm=Inflation%20(CPI)&dq=GBR.M.N.CPI.IX._TXCP01_NRG..&lom=LASTNPERIODS&lo=2000&to%5BTIME_PERIOD%5D=false&vw=tl&lb=bt)

    ### Unemployment
    Unemployment rate, 16 and over, %, seasonally adjusted. Source: [ONS](https://www.ons.gov.uk/employmentandlabourmarket/peoplenotinwork/unemployment/timeseries/mgsx/lms)
    """
    )
    return


@app.cell
def _(pd):
    cpi_uk = pd.read_csv("data/OECD CPI Data GBR.csv")
    cpi_uk = cpi_uk[["TIME_PERIOD", "OBS_VALUE"]]
    cpi_uk.index = pd.to_datetime(cpi_uk["TIME_PERIOD"])
    cpi_uk = cpi_uk.drop(columns=["TIME_PERIOD"])
    cpi_uk.sort_index(inplace=True)
    cpi_uk.head()
    return (cpi_uk,)


@app.cell
def _(pd):
    unemp_uk = pd.read_csv("data/Unemployment Rate 16+ Seasonally Adjusted_CLEAN.csv")
    # unemp_uk = unemp_uk.drop(unemp_uk.index[0:7])
    unemp_uk.index = pd.to_datetime(unemp_uk["Date"])
    unemp_uk = unemp_uk.drop(columns=["Date"])
    unemp_uk.head()
    return (unemp_uk,)


@app.cell
def _(cpi_uk, pd, unemp_uk):
    GBR_data = pd.merge(
        cpi_uk, unemp_uk, how="inner", left_index=True, right_index=True
    )
    GBR_data.rename(
        columns={"OBS_VALUE": "core_cpi", "Value": "unemployment_rate"}, inplace=True
    )
    GBR_data.head()
    return (GBR_data,)


@app.cell
def _(pd):
    GBR_first_date = pd.Timestamp("1971-02-01")
    # GBR_last_date = pd.Timestamp("2019-12-31")
    GBR_last_date = pd.Timestamp("2025-02-01")
    GBR_pre_2000_end = pd.Timestamp("1999-12-31")
    GBR_post_2000_start = pd.Timestamp("2000-01-01")
    return GBR_first_date, GBR_last_date, GBR_pre_2000_end


@app.cell
def _(mo):
    mo.md(r"""## Validate input data""")
    return


@app.cell(hide_code=True)
def _(GBR_data, go, make_subplots):
    fig_uk = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=("Core CPI Prices", "Unemployment Rate"),
    )

    fig_uk.add_trace(
        go.Scatter(
            x=GBR_data.index,
            y=GBR_data["core_cpi"].pct_change(12),
            name="Core CPI",
        ),
        row=1,
        col=1,
    )

    fig_uk.add_trace(
        go.Scatter(
            x=GBR_data.index,
            y=GBR_data["unemployment_rate"],
            name="Unemployment Rate",
        ),
        row=2,
        col=1,
    )

    fig_uk.update_yaxes(
        title_text="Percent change from<br>12 months earlier", row=1, col=1
    )
    fig_uk.update_yaxes(title_text="Percent", row=2, col=1)

    fig_uk.update_layout(
        height=750,
        showlegend=False,
        title_text="Figure 1. CPI Inflation and the Civilian Unemployment Rate - GBR",
    )

    fig_uk.show()
    return


@app.cell
def _(GBR_data, prepare_data, px):
    GBR_regression_df = prepare_data(GBR_data)

    GBR_fig_input = px.line(
        GBR_regression_df,
        y=["Y", "avg_lag_inflation", "unemployment_rate_lag1"],
        title="Input Data for regression",
    )

    GBR_fig_input.update_layout(
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return (GBR_regression_df,)


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Seasonally adjust CPI
    
    We now use sophisticated seasonal adjustment methods in this order of preference:
    1. **X13-ARIMA-SEATS** (gold standard used by statistical agencies)
    2. **STL decomposition** (robust loess-based method)
    3. **Simple additive decomposition** (fallback)
    
    #### Setting up X13-ARIMA-SEATS (Optional but Recommended)
    
    To use X13-ARIMA-SEATS, you need to install the X13 binary:
    
    **macOS (via Homebrew):**
    ```bash
    brew install x13as
    ```
    
    **Windows:**
    - Download from [U.S. Census Bureau](https://www.census.gov/data/software/x13as.html)
    - Add to your PATH
    
    **Linux:**
    ```bash
    # Ubuntu/Debian
    sudo apt-get install x13as
    
    # Or download from Census Bureau and compile
    ```
    
    **Python Configuration:**
    ```python
    # If X13 is not in PATH, specify location:
    import statsmodels.api as sm
    sm.tsa.x13_path = "/path/to/x13as"
    ```
    
    The code will automatically detect if X13 is available and fall back gracefully if not.
    """
    )
    return


@app.cell
def _(GBR_regression_df, plt, sm, pd):
    def try_x13_seasonal_adjustment(series, freq="M"):
        """
        Attempts X13-ARIMA-SEATS seasonal adjustment with fallback to STL.

        Args:
            series: pandas Series with datetime index
            freq: frequency string for X13

        Returns:
            tuple: (seasonally_adjusted_series, seasonal_component, method_used)
        """
        try:
            # Convert to the format X13 expects
            x13_series = series.dropna()

            # Try X13-ARIMA-SEATS
            print("Attempting X13-ARIMA-SEATS seasonal adjustment...")

            # Run X13 with automatic model selection
            result = sm.tsa.x13_arima_analysis(
                x13_series,
                freq=freq,
                x12path=None,  # Uses default X13 path if available
                prefer_x13=True,
                log=False,  # Don't take log transformation (already in rate form)
                trading=True,  # Enable trading day adjustment
                outlier=True,  # Enable outlier detection
                automdl=True,  # Automatic ARIMA model selection
            )

            # Extract seasonally adjusted series and seasonal factors
            seasadj = result.seasadj
            seasonal = (
                result.irregular + result.trend - seasadj
            )  # Reconstruct seasonal component

            print("✓ Successfully applied X13-ARIMA-SEATS seasonal adjustment")
            print(f"Final model: {result.mdl}")

            return seasadj, seasonal, "X13-ARIMA-SEATS"

        except Exception as e:
            print(f"X13 adjustment failed: {str(e)}")
            print("Falling back to STL decomposition...")

            # Fallback to STL (Seasonal and Trend decomposition using Loess)
            # STL is more sophisticated than simple seasonal_decompose
            try:
                stl_result = sm.tsa.STL(
                    series.dropna(),
                    seasonal=13,  # Seasonal smoother parameter
                    period=12,  # 12 months
                    robust=True,  # Robust to outliers
                ).fit()

                seasadj = stl_result.observed - stl_result.seasonal
                seasonal = stl_result.seasonal

                print("✓ Successfully applied STL seasonal adjustment")
                return seasadj, seasonal, "STL"

            except Exception as stl_e:
                print(f"STL adjustment also failed: {str(stl_e)}")
                print("Falling back to simple additive decomposition...")

                # Final fallback to simple seasonal decomposition
                decomp = sm.tsa.seasonal_decompose(
                    series.dropna(), model="additive", period=12
                )
                seasadj = decomp.observed - decomp.seasonal
                seasonal = decomp.seasonal

                print("✓ Applied simple additive seasonal adjustment")
                return seasadj, seasonal, "Simple Additive"

    # Apply seasonal adjustment
    original_series = GBR_regression_df["Y"].copy()
    seasadj_series, seasonal_component, method_used = try_x13_seasonal_adjustment(
        original_series
    )

    # Update the dataframe with seasonally adjusted values
    # Align the series properly in case of any missing values
    aligned_seasadj = seasadj_series.reindex(GBR_regression_df.index)
    aligned_seasonal = seasonal_component.reindex(GBR_regression_df.index)

    # Update in place
    GBR_regression_df["Y"] = aligned_seasadj

    # Create enhanced diagnostic plots
    plt.style.use("seaborn-v0_8-whitegrid")
    fig_seas, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    fig_seas.suptitle(
        f"Seasonal Adjustment using {method_used}", fontsize=16, fontweight="bold"
    )

    # Plot 1: Before and After comparison
    original_series.plot(ax=ax1, label="Original", alpha=0.7, color="blue")
    aligned_seasadj.plot(
        ax=ax1, label="Seasonally Adjusted", alpha=0.8, color="red", linewidth=2
    )
    ax1.set_title("Original vs Seasonally Adjusted")
    ax1.set_ylabel("Inflation Rate (%)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Seasonal Component
    aligned_seasonal.plot(
        ax=ax2, label="Seasonal Component", color="green", linewidth=2
    )
    ax2.set_title("Extracted Seasonal Component")
    ax2.set_ylabel("Seasonal Factor")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Month-by-month seasonal pattern
    seasonal_monthly = aligned_seasonal.groupby(aligned_seasonal.index.month).mean()
    months = [
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "May",
        "Jun",
        "Jul",
        "Aug",
        "Sep",
        "Oct",
        "Nov",
        "Dec",
    ]
    ax3.bar(
        months, seasonal_monthly.values, color="lightblue", edgecolor="navy", alpha=0.7
    )
    ax3.set_title("Average Seasonal Pattern by Month")
    ax3.set_ylabel("Average Seasonal Effect")
    ax3.tick_params(axis="x", rotation=45)
    ax3.grid(True, alpha=0.3, axis="y")

    # Plot 4: Residual diagnostics (original - trend - seasonal)
    if method_used in ["X13-ARIMA-SEATS", "STL"]:
        residuals = original_series - aligned_seasadj - aligned_seasonal
        residuals.plot(ax=ax4, color="purple", alpha=0.7)
        ax4.set_title("Irregular Component (Residuals)")
        ax4.set_ylabel("Residual")
        ax4.grid(True, alpha=0.3)
    else:
        # For simple decomposition, show the difference between methods
        ax4.text(
            0.5,
            0.5,
            f"Method Used:\n{method_used}\n\nConsider installing X13-ARIMA-SEATS\nfor more sophisticated adjustment",
            ha="center",
            va="center",
            transform=ax4.transAxes,
            fontsize=12,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow"),
        )
        ax4.set_title("Seasonal Adjustment Information")

    plt.tight_layout()
    plt.show()

    # Print comprehensive diagnostics
    print(f"\n=== Seasonal Adjustment Results ({method_used}) ===")
    print(
        f"Original series range: {original_series.min():.3f} to {original_series.max():.3f}"
    )
    print(
        f"Seasonally adjusted range: {aligned_seasadj.min():.3f} to {aligned_seasadj.max():.3f}"
    )
    print(
        f"Seasonal component range: {aligned_seasonal.min():.3f} to {aligned_seasonal.max():.3f}"
    )
    print(f"Seasonal component std: {aligned_seasonal.std():.3f}")

    print("\nSeasonal Component Summary:")
    print(aligned_seasonal.describe())

    print("\nMonthly Seasonal Patterns:")
    monthly_pattern = aligned_seasonal.groupby(aligned_seasonal.index.month).mean()
    for month, effect in zip(months, monthly_pattern.values):
        print(f"  {month}: {effect:+.3f}")

    return method_used, aligned_seasonal


@app.cell(hide_code=True)
def _(
    GBR_data,
    GBR_first_date,
    GBR_last_date,
    GBR_pre_2000_end,
    last_date,
    pd,
    post_2000_start,
):
    def _():
        df = GBR_data.copy()
        df["core_cpi_1m_change"] = df["core_cpi"].diff(periods=1)
        df["core_cpi_12m_change"] = df["core_cpi"].diff(periods=12)

        full_sample = df.loc[GBR_first_date:GBR_last_date]
        pre_2000_sample = df.loc[GBR_first_date:GBR_pre_2000_end]
        post_1999_sample = df.loc[post_2000_start:last_date]

        summary_rows = [
            # Full Sample
            {
                "Sample": "Full sample",
                "Variable": "CPI inflation (percent)",
                "Observations": full_sample["core_cpi_1m_change"].count(),
                "Mean (annual rate)": round(
                    full_sample["core_cpi_12m_change"].mean(), 3
                ),
                "Std. Deviation": round(full_sample["core_cpi_1m_change"].std(), 3),
                "Auto-Correlation": round(
                    full_sample["core_cpi_1m_change"].autocorr(lag=1), 3
                ),
            },
            {
                "Sample": "Full sample",
                "Variable": "Unemployment rate (percent)",
                "Observations": "-",
                "Mean (annual rate)": round(full_sample["unemployment_rate"].mean(), 3),
                "Std. Deviation": round(full_sample["unemployment_rate"].std(), 3),
                "Auto-Correlation": round(
                    full_sample["unemployment_rate"].autocorr(lag=1), 3
                ),
            },
            # Pre-2000 Sample
            {
                "Sample": "Pre-2000 sample",
                "Variable": "CPI inflation (percent)",
                "Observations": pre_2000_sample["core_cpi_1m_change"].count(),
                "Mean (annual rate)": round(
                    pre_2000_sample["core_cpi_12m_change"].mean(), 3
                ),
                "Std. Deviation": round(pre_2000_sample["core_cpi_1m_change"].std(), 3),
                "Auto-Correlation": round(
                    pre_2000_sample["core_cpi_1m_change"].autocorr(lag=1), 3
                ),
            },
            {
                "Sample": "Pre-2000 sample",
                "Variable": "Unemployment rate (percent)",
                "Observations": "-",
                "Mean (annual rate)": round(
                    pre_2000_sample["unemployment_rate"].mean(), 3
                ),
                "Std. Deviation": round(pre_2000_sample["unemployment_rate"].std(), 3),
                "Auto-Correlation": round(
                    pre_2000_sample["unemployment_rate"].autocorr(lag=1), 3
                ),
            },
            # Post-1999 Sample
            {
                "Sample": "Post-1999 sample",
                "Variable": "CPI inflation (percent)",
                "Observations": post_1999_sample["core_cpi_1m_change"].count(),
                "Mean (annual rate)": round(
                    post_1999_sample["core_cpi_12m_change"].mean(), 3
                ),
                "Std. Deviation": round(
                    post_1999_sample["core_cpi_1m_change"].std(), 3
                ),
                "Auto-Correlation": round(
                    post_1999_sample["core_cpi_1m_change"].autocorr(lag=1), 3
                ),
            },
            {
                "Sample": "Post-1999 sample",
                "Variable": "Unemployment rate (percent)",
                "Observations": "-",
                "Mean (annual rate)": round(
                    post_1999_sample["unemployment_rate"].mean(), 3
                ),
                "Std. Deviation": round(post_1999_sample["unemployment_rate"].std(), 3),
                "Auto-Correlation": round(
                    post_1999_sample["unemployment_rate"].autocorr(lag=1), 3
                ),
            },
        ]

        summary_df = pd.DataFrame(summary_rows)

        final_rows = [
            # Full Sample Section
            {
                "Variable": "Full sample",
                "Observations": "",
                "Mean (annual rate)": "",
                "Std. Deviation": "",
                "Auto-Correlation": "",
            },
            *summary_df[summary_df["Sample"] == "Full sample"]
            .drop("Sample", axis=1)
            .to_dict("records"),
            # Pre-2000 Section
            {
                "Variable": "Pre-2000 sample",
                "Observations": "",
                "Mean (annual rate)": "",
                "Std. Deviation": "",
                "Auto-Correlation": "",
            },
            *summary_df[summary_df["Sample"] == "Pre-2000 sample"]
            .drop("Sample", axis=1)
            .to_dict("records"),
            # Post-1999 Section
            {
                "Variable": "Post-1999 sample",
                "Observations": "",
                "Mean (annual rate)": "",
                "Std. Deviation": "",
                "Auto-Correlation": "",
            },
            *summary_df[summary_df["Sample"] == "Post-1999 sample"]
            .drop("Sample", axis=1)
            .to_dict("records"),
        ]

        final_table = pd.DataFrame(final_rows)
        return final_table

    _()
    return


@app.cell
def _(mo):
    mo.md(r"""## Perform Least Squares Regression""")
    return


@app.cell
def _(GBR_regression_df, first_date, last_date, post_2000_start, pre_2000_end):
    GBR_df_pre_2000 = GBR_regression_df.loc[first_date:pre_2000_end]
    GBR_df_post_2000 = GBR_regression_df.loc[post_2000_start:last_date]
    GBR_df_full_sample = GBR_regression_df.loc[first_date:last_date]

    GBR_results_summary = {}
    return (
        GBR_df_full_sample,
        GBR_df_post_2000,
        GBR_df_pre_2000,
        GBR_results_summary,
    )


@app.cell
def _(mo):
    mo.md(r"""### Prior Estimation""")
    return


@app.cell
def _(GBR_df_pre_2000, GBR_results_summary, mo, run_ols):
    (
        GBR_prior_mean_coeffs,
        GBR_prior_std_errs,
        GBR_prior_vcov_coeffs,
        GBR_prior_sigma_sq_error,
        GBR_X_pre,
        GBR_Y_pre,
        GBR_prior_nobs,
        GBR_prior_model,
    ) = run_ols(GBR_df_pre_2000)

    GBR_results_summary["OLS Pre-2000 (Prior)"] = {
        "b(1) Coeff": GBR_prior_mean_coeffs["avg_lag_inflation"],
        "b(1) SE": GBR_prior_std_errs["avg_lag_inflation"],
        "a Coeff": GBR_prior_mean_coeffs["unemployment_rate_lag1"],
        "a SE": GBR_prior_std_errs["unemployment_rate_lag1"],
        "N": GBR_prior_nobs,
    }
    # GBR_results_summary['OLS Pre-2000 (Prior)']
    mo.md(GBR_prior_model.summary().tables[1].as_html())
    return GBR_prior_mean_coeffs, GBR_prior_vcov_coeffs


@app.cell
def _(mo):
    mo.md(r"""### Post-2000 estimation""")
    return


@app.cell
def _(GBR_df_post_2000, GBR_results_summary, mo, run_ols):
    # OLS for Likelihood (Post-2000 data) - also used for "Uninformative Prior"
    (
        GBR_ls_coeffs_post,
        GBR_ls_std_errs_post,
        _,
        GBR_sigma_sq_error_post,
        GBR_X_post,
        GBR_Y_post,
        GBR_post_nobs,
        GBR_post_model,
    ) = run_ols(GBR_df_post_2000)
    GBR_results_summary["OLS Post-2000 (Uninf. Prior)"] = {
        "b(1) Coeff": GBR_ls_coeffs_post["avg_lag_inflation"],
        "b(1) SE": GBR_ls_std_errs_post["avg_lag_inflation"],
        "a Coeff": GBR_ls_coeffs_post["unemployment_rate_lag1"],
        "a SE": GBR_ls_std_errs_post["unemployment_rate_lag1"],
        "N": GBR_post_nobs,
    }
    # GBR_results_summary['OLS Post-2000 (Uninf. Prior)']
    mo.md(GBR_post_model.summary().tables[1].as_html())
    return (
        GBR_X_post,
        GBR_ls_coeffs_post,
        GBR_post_nobs,
        GBR_sigma_sq_error_post,
    )


@app.cell
def _(mo):
    mo.md(r"""### Full estimation""")
    return


@app.cell
def _(GBR_df_full_sample, GBR_results_summary, mo, run_ols):
    # OLS for Full Sample
    (
        GBR_full_coeffs,
        GBR_full_std_errs,
        _,
        GBR_sigma_sq_error_full,
        GBR_X_full,
        GBR_Y_full,
        GBR_full_nobs,
        GBR_full_model,
    ) = run_ols(GBR_df_full_sample)
    GBR_results_summary["OLS Full Sample"] = {
        "b(1) Coeff": GBR_full_coeffs["avg_lag_inflation"],
        "b(1) SE": GBR_full_std_errs["avg_lag_inflation"],
        "a Coeff": GBR_full_coeffs["unemployment_rate_lag1"],
        "a SE": GBR_full_std_errs["unemployment_rate_lag1"],
        "N": GBR_full_nobs,
    }
    # GBR_results_summary['OLS Full Sample']
    mo.md(GBR_full_model.summary().tables[1].as_html())
    return


@app.cell
def _(mo):
    mo.md(r"""## Bayesian 'Estimation'""")
    return


@app.cell
def _(
    GBR_X_post,
    GBR_ls_coeffs_post,
    GBR_post_nobs,
    GBR_prior_mean_coeffs,
    GBR_prior_vcov_coeffs,
    GBR_results_summary,
    GBR_sigma_sq_error_post,
    run_bayesian_estimation,
):
    # 4. Bayesian Estimations for different weights
    GBR_weights_on_prior = [0.5, 0.2, 0.05, 0.0]  # As in Kiley (2022) Table 2
    # weights_on_prior = [0.0]

    for GBR_w_val in GBR_weights_on_prior:
        GBR_bayes_coeffs, GBR_bayes_std_errs = run_bayesian_estimation(
            GBR_prior_mean_coeffs,
            GBR_prior_vcov_coeffs,
            GBR_ls_coeffs_post,
            GBR_X_post.values,
            GBR_sigma_sq_error_post,
            # full_coeffs,
            # X_full.values,
            # sigma_sq_error_full,
            GBR_w_val,
        )

        GBR_results_summary[f"Bayesian (w={GBR_w_val})"] = {
            "b(1) Coeff": GBR_bayes_coeffs[
                0
            ],  # Assuming order: avg_lag_inflation, unemployment_rate_lag1
            "b(1) SE": GBR_bayes_std_errs[0],
            "a Coeff": GBR_bayes_coeffs[1],
            "a SE": GBR_bayes_std_errs[1],
            "N": GBR_post_nobs,
        }
    return (GBR_weights_on_prior,)


@app.cell
def _(mo):
    mo.md(r"""## Present results""")
    return


@app.cell
def _(mo):
    mo.md(r"""### Table of results""")
    return


@app.cell
def _(
    GBR_df_post_2000,
    GBR_df_pre_2000,
    GBR_first_date,
    GBR_last_date,
    GBR_results_summary,
    column_order,
    pd,
    row_order,
):
    # 5. Results Presentation
    GBR_results_df = pd.DataFrame.from_dict(GBR_results_summary, orient="index")
    GBR_column_order = [
        "b(1) Coeff",
        "b(1) SE",
        "a Coeff",
        "a SE",
        "N",
    ]
    # No constant coefficients in this version
    GBR_results_df = GBR_results_df.reindex(
        columns=column_order
    )  # Use reindex for columns

    GBR_row_order = [
        "OLS Pre-2000 (Prior)",
        "OLS Post-2000 (Uninf. Prior)",
        "OLS Full Sample",
        "Bayesian (w=0.5)",
        "Bayesian (w=0.2)",
        "Bayesian (w=0.05)",
        "Bayesian (w=0.0)",
    ]
    GBR_existing_row_order = [row for row in row_order if row in GBR_results_df.index]
    GBR_results_df = GBR_results_df.reindex(
        index=GBR_existing_row_order
    )  # Use reindex for rows

    print(
        "Estimated Phillips Curve Coefficients and Standard Errors (Monthly Data Model WITHOUT Constant)"
    )
    print(
        "Model: delta_p(t) = b(1)*avg_lag_inflation(t) + a*unemployment_rate_lag1(t) + e(t)"
    )

    # ---

    print(
        f"Data range after processing: {GBR_first_date.strftime('%Y-%m-%d')} to {GBR_last_date.strftime('%Y-%m-%d')}"
    )
    print(
        f"Pre-2000 sample: {GBR_df_pre_2000.index.min().strftime('%Y-%m-%d')} to {GBR_df_pre_2000.index.max().strftime('%Y-%m-%d')}"
    )
    print(
        f"Post-1999 sample: {GBR_df_post_2000.index.min().strftime('%Y-%m-%d')} to {GBR_df_post_2000.index.max().strftime('%Y-%m-%d')}"
    )
    print("\n--- Results Summary ---")
    print(GBR_results_df.to_string(float_format="%.3f"))
    return (GBR_results_df,)


@app.cell
def _(mo):
    mo.md(r"""## Forecast""")
    return


@app.cell
def _(
    GBR_regression_df,
    GBR_results_df,
    GBR_weights_on_prior,
    iterative_forecast,
    pd,
):
    # Jan 2022. Then May 2023, then full data.
    GBR_end_of_history = pd.to_datetime("2021-12")
    GBR_forecast_steps = 24

    GBR_initial_lags_end = GBR_end_of_history
    GBR_initial_lags_start = GBR_end_of_history - pd.DateOffset(months=11)
    GBR_initial_inf_lags = GBR_regression_df.loc[
        GBR_initial_lags_start:GBR_initial_lags_end, "Y"
    ]

    # Set our unemployment assumption (constant at last value)
    GBR_last_known_unemployment = GBR_regression_df.loc[
        GBR_end_of_history, "unemployment_rate_lag1"
    ]

    # Create the date range for the 12-month forecast
    GBR_forecast_dates = pd.date_range(
        start=GBR_end_of_history, periods=GBR_forecast_steps + 1, freq="MS"
    )[1:]

    # Create the unemployment Series for the forecast period
    GBR_unemployment_assumption = pd.Series(
        data=GBR_last_known_unemployment, index=GBR_forecast_dates
    )

    GBR_forecasts = {}
    for GBR_w_val2 in GBR_weights_on_prior:
        GBR_key = f"Bayesian (w={GBR_w_val2})"

        GBR_coeffs_to_use = GBR_results_df.loc[GBR_key][
            ["b(1) Coeff", "a Coeff"]  # No constant coefficient
        ].values
        GBR_fcst = iterative_forecast(
            GBR_coeffs_to_use, GBR_initial_inf_lags.values, GBR_unemployment_assumption
        )
        GBR_forecasts[GBR_key] = GBR_fcst

    return GBR_forecasts, GBR_initial_lags_end


@app.cell
def _(GBR_data, GBR_forecasts, GBR_initial_lags_end, np, pd, process_forecast):
    # Empty df to hold data
    GBR_all_forecasts_df = pd.DataFrame(
        index=GBR_data.index[GBR_data.index > GBR_initial_lags_end]
    )

    # Create the initial DataFrame with historical data
    GBR_base_df = pd.DataFrame(index=GBR_data.index)
    GBR_base_df["core_cpi"] = GBR_data["core_cpi"].loc[:GBR_initial_lags_end]
    GBR_base_df["core_cpi_log"] = np.log(GBR_base_df["core_cpi"])
    GBR_base_df["delta_p"] = GBR_base_df["core_cpi_log"].diff(1) * 100
    GBR_base_df["delta_p_fcst"] = GBR_base_df["delta_p"]

    # Fill in the rebased forecasts using the base data
    for GBR_bayes_key in GBR_forecasts:
        GBR_rebased_series = process_forecast(
            GBR_forecasts[GBR_bayes_key], GBR_base_df, GBR_initial_lags_end
        )
        GBR_all_forecasts_df[GBR_bayes_key] = GBR_rebased_series

    GBR_historical_data_for_plot = (
        GBR_data["core_cpi"]
        .loc[pd.Timestamp("2018-01-01") : GBR_initial_lags_end]
        .diff(12)
    )

    GBR_historical_data_for_plot.name = "core_cpi_12m"
    return GBR_all_forecasts_df, GBR_historical_data_for_plot


@app.cell
def _(
    GBR_all_forecasts_df,
    GBR_historical_data_for_plot,
    GBR_initial_lags_end,
    go,
    pd,
):
    GBR_fig_fcts = go.Figure()

    # 1. Add historical data
    GBR_fig_fcts.add_trace(
        go.Scatter(
            x=GBR_historical_data_for_plot.index,
            y=GBR_historical_data_for_plot.values,
            name="Historical Data",
            mode="lines",
        )
    )

    # 2. Add trace for each forecast_series
    for GBR_col_name in GBR_all_forecasts_df.columns:
        GBR_fig_fcts.add_trace(
            go.Scatter(
                x=GBR_all_forecasts_df.index,
                y=GBR_all_forecasts_df[GBR_col_name].values,
                name=GBR_col_name,
                mode="lines",
                line=dict(width=2.5, dash="dot"),
            )
        )

        # 3. Update layout
        GBR_fig_fcts.update_layout(
            title=dict(
                text="<b> Forecast for Core CPI Inflation using estimated Phillips curve by Bayesian Policymaker (No Constant)</b>",
                x=0.5,
                font=dict(size=22, family="Times New Roman"),
            ),
            yaxis_title="Year-over-Year Percent Change",
            yaxis_ticksuffix="%",
            template="plotly_white",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.0,
                xanchor="right",
                x=1,
                font=dict(size=12, family="Times New Roman"),
            ),
            hovermode="x unified",
        )

    GBR_fig_fcts.add_vrect(
        x0=GBR_initial_lags_end + pd.DateOffset(months=1),
        x1=max(GBR_all_forecasts_df[s].index[-1] for s in GBR_all_forecasts_df.columns),
        # x1=pd.Timestamp("2023-01-01"),
        annotation_text="Forecast Period",
        annotation_position="top right",
        fillcolor="green",
        opacity=0.1,
        line_width=0,
    )

    GBR_fig_fcts.show()
    return


if __name__ == "__main__":
    app.run()
