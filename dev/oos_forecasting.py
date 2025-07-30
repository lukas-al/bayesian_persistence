"""
This module contains functions for running an out-of-sample forecasting exercise
for the Bayesian policymaker model.
"""

from collections import deque

import numpy as np
import pandas as pd
import patsy
import statsmodels.api as sm
from scipy.linalg import inv as scipy_inv
from tqdm.auto import tqdm


def calculate_rmse(forecast, actual):
    """
    Calculates the Root Mean Squared Error (RMSE) between forecast and actual values.
    """
    # Ensure alignment and drop NaNs
    combined = pd.concat(
        [forecast.rename("forecast"), actual.rename("actual")], axis=1
    ).dropna()
    if combined.empty:
        return np.nan
    return np.sqrt(((combined["forecast"] - combined["actual"]) ** 2).mean())


def prepare_basic_data(
    df_input, cpi_col_name="core_cpi", unemployment_col_name="unemployment_rate"
):
    """
    Prepares the data for Phillips Curve estimation.
    Calculates inflation, lagged unemployment, and creates dependent variable.
    """
    df = df_input.copy()
    df = df.sort_index()

    # Calculate Core CPI inflation: monthly log difference, as a percentage
    df["core_cpi_log"] = np.log(df[cpi_col_name])
    df["delta_p"] = df["core_cpi_log"].diff(1) * 100

    # Create lagged unemployment rate: u(t-1)
    df["unemployment_rate"] = df[unemployment_col_name]

    return df


def run_ols_formula(df_sample, y_col, x_cols):
    """
    Runs OLS regression and returns key statistics.
    """
    Y = df_sample[y_col]
    X = df_sample[x_cols]
    X = sm.add_constant(X, has_constant="skip", prepend=False)

    model = sm.OLS(Y, X, hasconst=True)
    results = model.fit()

    return (
        results.params,
        results.bse,
        results.cov_params(),
        results.mse_resid,
        X,
        Y,
        results.nobs,
        results,
    )


def run_bayesian_estimation(gamma_prior, V_prior, gamma_ls, X_post, sigma_sq_post, w):
    """
    Performs Bayesian estimation based on Equation 4 of Kiley (2023).
    """
    # Prior precision (V^-1)
    inv_V_prior = scipy_inv(V_prior)

    # Data precision (σ^-2 * X'X)
    data_precision = (1 / sigma_sq_post) * (X_post.T @ X_post)

    # Posterior variance-covariance matrix: (w*V^-1 + (1-w)*σ^-2*X'X)^-1
    posterior_vcov = scipy_inv((w * inv_V_prior) + ((1 - w) * data_precision))

    # Posterior mean
    prior_component = w * inv_V_prior @ gamma_prior
    data_component = (1 - w) * data_precision @ gamma_ls
    posterior_mean = posterior_vcov @ (prior_component + data_component)

    # Standard errors
    posterior_std_errs = np.sqrt(np.diag(posterior_vcov))

    return posterior_mean, posterior_std_errs


def iterative_forecast(coeffs, initial_inf_lags, unemployment_lags):
    """
    Generates multi-step ahead forecasts for inflation.
    """
    num_lags = len(initial_inf_lags)
    inflation_lags = deque(initial_inf_lags, maxlen=num_lags)
    forecasts = {}

    inf_coeff, unemp_coeff, const_coeff = coeffs

    for date, u_lag in unemployment_lags.items():
        avg_inf_lag = np.mean(inflation_lags)
        forecast = (inf_coeff * avg_inf_lag) + (unemp_coeff * u_lag) + const_coeff

        forecasts[date] = forecast
        inflation_lags.append(forecast)

    return pd.Series(forecasts, name="inflation_forecast")


def process_forecast(forecast_series, base_df, initial_lags_end):
    """
    Processes a forecast series to convert back to level terms.
    """
    # Get historical CPI levels up to the forecast start date
    cpi_history = base_df["core_cpi"].loc[:initial_lags_end].copy()

    # Get the last known log CPI value to start the forecast from
    last_log_cpi = np.log(cpi_history.iloc[-1])

    # Convert forecasted monthly inflation rate (%) to log differences
    log_diffs_fcst = forecast_series / 100

    # Cumulatively sum the log differences and add to the last known log level
    log_cpi_fcst = log_diffs_fcst.cumsum() + last_log_cpi

    # Convert log-level forecast back to level forecast
    cpi_fcst = np.exp(log_cpi_fcst)

    # Combine historical and forecasted CPI levels
    full_cpi_series = pd.concat([cpi_history, cpi_fcst])
    full_cpi_series.name = "core_cpi_fcst"

    # Calculate 12-month percentage change
    final_yoy_forecast = full_cpi_series.pct_change(12) * 100

    return final_yoy_forecast.loc[final_yoy_forecast.index > initial_lags_end]


def run_oos_exercise(
    data,
    formula,
    forecast_horizon,
    oos_start_date,
    oos_end_date,
    prior_start_date,
    prior_end_date,
    weights,
):
    """
    Main function to run the out-of-sample forecasting exercise.
    """
    oos_forecasts = {}
    oos_rmses = {}

    # Generate the date range for the OOS exercise
    oos_dates = pd.date_range(start=oos_start_date, end=oos_end_date, freq="MS")

    for t in tqdm(oos_dates):
        print(f"Running forecast for origin: {t.strftime('%Y-%m')}")

        # 1. Prepare data for this iteration
        historical_data = data.loc[:t]

        # 2. Create design matrices
        Y, X = patsy.dmatrices(formula, data=historical_data, return_type="dataframe")
        df_reg = pd.concat([Y, X.iloc[:, 1:]], axis=1).dropna()
        df_reg["const"] = 1.0

        y_col = Y.columns[0]
        x_cols = X.columns[1:].tolist() + ["const"]

        df_prior = df_reg.loc[prior_start_date:prior_end_date]
        df_likelihood = df_reg.loc[prior_end_date:t]

        # 3. Estimate prior and likelihood models
        try:
            (prior_coeffs, _, prior_vcov, _, _, _, _, _) = run_ols_formula(
                df_prior, y_col, x_cols
            )
            (like_coeffs, _, _, like_sigma_sq, X_like, _, _, _) = run_ols_formula(
                df_likelihood, y_col, x_cols
            )
        except Exception as e:
            print(f"  Estimation failed for {t}: {e}")
            continue

        oos_forecasts[t] = {}
        oos_rmses[t] = {}

        # 4. Generate forecasts for each weight
        for w in weights:
            print(f"  Calculating posterior for weight: {w}")
            bayes_coeffs, _ = run_bayesian_estimation(
                prior_coeffs.values,
                prior_vcov.values,
                like_coeffs.values,
                X_like.values,
                like_sigma_sq,
                w,
            )

            # Prepare for forecasting
            initial_lags_end = t
            initial_lags_start = t - pd.DateOffset(months=11)
            initial_inf_lags = historical_data.loc[
                initial_lags_start:initial_lags_end, "delta_p"
            ]

            last_known_unemployment = historical_data.loc[t, "unemployment_rate"]
            forecast_dates = pd.date_range(
                start=t, periods=forecast_horizon + 1, freq="MS"
            )[1:]
            unemployment_assumption = pd.Series(
                data=last_known_unemployment, index=forecast_dates
            )

            # Create base dataframe for processing forecasts
            base_df = pd.DataFrame(index=data.index)
            base_df["core_cpi"] = data["core_cpi"].loc[:initial_lags_end]
            base_df["core_cpi_log"] = np.log(base_df["core_cpi"])

            # Generate and process forecast
            fcst_series = iterative_forecast(
                bayes_coeffs, initial_inf_lags.values, unemployment_assumption
            )
            processed_fcst = process_forecast(fcst_series, base_df, initial_lags_end)

            oos_forecasts[t][w] = processed_fcst

            # 5. Calculate RMSE
            actual_data = data["core_cpi"].pct_change(12) * 100
            rmse = calculate_rmse(processed_fcst, actual_data)
            oos_rmses[t][w] = rmse

    return oos_forecasts, oos_rmses
