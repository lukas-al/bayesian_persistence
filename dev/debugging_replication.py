import marimo

__generated_with = "0.13.11"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd
    import statsmodels.api as sm
    import plotly.express as px
    from scipy.linalg import inv as scipy_inv
    from pprint import pprint

    return mo, np, pd, px, scipy_inv, sm


@app.cell
def _(mo):
    mo.md(
        r"""
    # Debugging the replication

    I can't get the existing to replicate. This notebook walks through the different components which aren't working, and figures out how to replicate.
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Replicating again - simplified
    Replicate the paper again. See if I can get the same results.
    """
    )
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
    mo.md(r"""### Prepare Data""")
    return


@app.cell
def _(np, sm):
    def prepare_data(df_input):
        """
        Prepares the data for Phillips Curve estimation.
        Calculates inflation, lagged unemployment, and average lagged inflation.
        """
        df = df_input.copy()
        df = df.sort_index()  # Ensure data is sorted by date

        # Calculate Core CPI inflation: 12-month log difference, as a percentage
        # delta_p(t) = (ln(CPI_t) - ln(CPI_{t-12})) * 100
        df["core_cpi_log"] = np.log(df["core_cpi"])
        df["delta_p"] = df["core_cpi_log"].diff(1) * 100

        # Create lagged unemployment rate: u(t-1)
        df["unemployment_rate_lag1"] = np.log(df["unemployment_rate"]).shift(1)

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

    def run_ols(
        df_sample, y_col="Y", x_cols=["avg_lag_inflation", "unemployment_rate_lag1"]
    ):
        """
        Runs OLS regression and returns key statistics.
        Model: Y = b1*X1 + b2*X2 (no constant, as per paper's Equation 1)
        """
        Y = df_sample[y_col]
        X = df_sample[x_cols]
        X = sm.add_constant(X, has_constant="skip", prepend=False)

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

    return prepare_data, run_ols


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

        # Checking edge case at 0
        if w == 0.0:
            print("--- CHECKING EDGE CASE AT W = 0 --- ")
            print("Posterior_vcov:", posterior_vcov)
            print("Prior Component:", prior_component)
            print("Data component:", data_component)
            print("Posterior x Data and Gamma_ls")
            print(posterior_vcov @ data_component, gamma_ls.values)

        return posterior_mean, posterior_std_errs

    return (run_bayesian_estimation,)


@app.cell
def _(us_data):
    print(us_data.head())
    return


@app.cell
def _(prepare_data, px, us_data):
    regression_df = prepare_data(us_data)

    px.line(
        regression_df,
        y=["Y", "avg_lag_inflation", "unemployment_rate_lag1"],
        title="Input Data",
    )
    return (regression_df,)


@app.cell
def _(pd, regression_df):
    first_date = regression_df.index.min()
    last_date = pd.Timestamp("2019-12-31")
    pre_2000_end = pd.Timestamp("1999-12-31")
    post_2000_start = pd.Timestamp("2000-01-01")
    return first_date, last_date, post_2000_start, pre_2000_end


@app.cell
def _(first_date, last_date, post_2000_start, pre_2000_end, regression_df):
    df_pre_2000 = regression_df.loc[first_date:pre_2000_end]
    df_post_2000 = regression_df.loc[post_2000_start:last_date]
    df_full_sample = regression_df.loc[first_date:last_date]
    return df_full_sample, df_post_2000, df_pre_2000


@app.cell
def _():
    results_summary = {}
    return (results_summary,)


@app.cell
def _(mo):
    mo.md(r"""### Estimate pre-2000 sample""")
    return


@app.cell
def _(df_pre_2000, results_summary, run_ols):
    (
        prior_mean_coeffs,
        prior_std_errs,
        prior_vcov_coeffs,
        prior_sigma_sq_error,
        X_pre,
        Y_pre,
        prior_nobs,
        prior_results,
    ) = run_ols(df_pre_2000)

    results_summary["OLS Pre-2000 (Prior)"] = {
        "b(1) Coeff": prior_mean_coeffs["avg_lag_inflation"],
        "b(1) SE": prior_std_errs["avg_lag_inflation"],
        "a Coeff": prior_mean_coeffs["unemployment_rate_lag1"],
        "a SE": prior_std_errs["unemployment_rate_lag1"],
        "N": prior_nobs,
    }
    results_summary["OLS Pre-2000 (Prior)"]
    return prior_mean_coeffs, prior_vcov_coeffs


@app.cell
def _(mo):
    mo.md(r"""### Estimate post 2000 sample""")
    return


@app.cell
def _(df_post_2000, results_summary, run_ols):
    # OLS for Likelihood (Post-2000 data) - also used for "Uninformative Prior"
    (
        ls_coeffs_post,
        ls_std_errs_post,
        _,
        sigma_sq_error_post,
        X_post,
        Y_post,
        post_nobs,
        _,
    ) = run_ols(df_post_2000)
    results_summary["OLS Post-2000 (Uninf. Prior)"] = {
        "b(1) Coeff": ls_coeffs_post["avg_lag_inflation"],
        "b(1) SE": ls_std_errs_post["avg_lag_inflation"],
        "a Coeff": ls_coeffs_post["unemployment_rate_lag1"],
        "a SE": ls_std_errs_post["unemployment_rate_lag1"],
        "N": post_nobs,
    }
    results_summary["OLS Post-2000 (Uninf. Prior)"]
    return X_post, ls_coeffs_post, post_nobs, sigma_sq_error_post


@app.cell
def _(mo):
    mo.md(r"""### Estimate full sample""")
    return


@app.cell
def _(df_full_sample, results_summary, run_ols):
    # OLS for Full Sample
    full_coeffs, full_std_errs, _, sigma_sq_error_full, X_full, Y_full, full_nobs, _ = (
        run_ols(df_full_sample)
    )
    results_summary["OLS Full Sample"] = {
        "b(1) Coeff": full_coeffs["avg_lag_inflation"],
        "b(1) SE": full_std_errs["avg_lag_inflation"],
        "a Coeff": full_coeffs["unemployment_rate_lag1"],
        "a SE": full_std_errs["unemployment_rate_lag1"],
        "N": full_nobs,
    }
    results_summary["OLS Full Sample"]
    return


@app.cell
def _(mo):
    mo.md(r"""### Estimate Bayesian""")
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

        print(bayes_coeffs)

    return


@app.cell
def _(mo):
    mo.md(r"""### Print results""")
    return


@app.cell
def _():
    # pprint(results_summary['OLS Post-2000 (Uninf. Prior)'])
    # # results_summary
    # pprint(results_summary["Bayesian (w=0.0)"])
    return


@app.cell
def _(df_post_2000, df_pre_2000, first_date, last_date, pd, results_summary):
    # 5. Results Presentation
    results_df = pd.DataFrame.from_dict(results_summary, orient="index")
    column_order = ["b(1) Coeff", "b(1) SE", "a Coeff", "a SE", "N"]
    # Add 'const Coeff' and 'const SE' to column_order if you uncomment their display
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
        "Estimated Phillips Curve Coefficients and Standard Errors (Monthly Data Model with Constant)"
    )
    print(
        "Model: delta_p(t) = const + b(1)*avg_lag_inflation(t) + a*unemployment_rate_lag1(t) + e(t)"
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
    return


if __name__ == "__main__":
    app.run()
