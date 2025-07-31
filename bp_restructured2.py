import marimo

__generated_with = "0.14.13"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Bayesian Policy Maker
    How much information should a policy maker take from pre-2000 inflation as regards persistence in the post-2000 era.

    This note replicates the methodology outlined in [Kiley](https://www.ijcb.org/journal/ijcb24q1a6.pdf), extending it to UK data.

    Restructured for the final time...
    """
    )
    return


@app.cell
def _(mo):
    mo.md(r"""## Shared functions""")
    return


@app.cell
def _():
    from collections import deque

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import patsy
    import plotly.graph_objects as go
    import plotly.express as px
    import statsmodels.api as sm
    from plotly.subplots import make_subplots
    from scipy.linalg import inv as scipy_inv

    return deque, go, make_subplots, mo, np, patsy, pd, plt, px, scipy_inv, sm


@app.cell
def _(np):
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

    return (prepare_basic_data,)


@app.cell
def _(pd):
    # Patsy transformation functions for Phillips curve specifications
    def lag(series, n=1):
        """Simple lag function: lag(variable, n)"""
        if isinstance(series, pd.Series):
            return series.shift(n)
        elif isinstance(series, pd.DataFrame):
            return series.shift(n)
        else:
            return pd.Series(series).shift(n)

    def avg_lag(series, start_lag, end_lag):
        """Average of lags from start_lag to end_lag: avg_lag(variable, 1, 12)"""
        if isinstance(series, (pd.Series, pd.DataFrame)):
            lagged_series = [series.shift(i) for i in range(start_lag, end_lag + 1)]
        else:
            series = pd.Series(series)
            lagged_series = [series.shift(i) for i in range(start_lag, end_lag + 1)]

        return pd.concat(lagged_series, axis=1).mean(axis=1)

    def rolling_avg_lag(series, window, lag=1):
        """Rolling average starting from lag: rolling_avg_lag(variable, 12, 1)"""
        if isinstance(series, (pd.Series, pd.DataFrame)):
            return series.shift(lag).rolling(window=window).mean()
        else:
            return pd.Series(series).shift(lag).rolling(window=window).mean()

    return


@app.cell
def _(sm):
    def run_ols_formula(df_sample, y_col, x_cols):
        """
        Runs OLS regression and returns key statistics.
        """
        Y = df_sample[y_col]
        X = df_sample[x_cols]
        X = sm.add_constant(X, has_constant="skip", prepend=False)

        model = sm.OLS(Y, X)
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

    return (run_ols_formula,)


@app.cell
def _(np, scipy_inv):
    def run_bayesian_estimation(
        gamma_prior, V_prior, gamma_ls, X_post, sigma_sq_post, w
    ):
        """
        Performs Bayesian estimation based on Equation 4 of Kiley (2023).
        """
        print(f"Calculating posterior for w={w}")

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

    return (run_bayesian_estimation,)


@app.cell
def _(deque, np, pd):
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
            # forecast = (inf_coeff * avg_inf_lag) + (unemp_coeff * u_lag)

            forecasts[date] = forecast
            inflation_lags.append(forecast)

        return pd.Series(forecasts, name="inflation_forecast")

    return (iterative_forecast,)


@app.cell
def _(np, pd):
    def process_forecast(forecast_series, base_df, initial_lags_end):
        """
        Processes a forecast series to convert back to level terms.
        """
        df = base_df.copy()

        forecast_series.name = "delta_p_fcst"
        df = forecast_series.to_frame().combine_first(df)

        # Convert from log-differences back to level
        df["delta_p_fcst"] = df["delta_p_fcst"] / 100
        df["delta_p_fcst"] = df["delta_p_fcst"].fillna(df["core_cpi_log"])
        df["delta_p_fcst"] = df["delta_p_fcst"].cumsum()
        df["delta_p_fcst"] = np.exp(df["delta_p_fcst"])

        # Convert to 12-month change
        df["delta_p_fcst"] = df["delta_p_fcst"].diff(12)

        return df["delta_p_fcst"].loc[df.index > initial_lags_end]

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

    return (process_forecast,)


@app.cell
def _(result, sm):
    def try_seasonal_adjustment(series, freq="M"):
        """
        Attempts seasonal adjustment with fallback methods.
        """
        try:
            print("Attempting X13 seasonal adjustment")
            x13_result = sm.tsa.x13_arima_analysis(
                series.dropna(),
                freq=freq,
                prefer_x13=True,
                log=False,
                trading=True,
                outlier=True,
            )
            seasadj = result.seasadj
            seasonal = result.irregular + result.trend - seasadj
            print("Successfully applied X13/X12 seasonal adjustment")
            return seasadj, seasonal, "X13"

        except Exception as e:
            print(f"X13 adjustment failed: {str(e)}")
            print("Falling back to STL seasonal adjustment...")
            stl_result = sm.tsa.STL(
                series.dropna(),
                seasonal=13,
                period=12,
                robust=True,
            ).fit()

            seasadj = stl_result.observed - stl_result.seasonal
            seasonal = stl_result.seasonal

            print("✓ Successfully applied STL seasonal adjustment")
            return seasadj, seasonal, "STL"

    return (try_seasonal_adjustment,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""#""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Load data""")
    return


@app.cell
def _(pd):
    # Load US data
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
    us_data.set_index("observation_date", inplace=True)

    print(f"US data loaded: {len(us_data)} observations")
    return (us_data,)


@app.cell
def _(pd):
    # Load UK data
    cpi_uk = pd.read_csv("data/OECD CPI Data GBR.csv")
    cpi_uk = cpi_uk[["TIME_PERIOD", "OBS_VALUE"]]
    cpi_uk.index = pd.to_datetime(cpi_uk["TIME_PERIOD"])
    cpi_uk = cpi_uk.drop(columns=["TIME_PERIOD"]).sort_index()

    unemp_uk = pd.read_csv("data/Unemployment Rate 16+ Seasonally Adjusted_CLEAN.csv")
    unemp_uk.index = pd.to_datetime(unemp_uk["Date"])
    unemp_uk = unemp_uk.drop(columns=["Date"])

    uk_data = pd.merge(cpi_uk, unemp_uk, how="inner", left_index=True, right_index=True)
    uk_data.rename(
        columns={"OBS_VALUE": "core_cpi", "Value": "unemployment_rate"},
        inplace=True,
    )

    print(f"UK data loaded: {len(uk_data)} observations")
    return (uk_data,)


@app.cell
def _(go, make_subplots, uk_data, us_data):
    # Visualise raw data
    def _():
        # Create visualization plots for both countries
        fig = make_subplots(
            rows=2,
            cols=2,
            shared_xaxes=True,
            vertical_spacing=0.1,
            horizontal_spacing=0.1,
            subplot_titles=(
                "US: Core CPI",
                "UK: Core CPI (NSA)",
                "US: Unemployment Rate",
                "UK: Unemployment Rate",
            ),
        )

        # US inflation
        fig.add_trace(
            go.Scatter(
                x=us_data.index,
                y=us_data["core_cpi"],
                name="US CPI",
                line=dict(color="blue"),
            ),
            row=1,
            col=1,
        )

        # UK inflation (seasonally adjusted)
        fig.add_trace(
            go.Scatter(
                x=uk_data.index,
                y=uk_data["core_cpi"],
                name="UK CPI (NSA)",
                line=dict(color="red"),
            ),
            row=1,
            col=2,
        )

        # US unemployment
        fig.add_trace(
            go.Scatter(
                x=us_data.index,
                y=us_data["unemployment_rate"],
                name="US Unemployment",
                line=dict(color="green"),
            ),
            row=2,
            col=1,
        )

        # UK unemployment
        fig.add_trace(
            go.Scatter(
                x=uk_data.index,
                y=uk_data["unemployment_rate"],
                name="UK Unemployment",
                line=dict(color="orange"),
            ),
            row=2,
            col=2,
        )

        # Update layout
        fig.update_yaxes(title_text="Monthly % Change", row=1, col=1)
        fig.update_yaxes(title_text="Monthly % Change", row=1, col=2)
        fig.update_yaxes(title_text="Percent", row=2, col=1)
        fig.update_yaxes(title_text="Percent", row=2, col=2)

        fig.update_layout(
            height=600,
            showlegend=False,
            title_text="Raw Input Data for Estimation",
        )
        return fig.show()

    _()
    return


@app.cell
def _(pd):
    # Define date ranges for analysis
    first_date = pd.Timestamp("1958-01-01")
    last_date = pd.Timestamp("2019-12-31")
    pre_2000_end = pd.Timestamp("1999-12-31")
    post_2000_start = pd.Timestamp("2000-01-01")

    # UK-specific dates (start later due to data availability)
    uk_first_date = pd.Timestamp("1971-02-01")
    uk_last_date = pd.Timestamp("2019-12-31")
    return first_date, last_date, post_2000_start, pre_2000_end


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Clean data""")
    return


@app.cell
def _(prepare_basic_data, uk_data, us_data):
    # Prepare US data for analysis
    us_clean = prepare_basic_data(us_data)
    print(f"US data prepared: {len(us_clean)} observations")
    print(f"Date range: {us_clean.index.min()} to {us_clean.index.max()}")

    # Prepare UK data for analysis
    uk_clean = prepare_basic_data(uk_data)
    print(f"\nUK data prepared: {len(uk_clean)} observations")
    print(f"Date range: {uk_clean.index.min()} to {uk_clean.index.max()}")
    return uk_clean, us_clean


@app.cell
def _(plt, try_seasonal_adjustment, uk_clean):
    def _():
        # Seasonally adjust the UK data (UK CPI is not seasonally adjusted, US CPI is already SA)
        original_series = uk_clean["delta_p"].copy()
        seasadj_series, seasonal_component, method_used = try_seasonal_adjustment(
            original_series
        )

        # Update UK data with seasonally adjusted values
        aligned_seasadj = seasadj_series.reindex(uk_clean.index)
        aligned_seasonal = seasonal_component.reindex(uk_clean.index)

        # Create a copy for seasonally adjusted data
        uk_clean_sa = uk_clean.copy()
        uk_clean_sa["delta_p"] = aligned_seasadj

        # Create diagnostic plot
        plt.style.use("default")
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

        # Plot original vs seasonally adjusted
        original_series.plot(ax=ax1, label="Original", alpha=0.7, color="blue")
        aligned_seasadj.plot(
            ax=ax1, label="Seasonally Adjusted", alpha=0.8, color="red", linewidth=2
        )
        ax1.set_title(f"UK Inflation: Original vs Seasonally Adjusted ({method_used})")
        ax1.set_ylabel("Monthly Inflation Rate (%)")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot seasonal component
        aligned_seasonal.plot(
            ax=ax2, label="Seasonal Component", color="green", linewidth=2
        )
        ax2.set_title("Extracted Seasonal Component")
        ax2.set_ylabel("Seasonal Factor")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        print(f"\nSeasonal adjustment completed using {method_used}")
        print(f"Original series std: {original_series.std():.3f}")
        return uk_clean_sa, print(
            f"Seasonally adjusted std: {aligned_seasadj.std():.3f}"
        )

    uk_clean_sa, _ = _()
    return (uk_clean_sa,)


@app.cell
def _(mo):
    mo.md(r"""## Visualise data""")
    return


@app.cell
def _(go, make_subplots, uk_clean_sa, us_clean):
    def _(uk_clean_sa):
        # Create visualization plots for both countries
        fig = make_subplots(
            rows=2,
            cols=2,
            shared_xaxes=True,
            vertical_spacing=0.1,
            horizontal_spacing=0.1,
            subplot_titles=(
                "US: Core CPI Inflation",
                "UK: Core CPI Inflation (SA)",
                "US: Unemployment Rate",
                "UK: Unemployment Rate",
            ),
        )

        # US inflation
        fig.add_trace(
            go.Scatter(
                x=us_clean.index,
                y=us_clean["delta_p"],
                name="US CPI Inflation",
                line=dict(color="blue"),
            ),
            row=1,
            col=1,
        )

        # UK inflation (seasonally adjusted)
        fig.add_trace(
            go.Scatter(
                x=uk_clean_sa.index,
                y=uk_clean_sa["delta_p"],
                name="UK CPI Inflation (SA)",
                line=dict(color="red"),
            ),
            row=1,
            col=2,
        )

        # US unemployment
        fig.add_trace(
            go.Scatter(
                x=us_clean.index,
                y=us_clean["unemployment_rate"],
                name="US Unemployment",
                line=dict(color="green"),
            ),
            row=2,
            col=1,
        )

        # UK unemployment
        fig.add_trace(
            go.Scatter(
                x=uk_clean_sa.index,
                y=uk_clean_sa["unemployment_rate"],
                name="UK Unemployment",
                line=dict(color="orange"),
            ),
            row=2,
            col=2,
        )

        # Update layout
        fig.update_yaxes(title_text="Monthly % Change", row=1, col=1)
        fig.update_yaxes(title_text="Monthly % Change", row=1, col=2)
        fig.update_yaxes(title_text="Percent", row=2, col=1)
        fig.update_yaxes(title_text="Percent", row=2, col=2)

        fig.update_layout(
            height=600,
            showlegend=False,
            title_text="Input Data for Phillips Curve Estimation",
        )
        return fig.show()

    _(uk_clean_sa)
    return


@app.cell
def _(
    first_date,
    last_date,
    mo,
    pd,
    post_2000_start,
    pre_2000_end,
    uk_clean_sa,
    us_clean,
):
    # Create summary statistics table
    def create_summary_stats(data, name, start_date, end_date):
        """Create summary statistics for a dataset"""
        sample_data = data.loc[start_date:end_date]

        return {
            "Country": name,
            "Observations": len(sample_data.dropna()),
            "Inflation Mean": f"{sample_data['delta_p'].mean():.3f}",
            "Inflation Std": f"{sample_data['delta_p'].std():.3f}",
            "Unemployment Mean": f"{sample_data['unemployment_rate'].mean():.3f}",
            "Unemployment Std": f"{sample_data['unemployment_rate'].std():.3f}",
        }

    # Create summary statistics
    summary_data = []

    # Full sample statistics
    summary_data.extend(
        [
            create_summary_stats(us_clean, "US", first_date, last_date),
            create_summary_stats(uk_clean_sa, "UK", first_date, last_date),
        ]
    )

    # Pre-2000 statistics
    summary_data.extend(
        [
            create_summary_stats(us_clean, "US (Pre-2000)", first_date, pre_2000_end),
            create_summary_stats(
                uk_clean_sa, "UK (Pre-2000)", first_date, pre_2000_end
            ),
        ]
    )

    # Post-2000 statistics
    summary_data.extend(
        [
            create_summary_stats(
                us_clean, "US (Post-2000)", post_2000_start, last_date
            ),
            create_summary_stats(
                uk_clean_sa, "UK (Post-2000)", post_2000_start, last_date
            ),
        ]
    )

    summary_df = pd.DataFrame(summary_data)
    # print("=" * 50)
    mo.vstack(
        [
            mo.md("Summary Statistics for Input Data"),
            mo.md("".join(["=" for i in range(50)])),
            summary_df,
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Create patsy formulas""")
    return


@app.cell
def _():
    # Phillips curve specifications following Kiley (2023)
    phillips_curve_specs = {
        # Main specification from Kiley (2023): N=12 month average
        "main_spec": {
            "formula": "delta_p ~ avg_lag(delta_p, 1, 12) + lag(unemployment_rate, 1)",
            "description": "Main: 12-month average of lagged inflation (N=12)",
            "kiley_ref": "Equation (1) with N=12",
        },
        # Shorter lag specification for robustness
        "short_lags": {
            "formula": "delta_p ~ avg_lag(delta_p, 1, 6) + lag(unemployment_rate, 1)",
            "description": "Alternative: 6-month average of lagged inflation (N=6)",
            "kiley_ref": "Robustness check with shorter lags",
        },
        # # Individual recent lags specification
        # "individual_lags": {
        #     "formula": "delta_p ~ lag(delta_p, 1) + lag(delta_p, 2) + lag(delta_p, 3) + lag(delta_p, 4) + lag(delta_p, 5) + lag(delta_p, 6) + lag(delta_p, 7) + lag(delta_p, 8) + lag(delta_p, 9) + lag(delta_p, 10) + lag(delta_p, 11) + lag(delta_p, 12) + lag(unemployment_rate, 1)",
        #     "description": "Alternative: Individual lags 1-12",
        #     "kiley_ref": "Individual lag specification for comparison",
        # },
    }

    print("Phillips Curve Specifications:")
    print("=" * 50)
    for sn, si in phillips_curve_specs.items():
        print(f"\n{sn.upper()}:")
        print(f"  Formula: {si['formula']}")
        print(f"  Description: {si['description']}")
        print(f"  Reference: {si['kiley_ref']}")
    return (phillips_curve_specs,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Build the data samples""")
    return


@app.cell
def _(
    first_date,
    last_date,
    patsy,
    pd,
    phillips_curve_specs,
    post_2000_start,
    pre_2000_end,
    uk_clean_sa,
    us_clean,
):
    # Create design matrices for each country and specification using patsy
    design_matrices = {}

    # Process both countries
    countries = {"US": us_clean, "UK": uk_clean_sa}

    for _country_name, _data in countries.items():
        print(f"\n{'=' * 50}")
        print(f"CREATING DESIGN MATRICES FOR {_country_name}")
        print(f"{'=' * 50}")

        design_matrices[_country_name] = {}

        for _spec_name, _spec_info in phillips_curve_specs.items():
            print(f"\nProcessing specification: {_spec_name}")
            print(f"Formula: {_spec_info['formula']}")

            # Parse formula and create design matrices
            Y, X = patsy.dmatrices(
                _spec_info["formula"],
                data=_data.copy(),
                # eval_env=env,
                return_type="dataframe",
            )

            # Combine into single dataframe
            _df_reg = pd.concat(
                [Y, X.iloc[:, 1:]], axis=1
            )  # Remove intercept column, add back manually
            _df_reg = _df_reg.dropna()

            # Add intercept manually for consistency with Kiley (2023)
            _df_reg.insert(len(_df_reg.columns), "const", 1.0)

            # Split data by time periods
            _df_pre_2000 = _df_reg.loc[first_date:pre_2000_end]
            _df_post_2000 = _df_reg.loc[post_2000_start:last_date]
            _df_full_sample = _df_reg.loc[first_date:last_date]

            # Store results
            design_matrices[_country_name][_spec_name] = {
                "full_data": _df_reg,
                "pre_2000": _df_pre_2000,
                "post_2000": _df_post_2000,
                "full_sample": _df_full_sample,
                "y_col": Y.columns[0],
                "x_cols": X.columns[1:].tolist() + ["const"],  # Include constant
                "spec_info": _spec_info,
            }

            print(f"  Full data: {len(_df_reg)} observations")
            print(f"  Pre-2000: {len(_df_pre_2000)} observations")
            print(f"  Post-2000: {len(_df_post_2000)} observations")
            print(f"  Columns: {_df_reg.columns.tolist()}")
    return (design_matrices,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Estimate""")
    return


@app.cell
def _(design_matrices, run_bayesian_estimation, run_ols_formula):
    # Run OLS and Bayesian estimation for all country-specification combinations
    regression_results = {}

    # Bayesian weights following Kiley (2023) Table 2
    _weights_on_prior = [0.5, 0.2, 0.05, 0.0]

    for _country_name, _country_specs in design_matrices.items():
        print(f"\n{'=' * 60}")
        print(f"RUNNING ESTIMATIONS FOR {_country_name}")
        print(f"{'=' * 60}")

        regression_results[_country_name] = {}

        for _spec_name, _spec_data in _country_specs.items():
            print(f"\n--- Specification: {_spec_name} ---")

            # Get data splits
            _y_col = _spec_data["y_col"]
            _x_cols = _spec_data["x_cols"]

            _df_pre_2000 = _spec_data["pre_2000"]
            _df_post_2000 = _spec_data["post_2000"]
            _df_full_sample = _spec_data["full_sample"]

            # Initialize results storage
            regression_results[_country_name][_spec_name] = {}

            print(f"Y variable: {_y_col}")
            print(f"X variables: {_x_cols}")

            # 1. OLS on Pre-2000 data (Prior)
            if len(_df_pre_2000) > 0:
                try:
                    (
                        prior_coeffs,
                        prior_std_errs,
                        prior_vcov,
                        prior_sigma_sq,
                        X_pre,
                        Y_pre,
                        prior_nobs,
                        prior_model,
                    ) = run_ols_formula(_df_pre_2000, _y_col, _x_cols)

                    regression_results[_country_name][_spec_name]["prior"] = {
                        "coeffs": prior_coeffs,
                        "std_errs": prior_std_errs,
                        "vcov": prior_vcov,
                        "sigma_sq": prior_sigma_sq,
                        "nobs": prior_nobs,
                        "model": prior_model,
                    }
                    print(f"  Pre-2000 OLS: {prior_nobs} observations")

                except Exception as e:
                    print(f"  Pre-2000 OLS failed: {e}")
                    regression_results[_country_name][_spec_name]["prior"] = None

            # 2. OLS on Post-2000 data (Likelihood)
            if len(_df_post_2000) > 0:
                try:
                    (
                        post_coeffs,
                        post_std_errs,
                        post_vcov,
                        post_sigma_sq,
                        X_post,
                        Y_post,
                        post_nobs,
                        post_model,
                    ) = run_ols_formula(_df_post_2000, _y_col, _x_cols)

                    regression_results[_country_name][_spec_name]["post_2000"] = {
                        "coeffs": post_coeffs,
                        "std_errs": post_std_errs,
                        "vcov": post_vcov,
                        "sigma_sq": post_sigma_sq,
                        "nobs": post_nobs,
                        "model": post_model,
                        "X": X_post,
                        "Y": Y_post,
                    }
                    print(f"  Post-2000 OLS: {post_nobs} observations")

                except Exception as e:
                    print(f"  Post-2000 OLS failed: {e}")
                    regression_results[_country_name][_spec_name]["post_2000"] = None

            # 3. OLS on Full Sample
            if len(_df_full_sample) > 0:
                try:
                    (
                        full_coeffs,
                        full_std_errs,
                        full_vcov,
                        full_sigma_sq,
                        X_full,
                        Y_full,
                        full_nobs,
                        full_model,
                    ) = run_ols_formula(_df_full_sample, _y_col, _x_cols)

                    regression_results[_country_name][_spec_name]["full_sample"] = {
                        "coeffs": full_coeffs,
                        "std_errs": full_std_errs,
                        "vcov": full_vcov,
                        "sigma_sq": full_sigma_sq,
                        "nobs": full_nobs,
                        "model": full_model,
                    }
                    print(f"  Full sample OLS: {full_nobs} observations")

                except Exception as e:
                    print(f"  Full sample OLS failed: {e}")
                    regression_results[_country_name][_spec_name]["full_sample"] = None

            # 4. Bayesian estimations with different weights
            prior_results = regression_results[_country_name][_spec_name].get("prior")
            post_results = regression_results[_country_name][_spec_name].get(
                "post_2000"
            )

            if prior_results and post_results:
                regression_results[_country_name][_spec_name]["bayesian"] = {}

                for _w in _weights_on_prior:
                    try:
                        bayes_coeffs, bayes_std_errs = run_bayesian_estimation(
                            prior_results["coeffs"].values,
                            prior_results["vcov"].values,
                            post_results["coeffs"].values,
                            post_results["X"].values,
                            post_results["sigma_sq"],
                            _w,
                        )

                        regression_results[_country_name][_spec_name]["bayesian"][
                            f"w_{_w}"
                        ] = {
                            "coeffs": bayes_coeffs,
                            "std_errs": bayes_std_errs,
                            "weight": _w,
                            "nobs": post_results["nobs"],
                        }

                    except Exception as e:
                        print(f"  Bayesian estimation (w={_w}) failed: {e}")
            else:
                print(
                    "  Skipping Bayesian estimation (missing prior or post-2000 data)"
                )

    print(f"\n{'=' * 60}")
    print("ESTIMATION COMPLETED")
    print(f"{'=' * 60}")
    return (regression_results,)


@app.cell
def _(mo):
    mo.md(r"""## Results table""")
    return


@app.cell
def _(mo, pd, phillips_curve_specs, regression_results):
    # Construct comprehensive results table following Kiley (2023) format
    comparison_data = []

    for _country_name, _country_results in regression_results.items():
        for _spec_name, _spec_results in _country_results.items():
            _spec_description = phillips_curve_specs[_spec_name]["description"]

            # Extract inflation persistence coefficient (first X variable)
            # This corresponds to b(1) in Kiley's notation

            # Prior (Pre-2000)
            if _spec_results.get("prior"):
                prior_result = _spec_results["prior"]
                # Get the first non-constant coefficient (inflation persistence)
                coeff_name = [
                    col for col in prior_result["coeffs"].index if col != "const"
                ][0]
                b1_coeff = prior_result["coeffs"][coeff_name]
                b1_se = prior_result["std_errs"][coeff_name]

                comparison_data.append(
                    {
                        "Country": _country_name,
                        "Specification": _spec_description,
                        "Estimation": "OLS Pre-2000 (Prior)",
                        "b(1) Coefficient": f"{b1_coeff:.3f}",
                        "b(1) Std Error": f"({b1_se:.3f})",
                        "Observations": prior_result["nobs"],
                        "Sample": "Pre-2000",
                    }
                )

            # Post-2000 (Uninformative Prior)
            if _spec_results.get("post_2000"):
                post_result = _spec_results["post_2000"]
                coeff_name = [
                    col for col in post_result["coeffs"].index if col != "const"
                ][0]
                b1_coeff = post_result["coeffs"][coeff_name]
                b1_se = post_result["std_errs"][coeff_name]

                comparison_data.append(
                    {
                        "Country": _country_name,
                        "Specification": _spec_description,
                        "Estimation": "OLS Post-2000 (Uninf. Prior)",
                        "b(1) Coefficient": f"{b1_coeff:.3f}",
                        "b(1) Std Error": f"({b1_se:.3f})",
                        "Observations": post_result["nobs"],
                        "Sample": "Post-2000",
                    }
                )

            # Full Sample
            if _spec_results.get("full_sample"):
                _full_result = _spec_results["full_sample"]
                _coeff_name = [
                    col for col in _full_result["coeffs"].index if col != "const"
                ][0]
                b1_coeff = _full_result["coeffs"][_coeff_name]
                b1_se = _full_result["std_errs"][_coeff_name]

                comparison_data.append(
                    {
                        "Country": _country_name,
                        "Specification": _spec_description,
                        "Estimation": "OLS Full Sample",
                        "b(1) Coefficient": f"{b1_coeff:.3f}",
                        "b(1) Std Error": f"({b1_se:.3f})",
                        "Observations": _full_result["nobs"],
                        "Sample": "Full",
                    }
                )

            # Bayesian estimates
            if _spec_results.get("bayesian"):
                for _weight_key, _bayes_result in _spec_results["bayesian"].items():
                    _w = _bayes_result["weight"]
                    b1_coeff = _bayes_result["coeffs"][0]  # First coefficient
                    b1_se = _bayes_result["std_errs"][0]  # First standard error

                    comparison_data.append(
                        {
                            "Country": _country_name,
                            "Specification": _spec_description,
                            "Estimation": f"Bayesian (w={_w})",
                            "b(1) Coefficient": f"{b1_coeff:.3f}",
                            "b(1) Std Error": f"({b1_se:.3f})",
                            "Observations": _bayes_result["nobs"],
                            "Sample": "Post-2000",
                        }
                    )

    comparison_df = pd.DataFrame(comparison_data)

    _items = []
    for _country in comparison_df["Country"].unique():
        _items.append(mo.md(f"##{_country}:"))
        for _spec in comparison_df["Specification"].unique():
            _items.append(mo.md(f"**{_country} {_spec}:**"))
            _items.append(
                comparison_df[
                    (comparison_df["Country"] == _country)
                    & (comparison_df["Specification"] == _spec)
                ]
            )
            _items.append(mo.md("-" * 40))

    mo.vstack(_items)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Construct forecasts""")
    return


@app.cell
def _(
    iterative_forecast,
    np,
    pd,
    process_forecast,
    regression_results,
    uk_clean_sa,
    us_clean,
):
    # Create forecasts for the main specification using Bayesian estimates
    print("Creating forecasts using main specification...")

    # Forecast parameters
    end_of_history = pd.to_datetime("2021-12")
    forecast_steps = 24

    all_forecasts = {}

    for _country_name in ["US", "UK"]:
        print(f"\nCreating forecasts for {_country_name}...")

        # Use main specification
        _spec_name = "main_spec"
        _spec_results = regression_results[_country_name][_spec_name]

        # Get the data used for this specification
        _data = us_clean if _country_name == "US" else uk_clean_sa

        # Create initial conditions
        initial_lags_end = end_of_history
        initial_lags_start = end_of_history - pd.DateOffset(months=11)

        # Get recent inflation values for initial conditions
        initial_inf_lags = _data.loc[initial_lags_start:initial_lags_end, "delta_p"]

        # Set unemployment assumption (constant at last known value)
        last_known_unemployment = _data.loc[end_of_history, "unemployment_rate"]

        # Create forecast dates
        forecast_dates = pd.date_range(
            start=end_of_history, periods=forecast_steps + 1, freq="MS"
        )[1:]

        # Create unemployment assumption series -> fixed at last_known_unemployment
        unemployment_assumption = pd.Series(
            data=last_known_unemployment, index=forecast_dates
        )

        # Create base dataframe for processing forecasts
        base_df = pd.DataFrame(index=_data.index)
        base_df["core_cpi"] = _data["core_cpi"].loc[:initial_lags_end]
        base_df["core_cpi_log"] = np.log(base_df["core_cpi"])
        base_df["delta_p"] = base_df["core_cpi_log"].diff(1) * 100
        base_df["delta_p_fcst"] = base_df["delta_p"]

        # Generate forecasts for different Bayesian weights
        country_forecasts = {}

        if _spec_results.get("bayesian"):
            for _weight_key, _bayes_result in _spec_results["bayesian"].items():
                _weight = _bayes_result["weight"]

                # Extract coefficients (assume order: inflation persistence, unemployment, constant)
                coeffs = [
                    _bayes_result["coeffs"][0],  # inflation persistence
                    _bayes_result["coeffs"][-2],  # unemployment (second to last)
                    _bayes_result["coeffs"][-1],  # constant (last)
                ]

                # Generate forecast
                forecast_series = iterative_forecast(
                    coeffs, initial_inf_lags.values, unemployment_assumption
                )

                # Process forecast to convert back to level terms
                processed_forecast = process_forecast(
                    forecast_series, base_df, initial_lags_end
                )

                country_forecasts[f"Bayesian (w={_weight})"] = processed_forecast

        all_forecasts[_country_name] = country_forecasts

        print(f"  Created {len(country_forecasts)} forecasts for {_country_name}")
    return (all_forecasts,)


@app.cell
def _(all_forecasts, go, pd, uk_clean_sa, us_clean):
    # Create forecast visualization plots

    # Function to create forecast plot for each country
    def create_forecast_plot(country_name, forecasts, data):
        fig = go.Figure()

        # Add historical data (last few years for context)
        historical_start = pd.Timestamp("2018-01-01")
        historical_data = (
            # data["core_cpi"].loc[historical_start:end_of_history].pct_change(12) * 100
            data["core_cpi"].loc[historical_start:].pct_change(12) * 100
        )

        fig.add_trace(
            go.Scatter(
                x=historical_data.index,
                y=historical_data.values,
                name="Historical Data",
                mode="lines",
                line=dict(color="black", width=2),
            )
        )

        # Add forecast traces
        colors = ["blue", "red", "green", "orange"]
        for i, (forecast_name, forecast_data) in enumerate(forecasts.items()):
            fig.add_trace(
                go.Scatter(
                    x=forecast_data.index,
                    y=forecast_data.values,
                    name=forecast_name,
                    mode="lines",
                    line=dict(color=colors[i % len(colors)], width=2.5, dash="dot"),
                )
            )

        # # Add vertical line at forecast start
        # fig.add_vline(
        #     x=end_of_history + pd.DateOffset(months=1),
        #     line_dash="dash",
        #     line_color="gray",
        #     annotation_text="Forecast Period",
        #     annotation_position="top right",
        # )

        # Update layout
        fig.update_layout(
            title=dict(
                text=f"<b>{country_name} Core CPI Inflation Forecasts</b><br>"
                "<sub>12-month percent change, using Bayesian Phillips Curve estimates</sub>",
                x=0.5,
                y=0.95,
                font=dict(size=16),
            ),
            yaxis_title="12-Month Percent Change",
            yaxis_ticksuffix="%",
            template="plotly_white",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5,
            ),
            hovermode="x unified",
            height=500,
        )

        return fig

    # Create plots for both countries
    for _country_name in ["US", "UK"]:
        _data = us_clean if _country_name == "US" else uk_clean_sa
        _forecasts = all_forecasts[_country_name]

        if _forecasts:
            fig = create_forecast_plot(_country_name, _forecasts, _data)
            fig.show()
        else:
            print(f"No forecasts available for {_country_name}")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    # OOS Forecasting exercise
    We now need to calculate the RMSE through time of the OOS forecasts for a range of weights, for both the UK and US. 

    So we run the forecast (for a specified horizon), over a specified time period (as long or short as we want), and see how the RMSE for the different weights evolves through time.

    Components to get this to work are:

    - Run out-of-sample forecast and store in big dict with a key for each timestep, and a sub-key for each weight, with the value being the forecast data. Re-estimate the model at each time step - updating the calculated weights etc.
    - Create another dict, mirroring the structure of the previous, where we calculate RMSE for each OOS across the weights, for each timestep. Gracefully deal with out-of-bound issues (where we no longer have enough realisations for the whole forecast horizon).
    - Then return the dicts of OOS forecasts and their corresponding RMSEs

    We need to be able to get this to work, the entire estimation process, etc, as a single series of steps which we can loop over as we go through time, and for different input datasets (countries) and specifications (formulae).
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Functions
    Was importing them previously, but will stick them here since I thinkn we might have been getting some deep bug
    """
    )
    return


@app.cell
def _(deque, mo, np, patsy, pd, scipy_inv, sm):
    """
    This module contains functions for running an out-of-sample forecasting exercise
    for the Bayesian policymaker model.
    """

    def oos_exercise_calculate_rmse(forecast, actual):
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

    def oos_exercise_prepare_basic_data(
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

    def oos_exercise_run_ols_formula(df_sample, y_col, x_cols):
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

    def oos_exercise_run_bayesian_estimation(
        gamma_prior, V_prior, gamma_ls, X_post, sigma_sq_post, w
    ):
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

    def oos_exercise_iterative_forecast(coeffs, initial_inf_lags, unemployment_lags):
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

    def oos_exercise_process_forecast(forecast_series, base_df, initial_lags_end):
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

        for t in mo.status.progress_bar(oos_dates):
            # print(f"Running forecast for origin: {t.strftime('%Y-%m')}")

            # 1. Prepare data for this iteration
            historical_data = data.loc[:t].copy()

            # 2. Create design matrices
            Y, X = patsy.dmatrices(
                formula, data=historical_data, return_type="dataframe"
            )
            df_reg = pd.concat([Y, X.iloc[:, 1:]], axis=1).dropna()
            df_reg["const"] = 1.0

            y_col = Y.columns[0]
            x_cols = X.columns[1:].tolist() + ["const"]

            df_prior = df_reg.loc[prior_start_date:prior_end_date]
            df_likelihood = df_reg.loc[prior_end_date:t]

            # 3. Estimate prior and likelihood models
            try:
                (prior_coeffs, _, prior_vcov, _, _, _, _, _) = (
                    oos_exercise_run_ols_formula(df_prior, y_col, x_cols)
                )
                (like_coeffs, _, _, like_sigma_sq, X_like, _, _, _) = (
                    oos_exercise_run_ols_formula(df_likelihood, y_col, x_cols)
                )
            except Exception as e:
                print(f"  Estimation failed for {t}: {e}")
                continue

            oos_forecasts[t] = {}
            oos_rmses[t] = {}

            # 4. Generate forecasts for each weight
            for w in weights:
                # print(f"  Calculating posterior for weight: {w}")
                bayes_coeffs, _ = oos_exercise_run_bayesian_estimation(
                    prior_coeffs.values,
                    prior_vcov.values,
                    like_coeffs.values,
                    X_like.values,
                    like_sigma_sq,
                    w,
                )

                # Prepare for forecasting
                initial_lags_end = t
                # start forecast from realisation pd.DateOffset(months=11)
                initial_lags_start = t - pd.DateOffset(months=11)
                initial_inf_lags = historical_data.loc[
                    initial_lags_start:initial_lags_end, "delta_p"
                ]

                last_known_unemployment = historical_data.loc[t, "unemployment_rate"]
                forecast_dates = pd.date_range(
                    start=t, periods=forecast_horizon + 1, freq="MS"
                ) # [1:] - start forecast from realisation
                unemployment_assumption = pd.Series(
                    data=last_known_unemployment, index=forecast_dates
                )

                # Create base dataframe for processing forecasts
                base_df = pd.DataFrame(index=data.index)
                base_df["core_cpi"] = data["core_cpi"].loc[:initial_lags_end]
                base_df["core_cpi_log"] = np.log(base_df["core_cpi"])

                # Generate and process forecast
                fcst_series = oos_exercise_iterative_forecast(
                    bayes_coeffs, initial_inf_lags.values, unemployment_assumption
                )
                processed_fcst = oos_exercise_process_forecast(
                    fcst_series, base_df, initial_lags_end
                )

                oos_forecasts[t][w] = processed_fcst

                # 5. Calculate RMSE
                actual_data = data["core_cpi"].pct_change(12) * 100
                rmse = oos_exercise_calculate_rmse(processed_fcst, actual_data)
                oos_rmses[t][w] = rmse

        return oos_forecasts, oos_rmses

    return (run_oos_exercise,)


@app.cell
def _(mo):
    mo.md(r"""## Run OOS exercise""")
    return


@app.cell
def _(pd, phillips_curve_specs, run_oos_exercise, uk_clean_sa, us_clean):
    # OOS Configuration
    oos_config = {
        "formula": phillips_curve_specs["main_spec"]["formula"],
        "forecast_horizon": 12,
        "oos_start_date": pd.to_datetime("2010-01-01"),
        "oos_end_date": pd.to_datetime("2024-12-01"),
        "prior_start_date": pd.to_datetime("1971-02-01"),
        "prior_end_date": pd.to_datetime("1997-06-01"),
        "weights": [w / 100 for w in range(0, 100, 10)],
    }

    # Run for US
    us_oos_forecasts, us_oos_rmses = run_oos_exercise(data=us_clean, **oos_config)

    # Run for UK
    uk_oos_forecasts, uk_oos_rmses = run_oos_exercise(data=uk_clean_sa, **oos_config)
    return (
        oos_config,
        uk_oos_forecasts,
        uk_oos_rmses,
        us_oos_forecasts,
        us_oos_rmses,
    )


@app.cell
def _(pd, uk_oos_forecasts):
    uk_oos_forecasts[pd.Timestamp("2020-01-01")][0.0]
    return


@app.cell
def _(mo):
    mo.md(r"""## Killer charts""")
    return


@app.cell
def _(mo):
    mo.md(r"""### Visualize OOS RMSEs for each weight""")
    return


@app.cell
def _(go, oos_config, pd, uk_oos_rmses, us_oos_rmses):
    # Function to create RMSE plot
    def create_rmse_plot(rmses, country_name):
        fig = go.Figure()

        for w in oos_config["weights"]:
            rmse_series = pd.Series({t: rmses[t][w] for t in rmses if w in rmses[t]})
            fig.add_trace(
                go.Scatter(
                    x=rmse_series.index,
                    y=rmse_series.rolling(12).mean().values,
                    name=f"w={w}",
                    mode="lines",
                )
            )

        fig.update_layout(
            title=f"{country_name}: OOS RMSEs for Different Weights - 12 average",
            xaxis_title="Forecast Origin",
            yaxis_title="RMSE",
            legend_title="Weight",
        )
        return fig

    # Create plots
    us_rmse_plot = create_rmse_plot(us_oos_rmses, "US")
    us_rmse_plot.show()

    uk_rmse_plot = create_rmse_plot(uk_oos_rmses, "UK")
    uk_rmse_plot.show()
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### RMSE minimising W 
    - The W which minimises RMSE through time - continuous scale -> KILLER CHART
    """
    )
    return


@app.cell
def _(go, oos_config, pd, uk_oos_rmses, us_oos_rmses):
    def best_weight_plot(rmses, country_name):
        fig = go.Figure()
        date_range = pd.date_range(
            start=oos_config["oos_start_date"],
            end=oos_config["oos_end_date"],
            freq="MS",
        )

        best_weights = []
        for date in date_range:
            rmse_dict = rmses[date]
            if rmse_dict:
                best_w = min(rmse_dict, key=rmse_dict.get)
                best_weights.append((date, best_w))
            else:
                best_weights.append((date, None))

        dates, weights = zip(*best_weights)
        best_weights_series = pd.Series(weights, index=dates)
        best_weights_series.name = "best_weight"

        fig.add_trace(
            go.Scatter(
                x=best_weights_series.rolling(12).mean().index,
                y=best_weights_series.rolling(12).mean().values,
                mode="lines",
                name="Best Weight",
            )
        )

        fig.update_layout(
            title=f"{country_name}: Weight with Lowest OOS RMSE Over Time, 12m rolling average",
            xaxis_title="Forecast Origin",
            yaxis_title="Best Weight (w)",
            legend_title="Legend",
        )
        return fig

    us_best_w_plot = best_weight_plot(us_oos_rmses, "US")
    us_best_w_plot.show()

    uk_best_w_plot = best_weight_plot(uk_oos_rmses, "UK")
    uk_best_w_plot.show()

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Distribution through time""")
    return


@app.cell
def _():
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Forecasts at specified dates""")
    return


@app.cell
def _(
    go,
    np,
    pd,
    px,
    uk_clean_sa,
    uk_oos_forecasts,
    us_clean,
    us_oos_forecasts,
):
    # Create a list of dates to view the forecast at
    dates_to_view = [
        pd.Timestamp("2019-01-01"),
        pd.Timestamp("2021-01-01"),
        pd.Timestamp("2022-01-01"),
        pd.Timestamp("2023-01-01"),
        pd.Timestamp("2024-01-01"),
    ]

    weights_to_view = [
        0.0,
        0.2,
        0.5,
        0.7,
        0.9,
    ]

    # Create a figure and plot the forecasts at each timestep
    def create_hedgehog_plot(
        oos_forecasts, dates_to_view, weights_to_view, data, historical_start
    ):
        fig = go.Figure()

        # 1. Generate a color map based on the weights.
        # To ensure the gradient is meaningful, sort the weights first.
        sorted_weights = sorted(weights_to_view)
        n_weights = len(sorted_weights)

        if n_weights > 1:
            colors = px.colors.sample_colorscale("Turbo", np.linspace(0, 1, n_weights))
        elif n_weights == 1:
            colors = px.colors.sample_colorscale("Turbo", [0.5])
        else:
            colors = []

        # Map each weight to a color 🎨
        weight_to_color = {w: color for w, color in zip(sorted_weights, colors)}

        # 2. Loop through dates and then weights, applying the color map
        for date in dates_to_view:
            for w in weights_to_view:
                fig.add_trace(
                    go.Scatter(
                        x=oos_forecasts[date][w].index,
                        y=oos_forecasts[date][w].values,
                        hovertemplate="value: %{y:.2f}"
                        + "<br>"
                        + f"w: {w}"
                        + "<br>"
                        + f"Forecast date: {date.strftime('%Y-%m')}",
                        mode="lines",
                        # Apply the color based on the weight 'w'
                        line=dict(color=weight_to_color[w]),
                    )
                )

        # Add the true path
        historical_data = data["core_cpi"].loc[historical_start:].pct_change(12) * 100
        fig.add_trace(
            go.Scatter(
                x=historical_data.index,
                y=historical_data.values,
                name="True Path",
                mode="lines",
                line=dict(color="black", dash="dash"),
            )
        )
        fig.update_layout(
            title="Forecasts at Specified Dates (Colored by Weight)",
            xaxis_title="Date",
            yaxis_title="Forecast",
            showlegend=False,
            hovermode="x",
        )

        return fig

    us_hedgehog_plot = create_hedgehog_plot(
        us_oos_forecasts,
        dates_to_view,
        weights_to_view,
        us_clean,
        pd.Timestamp("2017-01-01"),
    )
    us_hedgehog_plot.update_layout(
        title="US Forecasts at Specified Dates",
    )
    us_hedgehog_plot.show()

    uk_hedgehog_plot = create_hedgehog_plot(
        uk_oos_forecasts,
        dates_to_view,
        weights_to_view,
        uk_clean_sa,
        pd.Timestamp("2017-01-01"),
    )
    uk_hedgehog_plot.update_layout(
        title="UK Forecasts at Specified Dates",
    )
    uk_hedgehog_plot.show()
    return


if __name__ == "__main__":
    app.run()
