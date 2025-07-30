import marimo

__generated_with = "0.14.13"
app = marimo.App(width="full")


@app.cell
def _():
    from collections import deque

    import marimo as mo
    import numpy as np
    import pandas as pd
    import patsy
    import plotly.graph_objects as go
    import statsmodels.api as sm
    from plotly.subplots import make_subplots
    from scipy.linalg import inv as scipy_inv

    return deque, go, make_subplots, mo, np, patsy, pd, scipy_inv, sm


@app.cell
def _(mo):
    mo.md(
        r"""
    # Bayesian Policy Maker
    How much information should a policy maker take from pre-2000 inflation as regards persistence in the post-2000 era.

    This note replicates the methodology outlined in [Kiley](https://www.ijcb.org/journal/ijcb24q1a6.pdf), extending it to UK data.
    """
    )
    return


@app.cell
def _(mo):
    mo.md(r"""## Configuration and Shared Functions""")
    return


@app.cell
def _(pd):
    # Country-specific configurations
    country_configs = {
        "US": {
            "name": "United States",
            "data_files": {
                "unemployment": "data/Unemployment Rate UNRATE.csv",
                "cpi": "data/CPI Less Food and Energy.csv",
            },
            "column_mappings": {
                "unemployment_raw": "UNRATE",
                "cpi_raw": "CPILFESL",
                "date_col": "observation_date",
            },
            "date_ranges": {
                "first_date": pd.Timestamp("1958-01-01"),
                "last_date": pd.Timestamp("2019-12-31"),
                "pre_2000_end": pd.Timestamp("1999-12-31"),
                "post_2000_start": pd.Timestamp("2000-01-01"),
            },
            "seasonal_adjustment": False,
            "forecast_config": {
                "end_of_history": pd.to_datetime("2021-12"),
                "forecast_steps": 24,
            },
        },
        "GBR": {
            "name": "United Kingdom",
            "data_files": {
                "unemployment": "data/Unemployment Rate 16+ Seasonally Adjusted_CLEAN.csv",
                "cpi": "data/OECD CPI Data GBR.csv",
            },
            "column_mappings": {
                "unemployment_raw": "Value",
                "cpi_raw": "OBS_VALUE",
                "date_col_unemp": "Date",
                "date_col_cpi": "TIME_PERIOD",
            },
            "date_ranges": {
                "first_date": pd.Timestamp("1971-02-01"),
                "last_date": pd.Timestamp("2025-02-01"),
                "pre_2000_end": pd.Timestamp("1999-12-31"),
                "post_2000_start": pd.Timestamp("2000-01-01"),
            },
            "seasonal_adjustment": True,
            "forecast_config": {
                "end_of_history": pd.to_datetime("2021-12"),
                "forecast_steps": 24,
            },
        },
    }

    return (country_configs,)


@app.cell
def _(pd):
    # Patsy transformation functions - defined directly for use in formulas
    def lag(series, n=1):
        """Simple lag function: lag(variable, n)"""
        if isinstance(series, pd.Series):
            return series.shift(n)
        elif isinstance(series, pd.DataFrame):
            return series.shift(n)
        else:
            # Handle numpy arrays
            return pd.Series(series).shift(n)

    def avg_lag(series, start_lag, end_lag):
        """Average of lags from start_lag to end_lag: avg_lag(variable, 1, 12)"""
        if isinstance(series, (pd.Series, pd.DataFrame)):
            _lagged_series = [series.shift(i) for i in range(start_lag, end_lag + 1)]
        else:
            # Handle numpy arrays
            series = pd.Series(series)
            _lagged_series = [series.shift(i) for i in range(start_lag, end_lag + 1)]

        return pd.concat(_lagged_series, axis=1).mean(axis=1)

    def weighted_avg_lag(series, start_lag, end_lag, weights=None):
        """Weighted average of lags: weighted_avg_lag(variable, 1, 12, [0.3, 0.25, ...])"""
        if isinstance(series, (pd.Series, pd.DataFrame)):
            _lagged_series = [series.shift(i) for i in range(start_lag, end_lag + 1)]
        else:
            series = pd.Series(series)
            _lagged_series = [series.shift(i) for i in range(start_lag, end_lag + 1)]

        if weights is None:
            # Default to equal weights
            weights = [1.0] * (end_lag - start_lag + 1)

        # Ensure weights match number of lags
        weights = weights[: len(_lagged_series)]
        if len(weights) < len(_lagged_series):
            weights.extend([0.0] * (len(_lagged_series) - len(weights)))

        _weighted_sum = sum(
            w * lag_series for w, lag_series in zip(weights, _lagged_series)
        )
        _total_weight = sum(weights)

        return (
            _weighted_sum / _total_weight
            if _total_weight > 0
            else _lagged_series[0] * 0
        )

    def rolling_avg_lag(series, window, lag=1):
        """Rolling average starting from lag: rolling_avg_lag(variable, 12, 1)"""
        if isinstance(series, (pd.Series, pd.DataFrame)):
            return series.shift(lag).rolling(window=window).mean()
        else:
            return pd.Series(series).shift(lag).rolling(window=window).mean()

    return avg_lag, lag, rolling_avg_lag, weighted_avg_lag


@app.cell
def _():
    # Phillips curve specifications - defined directly following Kiley (2023)
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
        # Individual recent lags specification
        "individual_lags": {
            "formula": "delta_p ~ lag(delta_p, 1) + lag(delta_p, 2) + lag(delta_p, 3) + lag(delta_p, 4) + lag(unemployment_rate, 1)",
            "description": "Alternative: Individual lags 1-4",
            "kiley_ref": "Individual lag specification for comparison",
        },
    }

    return (phillips_curve_specs,)


@app.cell
def _(sm):
    def run_ols_formula(df_sample, y_col, x_cols):
        """
        Run OLS regression using column names from Patsy formula parsing.
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
        Performs Bayesian estimation based on Equation 4 of Kiley (2022).
        """
        print(f"Calculating posterior for {w}")

        # Prior precision (V^-1)
        inv_V_prior = scipy_inv(V_prior)

        # Data precision (σ^-2 * X'X)
        data_precision = (1 / sigma_sq_post) * (X_post.T @ X_post)

        # Posterior variance-covariance matrix
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
        Generates multi-step ahead forecasts for inflation as a pandas Series.
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

    return (iterative_forecast,)


@app.cell
def _(np):
    def process_forecast(forecast_series, base_df, initial_lags_end):
        """
        Processes a single forecast series to rebase it.
        """
        df = base_df.copy()
        forecast_series.name = "delta_p_fcst"

        # Combine the historical data with the forecast
        df = forecast_series.to_frame().combine_first(df)

        # Convert from log-differences back to level
        df["delta_p_fcst"] = df["delta_p_fcst"] / 100
        df["delta_p_fcst"] = df["delta_p_fcst"].fillna(df["core_cpi_log"])
        df["delta_p_fcst"] = df["delta_p_fcst"].cumsum()
        df["delta_p_fcst"] = np.exp(df["delta_p_fcst"])

        # Bring into 12 change
        df["delta_p_fcst"] = df["delta_p_fcst"].diff(12)

        # Return only the forecasted portion
        return df["delta_p_fcst"].loc[df.index > initial_lags_end]

    return (process_forecast,)


@app.cell
def _(pd):
    def load_country_data(config):
        """
        Loads and processes data for a specific country.
        """
        country_name = config["name"]
        print(f"Loading data for {country_name}")

        if country_name == "United States":
            # US data loading
            unemployment = pd.read_csv(config["data_files"]["unemployment"])
            unemployment[config["column_mappings"]["date_col"]] = pd.to_datetime(
                unemployment[config["column_mappings"]["date_col"]]
            )

            core_cpi = pd.read_csv(config["data_files"]["cpi"])
            core_cpi[config["column_mappings"]["date_col"]] = pd.to_datetime(
                core_cpi[config["column_mappings"]["date_col"]]
            )

            data = pd.merge(
                unemployment,
                core_cpi,
                on=config["column_mappings"]["date_col"],
                how="left",
            )
            data = data.rename(
                columns={
                    config["column_mappings"]["unemployment_raw"]: "unemployment_rate",
                    config["column_mappings"]["cpi_raw"]: "core_cpi",
                }
            )
            data.set_index(config["column_mappings"]["date_col"], inplace=True)

        else:  # GBR
            # UK data loading
            cpi_data = pd.read_csv(config["data_files"]["cpi"])
            cpi_data = cpi_data[["TIME_PERIOD", "OBS_VALUE"]]
            cpi_data.index = pd.to_datetime(cpi_data["TIME_PERIOD"])
            cpi_data = cpi_data.drop(columns=["TIME_PERIOD"]).sort_index()

            unemp_data = pd.read_csv(config["data_files"]["unemployment"])
            unemp_data.index = pd.to_datetime(unemp_data["Date"])
            unemp_data = unemp_data.drop(columns=["Date"])

            data = pd.merge(
                cpi_data, unemp_data, how="inner", left_index=True, right_index=True
            )
            data.rename(
                columns={"OBS_VALUE": "core_cpi", "Value": "unemployment_rate"},
                inplace=True,
            )

        return data

    return (load_country_data,)


@app.cell
def _(sm):
    def apply_seasonal_adjustment(regression_df):
        """
        Applies seasonal adjustment to the inflation data if needed.
        """

        def try_x13_seasonal_adjustment(series, freq="M"):
            try:
                x13_series = series.dropna()
                print("Attempting X13-ARIMA-SEATS seasonal adjustment...")

                result = sm.tsa.x13_arima_analysis(
                    x13_series,
                    freq=freq,
                    x12path=None,
                    prefer_x13=True,
                    log=False,
                    trading=True,
                    outlier=True,
                    automdl=True,
                )

                seasadj = result.seasadj
                seasonal = result.irregular + result.trend - seasadj
                print("✓ Successfully applied X13-ARIMA-SEATS seasonal adjustment")
                return seasadj, seasonal, "X13-ARIMA-SEATS"

            except Exception as e:
                print(f"X13 adjustment failed: {str(e)}")
                print("Falling back to STL decomposition...")

                try:
                    stl_result = sm.tsa.STL(
                        series.dropna(), seasonal=13, period=12, robust=True
                    ).fit()

                    seasadj = stl_result.observed - stl_result.seasonal
                    seasonal = stl_result.seasonal
                    print("✓ Successfully applied STL seasonal adjustment")
                    return seasadj, seasonal, "STL"

                except Exception as stl_e:
                    print(f"STL adjustment also failed: {str(stl_e)}")
                    print("Falling back to simple additive decomposition...")

                    decomp = sm.tsa.seasonal_decompose(
                        series.dropna(), model="additive", period=12
                    )
                    seasadj = decomp.observed - decomp.seasonal
                    seasonal = decomp.seasonal
                    print("✓ Applied simple additive seasonal adjustment")
                    return seasadj, seasonal, "Simple Additive"

        original_series = regression_df["Y"].copy()
        seasadj_series, seasonal_component, method_used = try_x13_seasonal_adjustment(
            original_series
        )

        # Update the dataframe with seasonally adjusted values
        aligned_seasadj = seasadj_series.reindex(regression_df.index)
        aligned_seasonal = seasonal_component.reindex(regression_df.index)
        regression_df["Y"] = aligned_seasadj

        return regression_df, method_used, aligned_seasonal

    return (apply_seasonal_adjustment,)


@app.cell
def _(mo):
    mo.md(r"""## Data Loading and Preparation""")
    return


@app.cell
def _(apply_seasonal_adjustment, country_configs, load_country_data, np):
    # Load and prepare data for US
    print(f"\n{'=' * 50}")
    print(f"LOADING DATA FOR {country_configs['US']['name'].upper()}")
    print(f"{'=' * 50}")

    # Load raw US data
    us_raw_data = load_country_data(country_configs["US"])
    us_config = country_configs["US"]

    # Prepare basic variables (inflation, unemployment)
    us_data = us_raw_data.copy()
    us_data = us_data.sort_index()

    # Calculate inflation rate
    us_data["core_cpi_log"] = np.log(us_data["core_cpi"])
    us_data["delta_p"] = us_data["core_cpi_log"].diff(1) * 100
    us_data["unemployment_rate"] = us_data["unemployment_rate"]

    # Apply seasonal adjustment if needed (US doesn't need it)
    if us_config["seasonal_adjustment"]:
        print(f"Applying seasonal adjustment for {us_config['name']}")
        # Create a temporary regression df for seasonal adjustment
        _temp_df = us_data[["delta_p", "unemployment_rate"]].copy()
        _temp_df["Y"] = _temp_df["delta_p"]
        _temp_df = _temp_df.dropna()

        _adjusted_df, _method = apply_seasonal_adjustment(_temp_df)
        # Update the main data with seasonally adjusted inflation
        us_data.loc[_adjusted_df.index, "delta_p"] = _adjusted_df["Y"]
        print(f"Seasonal adjustment method used: {_method}")

    print(f"Data period: {us_data.index.min()} to {us_data.index.max()}")
    print(f"Observations: {len(us_data)}")
    print(f"Missing values in delta_p: {us_data['delta_p'].isna().sum()}")
    print(
        f"Missing values in unemployment_rate: {us_data['unemployment_rate'].isna().sum()}"
    )

    # Load and prepare data for GBR
    print(f"\n{'=' * 50}")
    print(f"LOADING DATA FOR {country_configs['GBR']['name'].upper()}")
    print(f"{'=' * 50}")

    # Load raw GBR data
    gbr_raw_data = load_country_data(country_configs["GBR"])
    gbr_config = country_configs["GBR"]

    # Prepare basic variables (inflation, unemployment)
    gbr_data = gbr_raw_data.copy()
    gbr_data = gbr_data.sort_index()

    # Calculate inflation rate
    gbr_data["core_cpi_log"] = np.log(gbr_data["core_cpi"])
    gbr_data["delta_p"] = gbr_data["core_cpi_log"].diff(1) * 100
    gbr_data["unemployment_rate"] = gbr_data["unemployment_rate"]

    # Apply seasonal adjustment if needed (GBR needs it)
    # if gbr_config["seasonal_adjustment"]:
    print(f"Applying seasonal adjustment for {gbr_config['name']}")
    # Create a temporary regression df for seasonal adjustment
    _temp_df = gbr_data[["delta_p", "unemployment_rate"]].copy()
    _temp_df["Y"] = _temp_df["delta_p"]
    _temp_df = _temp_df.dropna()

    # Store original series for plotting
    gbr_original_series = _temp_df["Y"].copy()

    _adjusted_df, _method, _seasonal_component = apply_seasonal_adjustment(_temp_df)
    # Update the main data with seasonally adjusted inflation
    gbr_data.loc[_adjusted_df.index, "delta_p"] = _adjusted_df["Y"]
    print(f"Seasonal adjustment method used: {_method}")

    # Store adjustment results for plotting
    gbr_seasadj_series = _adjusted_df["Y"].copy()
    gbr_seasonal_component = _seasonal_component.copy()
    gbr_method_used = _method
    # else:
    #     _gbr_original_series = None
    #     _gbr_seasadj_series = None
    #     _gbr_seasonal_component = None
    #     _gbr_method_used = None

    print(f"Data period: {gbr_data.index.min()} to {gbr_data.index.max()}")
    print(f"Observations: {len(gbr_data)}")
    print(f"Missing values in delta_p: {gbr_data['delta_p'].isna().sum()}")
    print(
        f"Missing values in unemployment_rate: {gbr_data['unemployment_rate'].isna().sum()}"
    )

    return (
        gbr_config,
        gbr_data,
        gbr_method_used,
        gbr_original_series,
        gbr_raw_data,
        gbr_seasadj_series,
        gbr_seasonal_component,
        us_config,
        us_data,
        us_raw_data,
    )


@app.cell
def _(
    gbr_method_used,
    gbr_original_series,
    gbr_seasadj_series,
    gbr_seasonal_component,
):
    # Visualise the difference between the nsa and seasonally adjusted series
    import matplotlib.pyplot as plt

    # Create enhanced diagnostic plots
    plt.style.use("seaborn-v0_8-whitegrid")
    fig_seas, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    fig_seas.suptitle(
        f"Seasonal Adjustment using {gbr_method_used}",
        fontsize=16,
        fontweight="bold",
    )

    # Plot 1: Before and After comparison
    gbr_original_series.plot(ax=ax1, label="Original", alpha=0.7, color="blue")
    gbr_seasadj_series.plot(
        ax=ax1, label="Seasonally Adjusted", alpha=0.8, color="red", linewidth=2
    )
    ax1.set_title("Original vs Seasonally Adjusted")
    ax1.set_ylabel("Inflation Rate (%)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Seasonal Component
    gbr_seasonal_component.plot(
        ax=ax2, label="Seasonal Component", color="green", linewidth=2
    )
    ax2.set_title("Extracted Seasonal Component")
    ax2.set_ylabel("Seasonal Factor")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Month-by-month seasonal pattern
    seasonal_monthly = gbr_seasonal_component.groupby(
        gbr_seasonal_component.index.month
    ).mean()
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
        months,
        seasonal_monthly.values,
        color="lightblue",
        edgecolor="navy",
        alpha=0.7,
    )
    ax3.set_title("Average Seasonal Pattern by Month")
    ax3.set_ylabel("Average Seasonal Effect")
    ax3.tick_params(axis="x", rotation=45)
    ax3.grid(True, alpha=0.3, axis="y")

    # Plot 4: Residual diagnostics (original - trend - seasonal)
    if gbr_method_used in ["X13-ARIMA-SEATS", "STL"]:
        residuals = gbr_original_series - gbr_seasadj_series - gbr_seasonal_component
        residuals.plot(ax=ax4, color="purple", alpha=0.7)
        ax4.set_title("Irregular Component (Residuals)")
        ax4.set_ylabel("Residual")
        ax4.grid(True, alpha=0.3)
    else:
        # For simple decomposition, show the difference between methods
        ax4.text(
            0.5,
            0.5,
            f"Method Used:\n{gbr_method_used}\n\nConsider installing X13-ARIMA-SEATS\nfor more sophisticated adjustment",
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
    print(f"\n=== Seasonal Adjustment Results ({gbr_method_used}) ===")
    print(
        f"Original series range: {gbr_original_series.min():.3f} to {gbr_original_series.max():.3f}"
    )
    print(
        f"Seasonally adjusted range: {gbr_seasadj_series.min():.3f} to {gbr_seasadj_series.max():.3f}"
    )
    print(
        f"Seasonal component range: {gbr_seasonal_component.min():.3f} to {gbr_seasonal_component.max():.3f}"
    )
    print(f"Seasonal component std: {gbr_seasonal_component.std():.3f}")

    print("\nSeasonal Component Summary:")
    print(gbr_seasonal_component.describe())
    return


@app.cell
def _(
    create_validation_plots,
    gbr_config,
    gbr_raw_data,
    mo,
    us_config,
    us_raw_data,
):
    # Display validation plots for both countries
    _validation_items = []

    # US validation plot
    _us_validation_fig = create_validation_plots("US", us_raw_data, us_config)
    _validation_items.extend(
        [
            mo.md(f"### {us_config['name']} - Data Validation"),
            _us_validation_fig,
        ]
    )

    # GBR validation plot
    _gbr_validation_fig = create_validation_plots("GBR", gbr_raw_data, gbr_config)
    _validation_items.extend(
        [
            mo.md(f"### {gbr_config['name']} - Data Validation"),
            _gbr_validation_fig,
        ]
    )

    mo.vstack(_validation_items)
    return


@app.cell
def _(mo):
    mo.md(r"""## Phillips Curve Specifications""")
    return


@app.cell
def _(phillips_curve_specs):
    # Display the specifications
    print("Phillips Curve Specifications:")
    print("=" * 50)
    for _spec_name, _spec_info in phillips_curve_specs.items():
        print(f"\n{_spec_name.upper()}:")
        print(f"  Formula: {_spec_info['formula']}")
        print(f"  Description: {_spec_info['description']}")
        print(f"  Reference: {_spec_info['kiley_ref']}")

    return


@app.cell
def _(mo):
    mo.md(r"""## Design Matrix Creation and Data Splitting""")
    return


@app.cell
def _(
    avg_lag,
    gbr_config,
    gbr_data,
    lag,
    patsy,
    pd,
    phillips_curve_specs,
    rolling_avg_lag,
    us_config,
    us_data,
    weighted_avg_lag,
):
    # Create design matrices for each country and specification
    design_matrices = {}

    # Transformation functions namespace for Patsy
    _transforms = {
        "lag": lag,
        "avg_lag": avg_lag,
        "weighted_avg_lag": weighted_avg_lag,
        "rolling_avg_lag": rolling_avg_lag,
    }

    # Process US data
    print(f"\n{'=' * 50}")
    print(f"CREATING DESIGN MATRICES FOR {us_config['name'].upper()}")
    print(f"{'=' * 50}")

    design_matrices["US"] = {}

    for _spec_name, _spec_info in phillips_curve_specs.items():
        print(f"\nProcessing specification: {_spec_name}")
        print(f"Formula: {_spec_info['formula']}")

        # Create data with custom functions available
        _data_with_transforms = us_data.copy()

        # Add the transform functions to the environment
        _env = patsy.EvalEnvironment.capture(1)
        for _func_name, _func in _transforms.items():
            _env.namespace[_func_name] = _func

        # Parse formula and create design matrices
        _Y, _X = patsy.dmatrices(
            _spec_info["formula"],
            data=_data_with_transforms,
            eval_env=_env,
            return_type="dataframe",
        )

        # Combine into single dataframe
        _df_reg = pd.concat([_Y, _X.iloc[:, 1:]], axis=1)  # Remove intercept
        _df_reg = _df_reg.dropna()

        # Split data by time periods
        _dates = us_config["date_ranges"]
        _df_pre_2000 = _df_reg.loc[_dates["first_date"] : _dates["pre_2000_end"]]
        _df_post_2000 = _df_reg.loc[_dates["post_2000_start"] : _dates["last_date"]]
        _df_full_sample = _df_reg.loc[_dates["first_date"] : _dates["last_date"]]

        # Store results
        design_matrices["US"][_spec_name] = {
            "full_data": _df_reg,
            "pre_2000": _df_pre_2000,
            "post_2000": _df_post_2000,
            "full_sample": _df_full_sample,
            "y_col": _Y.columns[0],
            "x_cols": _X.columns[1:].tolist(),
            "spec_info": _spec_info,
        }

        print(f"  Full sample: {len(_df_reg)} observations")
        print(f"  Pre-2000: {len(_df_pre_2000)} observations")
        print(f"  Post-2000: {len(_df_post_2000)} observations")

    # Process GBR data
    print(f"\n{'=' * 50}")
    print(f"CREATING DESIGN MATRICES FOR {gbr_config['name'].upper()}")
    print(f"{'=' * 50}")

    design_matrices["GBR"] = {}

    for _spec_name, _spec_info in phillips_curve_specs.items():
        print(f"\nProcessing specification: {_spec_name}")
        print(f"Formula: {_spec_info['formula']}")

        # Create data with custom functions available
        _data_with_transforms = gbr_data.copy()

        # Add the transform functions to the environment
        _env = patsy.EvalEnvironment.capture(1)
        for _func_name, _func in _transforms.items():
            _env.namespace[_func_name] = _func

        # Parse formula and create design matrices
        _Y, _X = patsy.dmatrices(
            _spec_info["formula"],
            data=_data_with_transforms,
            eval_env=_env,
            return_type="dataframe",
        )

        # Combine into single dataframe
        _df_reg = pd.concat([_Y, _X.iloc[:, 1:]], axis=1)  # Remove intercept
        _df_reg = _df_reg.dropna()

        # Split data by time periods
        _dates = gbr_config["date_ranges"]
        _df_pre_2000 = _df_reg.loc[_dates["first_date"] : _dates["pre_2000_end"]]
        _df_post_2000 = _df_reg.loc[_dates["post_2000_start"] : _dates["last_date"]]
        _df_full_sample = _df_reg.loc[_dates["first_date"] : _dates["last_date"]]

        # Store results
        design_matrices["GBR"][_spec_name] = {
            "full_data": _df_reg,
            "pre_2000": _df_pre_2000,
            "post_2000": _df_post_2000,
            "full_sample": _df_full_sample,
            "y_col": _Y.columns[0],
            "x_cols": _X.columns[1:].tolist(),
            "spec_info": _spec_info,
        }

        print(f"  Full sample: {len(_df_reg)} observations")
        print(f"  Pre-2000: {len(_df_pre_2000)} observations")
        print(f"  Post-2000: {len(_df_post_2000)} observations")

    return (design_matrices,)


@app.cell
def _(mo):
    mo.md(r"""## Phillips Curve Estimation""")
    return


@app.cell
def _(design_matrices, pd, run_bayesian_estimation, run_ols_formula):
    # Run regressions for all country-specification combinations
    regression_results = {}

    for _country_code, _country_specs in design_matrices.items():
        print(f"\n{'=' * 60}")
        print(f"ESTIMATING PHILLIPS CURVES FOR {_country_code}")
        print(f"{'=' * 60}")

        regression_results[_country_code] = {}

        for _spec_name, _spec_data in _country_specs.items():
            if _spec_data is None:
                print(f"Input data for {_spec_name} is none")
                continue

            print(f"\n--- Specification: {_spec_name} ---")
            print(f"Description: {_spec_data['spec_info']['description']}")

            # Extract data components
            _df_pre_2000 = _spec_data["pre_2000"]
            _df_post_2000 = _spec_data["post_2000"]
            _df_full_sample = _spec_data["full_sample"]
            _y_col = _spec_data["y_col"]
            _x_cols = _spec_data["x_cols"]

            # Run OLS regressions
            _results_summary = {}

            # Prior estimation (Pre-2000)
            (
                _prior_mean_coeffs,
                _prior_std_errs,
                _prior_vcov_coeffs,
                _prior_sigma_sq_error,
                _X_pre,
                _Y_pre,
                _prior_nobs,
                _,
            ) = run_ols_formula(_df_pre_2000, _y_col, _x_cols)

            _results_summary["OLS Pre-2000 (Prior)"] = {}
            for _col in _X_pre.columns:
                if _col != "const":
                    _results_summary["OLS Pre-2000 (Prior)"][f"{_col}_Coeff"] = (
                        _prior_mean_coeffs[_col]
                    )
                    _results_summary["OLS Pre-2000 (Prior)"][f"{_col}_SE"] = (
                        _prior_std_errs[_col]
                    )
            _results_summary["OLS Pre-2000 (Prior)"]["N"] = _prior_nobs

            # Post-2000 estimation

            (
                _ls_coeffs_post,
                _ls_std_errs_post,
                _,
                _sigma_sq_error_post,
                _X_post,
                _Y_post,
                _post_nobs,
                _,
            ) = run_ols_formula(_df_post_2000, _y_col, _x_cols)

            _results_summary["OLS Post-2000 (Uninf. Prior)"] = {}
            for _col in _X_post.columns:
                if _col != "const":
                    _results_summary["OLS Post-2000 (Uninf. Prior)"][
                        f"{_col}_Coeff"
                    ] = _ls_coeffs_post[_col]
                    _results_summary["OLS Post-2000 (Uninf. Prior)"][f"{_col}_SE"] = (
                        _ls_std_errs_post[_col]
                    )
            _results_summary["OLS Post-2000 (Uninf. Prior)"]["N"] = _post_nobs

            # Full sample estimation
            (
                _full_coeffs,
                _full_std_errs,
                _,
                _,
                _X_full,
                _Y_full,
                _full_nobs,
                _,
            ) = run_ols_formula(_df_full_sample, _y_col, _x_cols)

            _results_summary["OLS Full Sample"] = {}
            for _col in _X_full.columns:
                if _col != "const":
                    _results_summary["OLS Full Sample"][f"{_col}_Coeff"] = _full_coeffs[
                        _col
                    ]
                    _results_summary["OLS Full Sample"][f"{_col}_SE"] = _full_std_errs[
                        _col
                    ]
            _results_summary["OLS Full Sample"]["N"] = _full_nobs

            # Bayesian estimations (only if we have both prior and post-2000 data)
            _weights_on_prior = [0.5, 0.2, 0.05, 0.0]  # Following Kiley (2023)

            for _w_val in _weights_on_prior:
                _bayes_coeffs, _bayes_std_errs = run_bayesian_estimation(
                    _prior_mean_coeffs,
                    _prior_vcov_coeffs,
                    _ls_coeffs_post,
                    _X_post.values,
                    _sigma_sq_error_post,
                    _w_val,
                )

                _results_summary[f"Bayesian (w={_w_val})"] = {}
                for _i, _col in enumerate(_X_post.columns):
                    if _col != "const":
                        _results_summary[f"Bayesian (w={_w_val})"][f"{_col}_Coeff"] = (
                            _bayes_coeffs[_i]
                        )
                        _results_summary[f"Bayesian (w={_w_val})"][f"{_col}_SE"] = (
                            _bayes_std_errs[_i]
                        )
                _results_summary[f"Bayesian (w={_w_val})"]["N"] = _post_nobs

            # Store results
            regression_results[_country_code][_spec_name] = {
                "results_summary": _results_summary,
                "spec_info": _spec_data["spec_info"],
                "data_info": {
                    "pre_2000_obs": len(_df_pre_2000),
                    "post_2000_obs": len(_df_post_2000),
                    "full_sample_obs": len(_df_full_sample),
                },
            }

            # Display results
            _results_df = pd.DataFrame.from_dict(_results_summary, orient="index")
            print(f"\nResults for {_spec_name}:")
            print(_results_df.to_string(float_format="%.3f"))

    return (regression_results,)


@app.cell
def _(go, make_subplots):
    def create_validation_plots(country_code, data, config):
        """
        Creates validation plots for input data.
        """
        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            subplot_titles=("Core CPI Prices", "Unemployment Rate"),
        )

        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data["core_cpi"].pct_change(12),
                name="Core CPI",
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data["unemployment_rate"],
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
            title_text=f"Figure 1. CPI Inflation and Unemployment Rate - {config['name']}",
        )

        return fig

    return (create_validation_plots,)


@app.cell
def _(iterative_forecast, np, pd, process_forecast):
    def create_forecasts(country_code, config, analysis_results, data):
        """
        Creates forecasts for a country.
        """
        forecast_config = config["forecast_config"]
        end_of_history = forecast_config["end_of_history"]
        forecast_steps = forecast_config["forecast_steps"]

        regression_df = analysis_results["regression_df"]
        results_summary = analysis_results["results_summary"]
        weights_on_prior = analysis_results["weights_on_prior"]

        # Prepare forecast inputs
        initial_lags_end = end_of_history
        initial_lags_start = end_of_history - pd.DateOffset(months=11)
        initial_inf_lags = regression_df.loc[initial_lags_start:initial_lags_end, "Y"]

        last_known_unemployment = regression_df.loc[
            end_of_history, "unemployment_rate_lag1"
        ]
        forecast_dates = pd.date_range(
            start=end_of_history, periods=forecast_steps + 1, freq="MS"
        )[1:]
        unemployment_assumption = pd.Series(
            data=last_known_unemployment, index=forecast_dates
        )

        # Generate forecasts
        forecasts = {}
        for w_val in weights_on_prior:
            key = f"Bayesian (w={w_val})"
            coeffs_to_use = pd.Series(
                [
                    results_summary[key]["b(1) Coeff"],
                    results_summary[key]["a Coeff"],
                    results_summary[key]["const Coeff"],
                ]
            )
            fcst = iterative_forecast(
                coeffs_to_use, initial_inf_lags.values, unemployment_assumption
            )
            forecasts[key] = fcst

        # Process forecasts
        all_forecasts_df = pd.DataFrame(index=data.index[data.index > initial_lags_end])
        base_df = pd.DataFrame(index=data.index)
        base_df["core_cpi"] = data["core_cpi"].loc[:initial_lags_end]
        base_df["core_cpi_log"] = np.log(base_df["core_cpi"])
        base_df["delta_p"] = base_df["core_cpi_log"].diff(1) * 100
        base_df["delta_p_fcst"] = base_df["delta_p"]

        for bayes_key in forecasts:
            rebased_series = process_forecast(
                forecasts[bayes_key], base_df, initial_lags_end
            )
            all_forecasts_df[bayes_key] = rebased_series

        historical_data_for_plot = (
            data["core_cpi"].loc[pd.Timestamp("2018-01-01") : initial_lags_end].diff(12)
        )
        historical_data_for_plot.name = "core_cpi_12m"

        return all_forecasts_df, historical_data_for_plot, initial_lags_end

    return


@app.cell
def _(go, pd):
    def create_forecast_plot(
        country_code,
        config,
        all_forecasts_df,
        historical_data_for_plot,
        initial_lags_end,
    ):
        """
        Creates forecast plots for a country.
        """
        fig = go.Figure()

        # Add historical data
        fig.add_trace(
            go.Scatter(
                x=historical_data_for_plot.index,
                y=historical_data_for_plot.values,
                name="Historical Data",
                mode="lines",
            )
        )

        # Add forecast traces
        for col_name in all_forecasts_df.columns:
            fig.add_trace(
                go.Scatter(
                    x=all_forecasts_df.index,
                    y=all_forecasts_df[col_name].values,
                    name=col_name,
                    mode="lines",
                    line=dict(width=2.5, dash="dot"),
                )
            )

        fig.update_layout(
            title=dict(
                text=f"<b>Forecast for Core CPI Inflation - {config['name']}</b>",
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

        fig.add_vrect(
            x0=initial_lags_end + pd.DateOffset(months=1),
            x1=max(all_forecasts_df[s].index[-1] for s in all_forecasts_df.columns),
            annotation_text="Forecast Period",
            annotation_position="top right",
            fillcolor="green",
            opacity=0.1,
            line_width=0,
        )

        return fig

    return


@app.cell
def _(mo):
    mo.md(r"""## Results Summary and Cross-Specification Comparison""")
    return


@app.cell
def _(mo, pd, regression_results):
    # Create comprehensive results table following Kiley (2023) style
    _comparison_data = []

    for _country_code, _country_results in regression_results.items():
        for _spec_name, _spec_results in _country_results.items():
            _results_summary = _spec_results["results_summary"]
            _spec_info = _spec_results["spec_info"]

            for _method, _coeffs in _results_summary.items():
                # Find persistence coefficient (usually the first lag-related coefficient)
                _persistence_coeff = None
                _persistence_se = None
                _unemployment_coeff = None
                _unemployment_se = None

                for _key, _value in _coeffs.items():
                    if "lag(unemployment_rate, 1)" in _key and "Coeff" in _key:
                        _unemployment_coeff = _value
                    elif "lag(unemployment_rate, 1)" in _key and "SE" in _key:
                        _unemployment_se = _value
                    elif (
                        "Coeff" in _key
                        and "unemployment" not in _key
                        and "N" not in _key
                    ):
                        _persistence_coeff = _value
                    elif (
                        "SE" in _key and "unemployment" not in _key and "N" not in _key
                    ):
                        _persistence_se = _value

                if _persistence_coeff is not None:
                    _comparison_data.append(
                        {
                            "Country": _country_code,
                            "Specification": _spec_info["description"],
                            "Method": _method,
                            "Persistence (b1)": f"{_persistence_coeff:.3f} ({_persistence_se:.3f})"
                            if _persistence_se
                            else f"{_persistence_coeff:.3f}",
                            "Unemployment (a)": f"{_unemployment_coeff:.3f} ({_unemployment_se:.3f})"
                            if _unemployment_se
                            else f"{_unemployment_coeff:.3f}",
                            "N": _coeffs.get("N", ""),
                        }
                    )

    if len(_comparison_data) == 0:
        raise ValueError("Data not present in comparison_data -> is empty")

    _comparison_df = pd.DataFrame(_comparison_data)

    # Create summary following Kiley (2023) Table 3 format
    _summary_items = [
        mo.md(r"""
        ### Cross-Specification Results Summary

        **Phillips Curve Estimates Following Kiley (2023)**

        Model: Δp(t) = b(1) * [inflation persistence term] + a * u(t-1) + e(t)

        Standard errors in parentheses.
        """),
        mo.ui.table(_comparison_df, page_size=20),
    ]

    # Add specification-specific insights
    _summary_items.append(
        mo.md(f"""
        ### Key Findings

        **Sample Sizes:**
        - Total specifications tested: {len(_comparison_df["Specification"].unique())}
        - Countries analyzed: {", ".join(_comparison_df["Country"].unique())}

        **Persistence Patterns:**
        The table above shows how inflation persistence (b1 coefficient) varies across:
        1. Different lag specifications (main vs short vs individual lags)
        2. Different estimation periods (pre-2000 vs post-2000 vs full sample)  
        3. Different Bayesian weights on pre-2000 information

        This follows the robustness testing approach in Kiley (2023), where "results were similar for other specifications."
        """)
    )

    mo.vstack(_summary_items)
    return


if __name__ == "__main__":
    app.run()
