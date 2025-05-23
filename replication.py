import marimo

__generated_with = "0.13.11"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd
    import plotly.express as px
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go
    import seaborn as sns
    import statsmodels.api as sm
    import matplotlib.pyplot as plt
    import pymc as pm
    import scipy.stats as stats
    import arviz as az
    from collections import deque
    import dateutil
    return (
        az,
        deque,
        go,
        make_subplots,
        mo,
        np,
        pd,
        plt,
        pm,
        px,
        sm,
        sns,
        stats,
    )


@app.cell
def _(mo):
    mo.md(
        r"""
    # Replicating FRB Bayesian Policy Maker
    Essentially just replicating this paper. 
    ## Source Material

    - [2023 Update](https://www.federalreserve.gov/econres/notes/feds-notes/a-bayesian-update-on-inflation-and-inflation-persistence-20230707.html) 
    - [2022b note](https://www.federalreserve.gov/econres/notes/feds-notes/anchored-or-not-a-short-summary-of-a-bayesian-approach-to-the-persistence-of-inflation-20220408.html)
    - [2022a working paper](https://www.federalreserve.gov/econres/feds/files/2022016pap.pdf)

    ## Outline

    1. Replicate for the US
    2. Extend for the US
    3. Reproduce for the UK
    4. Iterate and expand
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    # US Replication
    Original paper first
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 1. Load data

    Inputs are: 

    - [Core CPI](https://fred.stlouisfed.org/series/CPILFESL)
    - [Unemployment Rate](https://fred.stlouisfed.org/series/UNRATE)
    """
    )
    return


@app.cell
def _(pd):
    us_unemployment = pd.read_csv('data/Unemployment Rate UNRATE.csv')
    us_unemployment['observation_date'] = pd.to_datetime(us_unemployment['observation_date'])
    us_unemployment.head()
    return (us_unemployment,)


@app.cell
def _(pd):
    us_core_cpi = pd.read_csv('data/Consumer Price Index Less Food and Energy.csv')
    us_core_cpi['observation_date'] = pd.to_datetime(us_core_cpi['observation_date'])
    us_core_cpi.head()
    return (us_core_cpi,)


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 2. Transform data

    - Core CPI: 12 month change in log CPI
    - Unemployment Rate: diff (no change required)
    """
    )
    return


@app.cell
def _(np, pd, us_core_cpi, us_unemployment):
    # Merge the two dataframes on the observation_date column
    us_data = pd.merge(us_unemployment, us_core_cpi, on='observation_date', how='inner')
    us_data = us_data.rename(columns={
        'UNRATE': 'unemployment_rate',
        'CPILFESL': 'core_cpi'
    })
    us_data = us_data.dropna()

    us_data['core_cpi_1m_change'] = us_data['core_cpi'].diff(periods=1)
    us_data['core_cpi_log_1m_change'] = np.log(us_data['core_cpi']).diff(periods=1)

    us_data['core_cpi_12m_change'] = us_data['core_cpi'].diff(periods=12)
    us_data['core_cpi_log_12m_change'] = np.log(us_data['core_cpi']).diff(periods=12)

    us_data.index = us_data['observation_date']

    us_data.head()
    return (us_data,)


@app.cell
def _(mo):
    mo.md(r"""## 3. Summary statistics and presentation""")
    return


@app.cell
def _(px, us_data):
    px.line(
        us_data,
        x='observation_date',
        y='core_cpi_12m_change',
        title='12 Month Change of Core CPI',
        labels={'observation_date': 'Date', 'core_cpi_12m_change': '12 Month Change in Core CPI'},
    ).show()
    return


@app.cell
def _(px, us_data):
    px.line(
        us_data,
        x='observation_date',
        y='unemployment_rate',
        title='12 Month Change in Unemployment Rate',
        labels={'observation_date': 'Date', 'unemployment_rate': 'Unemployment Rate'},
    ).show()

    return


@app.cell
def _(mo):
    mo.md(r"""### Figure 1:  CPI Inflation and the Civilian Unemployment Rate""")
    return


@app.cell
def _(go, make_subplots, us_data):
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=("Core CPI Prices", "Unemployment Rate")
    )

    fig.add_trace(
        go.Scatter(
            x=us_data['observation_date'],
            y=us_data['core_cpi_12m_change'],
            name='Core CPI'
        ),
        row=1,
        col=1
    )

    fig.add_trace(
        go.Scatter(
            x=us_data['observation_date'],
            y=us_data['unemployment_rate'],
            name='Unemployment Rate'
        ),
        row=2,
        col=1
    )

    fig.update_yaxes(title_text="Percent change from<br>12 months earlier", row=1, col=1)
    fig.update_yaxes(title_text="Percent", row=2, col=1)

    fig.update_layout(
        height=750,
        showlegend=False,
        title_text="Figure 1. CPI Inflation and the Civilian Unemployment Rate"
    )

    fig.show()
    return


@app.cell
def _(mo):
    mo.md(r"""### Table 1: Data Summary Statistics:""")
    return


@app.cell
def _(pd, us_data):
    df = us_data.set_index('observation_date')

    full_sample = df.loc['1958-01':'2019-12']
    pre_2000_sample = df.loc['1958-01':'1999-12']
    post_1999_sample = df.loc['2000-01':'2019-12']

    summary_rows = [
        # Full Sample
        {
            'Sample': 'Full sample', 'Variable': 'CPI inflation (percent)',
            'Observations': full_sample['core_cpi_1m_change'].count(),
            'Mean (annual rate)': round(full_sample['core_cpi_12m_change'].mean(), 3),
            'Std. Deviation': round(full_sample['core_cpi_1m_change'].std(), 3),
            'Auto-Correlation': round(full_sample['core_cpi_1m_change'].autocorr(lag=1), 3)
        },
        {
            'Sample': 'Full sample', 'Variable': 'Unemployment rate (percent)',
            'Observations': '-',
            'Mean (annual rate)': round(full_sample['unemployment_rate'].mean(), 3),
            'Std. Deviation': round(full_sample['unemployment_rate'].std(), 3),
            'Auto-Correlation': round(full_sample['unemployment_rate'].autocorr(lag=1), 3)
        },

        # Pre-2000 Sample
        {
            'Sample': 'Pre-2000 sample', 'Variable': 'CPI inflation (percent)',
            'Observations': pre_2000_sample['core_cpi_1m_change'].count(),
            'Mean (annual rate)': round(pre_2000_sample['core_cpi_12m_change'].mean(), 3),
            'Std. Deviation': round(pre_2000_sample['core_cpi_1m_change'].std(), 3),
            'Auto-Correlation': round(pre_2000_sample['core_cpi_1m_change'].autocorr(lag=1), 3)
        },
        {
            'Sample': 'Pre-2000 sample', 'Variable': 'Unemployment rate (percent)',
           'Observations': '-',
           'Mean (annual rate)': round(pre_2000_sample['unemployment_rate'].mean(), 3),
           'Std. Deviation': round(pre_2000_sample['unemployment_rate'].std(), 3),
           'Auto-Correlation': round(pre_2000_sample['unemployment_rate'].autocorr(lag=1), 3)
        },

        # Post-1999 Sample
        {
            'Sample': 'Post-1999 sample', 'Variable': 'CPI inflation (percent)',
            'Observations': post_1999_sample['core_cpi_1m_change'].count(),
            'Mean (annual rate)': round(post_1999_sample['core_cpi_12m_change'].mean(), 3),
            'Std. Deviation': round(post_1999_sample['core_cpi_1m_change'].std(), 3),
            'Auto-Correlation': round(post_1999_sample['core_cpi_1m_change'].autocorr(lag=1), 3)
        },
        {
            'Sample': 'Post-1999 sample', 'Variable': 'Unemployment rate (percent)',
            'Observations': '-',
            'Mean (annual rate)': round(post_1999_sample['unemployment_rate'].mean(), 3),
            'Std. Deviation': round(post_1999_sample['unemployment_rate'].std(), 3),
            'Auto-Correlation': round(post_1999_sample['unemployment_rate'].autocorr(lag=1), 3)
        },
    ]

    summary_df = pd.DataFrame(summary_rows)

    final_rows = [
        # Full Sample Section
        {'Variable': 'Full sample', 'Observations': '', 'Mean (annual rate)': '', 'Std. Deviation': '', 'Auto-Correlation': ''},
        *summary_df[summary_df['Sample'] == 'Full sample'].drop('Sample', axis=1).to_dict('records'),

        # Pre-2000 Section
        {'Variable': 'Pre-2000 sample', 'Observations': '', 'Mean (annual rate)': '', 'Std. Deviation': '', 'Auto-Correlation': ''},
        *summary_df[summary_df['Sample'] == 'Pre-2000 sample'].drop('Sample', axis=1).to_dict('records'),

        # Post-1999 Section
        {'Variable': 'Post-1999 sample', 'Observations': '', 'Mean (annual rate)': '', 'Std. Deviation': '', 'Auto-Correlation': ''},
        *summary_df[summary_df['Sample'] == 'Post-1999 sample'].drop('Sample', axis=1).to_dict('records'),
    ]

    final_table = pd.DataFrame(final_rows)
    final_table
    return full_sample, post_1999_sample, pre_2000_sample


@app.cell
def _(mo):
    mo.md(r"""### Figure 2: Autocorrelations of Core Inflation""")
    return


@app.cell
def _(go, us_data):
    # --- Autocorrelation Calculation ---

    # Isolate the core inflation data for the two periods
    pre_2000_inflation = us_data.loc[:'1999-12-31', 'core_cpi_1m_change']
    post_1999_inflation = us_data.loc['2000-01-01':'2019-12-31', 'core_cpi_1m_change']

    # Define the lags to calculate
    lags = range(1, 13)

    # Calculate the autocorrelation functions (ACF) for each period
    acf_pre_2000 = [pre_2000_inflation.autocorr(lag=l) for l in lags]
    acf_post_1999 = [post_1999_inflation.autocorr(lag=l) for l in lags]

    # --- Chart Generation ---

    fig_autocorr = go.Figure()

    # Add bar trace for Pre-2000 data
    fig_autocorr.add_trace(go.Bar(
        x=list(lags),
        y=acf_pre_2000,
        name='Pre-2000 data',
        marker_color='#1f4e79'  # Dark blue
    ))

    # Add bar trace for Post-1999 data
    fig_autocorr.add_trace(go.Bar(
        x=list(lags),
        y=acf_post_1999,
        name='Post-1999 data',
        marker_color='#a9a9a9'  # Gray
    ))

    # Update layout to match the figure style
    fig_autocorr.update_layout(
        title_text='Figure 2. Autocorrelations of Core Inflation',
        xaxis_title='Lag',
        yaxis_range=[-0.1, 0.7],
        barmode='group',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.3, # Position legend below the x-axis title
            xanchor="center",
            x=0.5
        ),
        template='simple_white', # Use a clean template
        margin=dict(b=100) # Add bottom margin for legend
    )

    # Ensure x-axis ticks are integers from 1 to 12
    fig_autocorr.update_xaxes(tickvals=list(lags))

    fig_autocorr.show()
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 4. Data prep

    Create the regression df with the lagged data as required
    """
    )
    return


@app.function
def create_lag_features_MONTHLY(df_input, inflation_col_name):
    """
    Prepares a DataFrame for Phillips Curve estimation by creating lagged variables.

    Args:
        df_input (pd.DataFrame): Input DataFrame with at least 'core_cpi_log_12m_change'
                                 and 'unemployment_rate' columns. It is assumed that
                                 the DataFrame is sorted by time in ascending order.

    Returns:
        pd.DataFrame: A new DataFrame with the original data plus the
                      newly created lagged features and a constant term.
    """
    df = df_input.copy()

    inflation_col = inflation_col_name
    unemployment_col = 'unemployment_rate'

    # Lagged Unemployment (u_{t-1})
    df['unemployment_rate_lag1'] = df[unemployment_col].shift(1)

    # Lagged Inflation for Monthly Analysis (Main Specification from Table 2)
    # avg_1_12_lagged_inf = average of core_cpi_log_12m_change(t-1) to core_cpi_log_12m_change(t-12)

    for i in range(1, 13):
        df[f'{inflation_col}_lag{i}'] = df[inflation_col].shift(i)

    # Calculate the average of lags 1 through 12
    lag_cols_1_12 = [f'{inflation_col}_lag{i}' for i in range(1, 13)]
    df['avg_1_12_lagged_inf'] = df[lag_cols_1_12].mean(axis=1)

    # Lagged Inflation for Alternative Monthly Lag Structure (Table 3)
    # avg_1_3_lagged_inf = average of core_cpi_log_12m_change(t-1) to core_cpi_log_12m_change(t-3)
    # This is (1/3) * sum_{j=1}^{3} delta_p(t-j)
    lag_cols_1_3 = [f'{inflation_col}_lag{i}' for i in range(1, 4)] # Lags 1, 2, 3
    df['avg_1_3_lagged_inf'] = df[lag_cols_1_3].mean(axis=1)

    # avg_4_12_lagged_inf = average of core_cpi_log_12m_change(t-4) to core_cpi_log_12m_change(t-12)
    # This is (1/9) * sum_{j=4}^{12} delta_p(t-j)
    lag_cols_4_12 = [f'{inflation_col}_lag{i}' for i in range(4, 13)] # Lags 4 through 12
    df['avg_4_12_lagged_inf'] = df[lag_cols_4_12].mean(axis=1)

    # Clean up intermediate individual lag columns
    individual_lag_cols_to_drop = [f'{inflation_col}_lag{i}' for i in range(1, 13)]
    df = df.drop(columns=individual_lag_cols_to_drop)

    # # --- Add a constant term for the regression ---
    # df['constant'] = 1.0

    return df


@app.cell
def _():
    y_col = 'core_cpi_log_1m_change'
    return (y_col,)


@app.cell
def _(full_sample, post_1999_sample, pre_2000_sample, y_col):
    full_sample_prepared = create_lag_features_MONTHLY(full_sample, y_col)
    pre_2000_sample_prepared = create_lag_features_MONTHLY(pre_2000_sample, y_col)
    post_1999_sample_prepared = create_lag_features_MONTHLY(post_1999_sample, y_col)
    return post_1999_sample_prepared, pre_2000_sample_prepared


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 5. Estimate priors

    Estimate the prior from the pre-2000 data using Ordinary Least Squares (OLS).

    We're looking for the least-squares estimate of the lags and the associated vcov matrix.

    Explanation:
    > Run a standard Ordinary Least Squares (OLS) regression on the pre_2000_sample_prepared data. The resulting coefficients will serve as the mean of the prior (Γ~), and the variance-covariance matrix of those coefficients will be the variance of the prior (V). This formalizes the "information from the 1960-1999 period."
    """
    )
    return


@app.cell
def _(sm):
    def estimate_prior(df_pre_2000_prepared, y_col, x_cols):
        """
        Estimates the prior distribution parameters from the pre-2000 sample.

        Args:
            df_pre_2000_prepared (pd.DataFrame): The pre-2000 data with lags created.
            y_col (str): The name of the dependent variable (inflation).
            x_cols (list): A list of names for the independent variables.

        Returns:
            tuple: A tuple containing:
                - pd.Series: The prior mean vector (gamma_tilde).
                - pd.DataFrame: The prior variance-covariance matrix (V).
        """
        # Drop rows with NaN values to get a clean sample for regression
        df = df_pre_2000_prepared.dropna(subset=[y_col] + x_cols).copy()

        Y = df[y_col]
        X = df[x_cols]

        # Fit the OLS model
        model = sm.OLS(Y, X).fit()

        # The prior mean (gamma_tilde) is the vector of estimated coefficients
        gamma_tilde = model.params

        # The prior variance (V) is the variance-covariance matrix of the coefficients
        V = model.cov_params()

        print("--- Prior Parameters Estimated from Pre-2000 Data ---")
        print(f"Model Summary: \n{model.summary()}")
        print("Prior Mean (gamma_tilde):\n", gamma_tilde)
        print("\nPrior Variance-Covariance Matrix (V):\n", V)

        return gamma_tilde, V
    return (estimate_prior,)


@app.cell
def _(estimate_prior, pre_2000_sample_prepared, y_col):
    x_cols_table2 = ['avg_1_12_lagged_inf', 'unemployment_rate_lag1']
    gamma_tilde, V = estimate_prior(pre_2000_sample_prepared, y_col, x_cols_table2)
    return V, gamma_tilde, x_cols_table2


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 6. Perform Bayesian Estimation

    ### A) Manual version

    Explanation:
    > Use those prior parameters (Γ~ and V) and combine them with the data from post_1999_sample_prepared using the Bayesian regression formula. By iterating through different weights (w), we can see how the final estimates change based on how much "conviction" we place on the pre-2000 experience.
    """
    )
    return


@app.cell
def _(np, pd, sm):
    def run_bayesian_estimation(df_post_1999_prepared, y_col, x_cols, gamma_tilde, V, weights):
        """
        Performs the Bayesian estimation for various prior weights.

        Args:
            df_post_1999_prepared (pd.DataFrame): The post-1999 data with lags.
            y_col (str): The name of the dependent variable.
            x_cols (list): A list of names for the independent variables.
            gamma_tilde (pd.Series): The prior mean vector.
            V (pd.DataFrame): The prior variance-covariance matrix.
            weights (list): A list of floats representing the weights (w) on the prior.

        Returns:
            dict: A dictionary where keys are the weights and values are dicts
                  containing the posterior mean and standard errors.
        """
        # --- First, get the OLS estimates from the post-1999 data ---
        df = df_post_1999_prepared.dropna(subset=[y_col] + x_cols).copy()
        Y_post = df[y_col]
        X_post = df[x_cols]

        # print("Model Inputs:")
        # print(Y_post.tail())
        # print(X_post.tail())

        model_post = sm.OLS(Y_post, X_post).fit()
        gamma_ls_post = model_post.params
        sigma2_post = model_post.mse_resid  # This is sigma^2 for the likelihood
        XtX_post = X_post.T @ X_post

        # print("Model Outputs:")
        # print(gamma_ls_post)
        # print(sigma2_post)

        # Invert the prior variance matrix
        V_inv = np.linalg.inv(V.to_numpy())

        results = {}

        print("\n--- Running Bayesian Estimation for Post-1999 Data ---")

        for w in weights:
            # Precision = Inverse of the variance
            # Posterior precision = weighted sum of prior precision and data (likelihood) precision
            prior_precision = w * V_inv
            data_precision = (1 - w) * (1 / sigma2_post) * XtX_post

            posterior_precision = prior_precision + data_precision
            posterior_variance = np.linalg.inv(posterior_precision)

            # Posterior mean = weighted average of prior mean and data mean
            # Weights are the precision matrices.
            prior_mean_component = w * V_inv @ gamma_tilde.to_numpy()
            data_mean_component = (1 - w) * (1 / sigma2_post) * XtX_post @ gamma_ls_post.to_numpy()

            posterior_mean = posterior_variance @ (prior_mean_component + data_mean_component)

            # Standard errors are the square root of the diagonal of the posterior variance matrix
            posterior_se = np.sqrt(np.diag(posterior_variance))

            results[w] = {
                'posterior_mean': pd.Series(posterior_mean, index=x_cols),
                'posterior_se': pd.Series(posterior_se, index=x_cols)
            }

        # Add the uninformative prior case (equivalent to w=0, or just OLS on post-1999)
        results[0] = {
            'posterior_mean': gamma_ls_post,
            'posterior_se': model_post.bse
        }
        print(model_post.summary())
        return results

    return (run_bayesian_estimation,)


@app.cell
def _(
    V,
    gamma_tilde,
    post_1999_sample_prepared,
    run_bayesian_estimation,
    x_cols_table2,
    y_col,
):
    weights_to_test = [0.5, 0.2, 0.05, 0]
    bayesian_results = run_bayesian_estimation(
        post_1999_sample_prepared, y_col, x_cols_table2, gamma_tilde, V, weights_to_test
    )
    return bayesian_results, weights_to_test


@app.cell
def _(mo):
    mo.md(r"""### Display results - Manual Version""")
    return


@app.cell
def _(bayesian_results, pd):
    print("\n--- Final Posterior Estimates (Replicating Table 2) ---")
    for w, result in sorted(bayesian_results.items(), reverse=True):
        label = f"Weight on Prior = {w}" if w > 0 else "Uninformative Prior (OLS)"
        print(f"\n--- {label} ---")

        # Combine mean and SE into a single DataFrame for nice printing
        results_df = pd.DataFrame({
            'Coefficient': result['posterior_mean'],
            'Std. Error': result['posterior_se']
        })
        print(results_df)

    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### PyMC version
    This doesn't exactly replicate how the paper implements it, but uses the more flexible / powerful pyMC module and MCMC sampling to draw from the posterior. This means we could extend this scenario to situations with no analytical solution.
    """
    )
    return


@app.cell
def _(pm):
    def run_pymc_estimation(df_post_1999_prepared, y_col, x_cols, gamma_tilde, V_base, w):
        """
        Performs Bayesian estimation for a single prior weight w using PyMC.

        Args:
            df_post_1999_prepared (pd.DataFrame): Post-1999 data with lags.
            y_col (str): Dependent variable name.
            x_cols (list): Independent variable names.
            gamma_tilde (pd.Series): Prior mean vector.
            V_base (pd.DataFrame): The base prior variance-covariance matrix (for w=0.5).
            w (float): The weight on the prior.

        Returns:
            arviz.InferenceData: The trace object containing posterior samples.
        """
        # Per Kiley (2022), a weight 'w' is equivalent to scaling the prior variance
        # by (1-w)/w.
        if w == 1.0: scale_factor = 0.0001
        elif w == 0.0: scale_factor = 1e6
        else: scale_factor = (1 - w) / w

        V_scaled = V_base * scale_factor

        # Prepare data for PyMC model
        df = df_post_1999_prepared.dropna(subset=[y_col] + x_cols).copy()
        Y_obs = df[y_col].values
        X_obs = df[x_cols].values

        print(f"\n--- Running PyMC Estimation for weight w = {w} ---")
        print(f"Prior variance scale factor = {scale_factor:.2f}")
        with pm.Model() as model_kiley:
            # Priors
            # The prior for our coefficients (betas) is a multivariate normal
            # distribution defined by the OLS results on the pre-2000 data.
            betas = pm.MvNormal('betas', mu=gamma_tilde.values, cov=V_scaled, shape=len(x_cols))

            # We also need a prior for the error variance (sigma) of the likelihood
            sigma = pm.HalfCauchy('sigma', beta=1)

            # Likelihood
            # The expected value of Y is the linear combination of X and our betas
            mu = pm.math.dot(X_obs, betas)

            # The likelihood function defines how the data is generated.
            # It's a Normal distribution centered at our expected value 'mu'.
            likelihood = pm.Normal('likelihood', mu=mu, sigma=sigma, observed=Y_obs)

            # Sampling
            # PyMC's NUTS sampler will draw samples from the posterior distribution
            trace = pm.sample(5000, tune=1000, chains=4, cores=4, target_accept=0.95, progressbar="combined+stats")

        return trace
    return (run_pymc_estimation,)


@app.cell
def _(mo):
    mo.md(r"""### Display results - PyMC Version""")
    return


@app.cell
def _(
    V,
    az,
    gamma_tilde,
    pd,
    post_1999_sample_prepared,
    run_pymc_estimation,
    weights_to_test,
    x_cols_table2,
    y_col,
):
    pymc_results = {}
    for w_val in weights_to_test:
        pymc_results[w_val] = run_pymc_estimation(
            post_1999_sample_prepared, y_col, x_cols_table2, gamma_tilde, V, w_val
        )

    # 3. Display results using ArviZ
    for w_val, trace in pymc_results.items():
        print(f"\n--- Posterior Summary for weight w = {w_val} ---")
        summary = az.summary(trace, var_names=['betas'])
        # Add labels for coefficients
        summary.index = pd.Index(x_cols_table2, name="Coefficient")
        print(summary)
    return (pymc_results,)


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 7. Visualise Results

    Visualise the posterior distributions (estimates of parameters)
    """
    )
    return


@app.cell
def _(np, stats):
    def plot_analytical_distribution(ax, posterior_mean, posterior_se, label_dist):
        """
        Plots a normal PDF on a given matplotlib axes.
        Legend is handled outside this function.
        """
        x_vals = np.linspace(posterior_mean - 4 * posterior_se, posterior_mean + 4 * posterior_se, 300)
        y_vals = stats.norm.pdf(x_vals, loc=posterior_mean, scale=posterior_se)
        ax.plot(x_vals, y_vals, color='red', linestyle='--', linewidth=2.5, label=label_dist)
    return (plot_analytical_distribution,)


@app.cell
def _(bayesian_results, pymc_results, x_cols_table2):
    param_to_plot_name = 'unemployment_rate_lag1' # unemployment_rate_lag1 avg_1_12_lagged_inf
    param_to_plot_index = x_cols_table2.index(param_to_plot_name)
    weight_to_plot = 0.5 # 0.5 0.2 0.05

    # Get the results for the chosen weight
    analytical_result = bayesian_results[weight_to_plot]
    pymc_trace = pymc_results[weight_to_plot]

    # Extract the specific mean and SE for the parameter from the analytical results
    analytical_mean = analytical_result['posterior_mean'][param_to_plot_name]
    analytical_se = analytical_result['posterior_se'][param_to_plot_name]

    # Extract the specific mean and se for the parameter from the PyMC results
    pymc_samples_for_param = pymc_trace.posterior['betas'].sel(betas_dim_0=param_to_plot_index)
    pymc_mean = pymc_samples_for_param.mean().item()
    pymc_se = pymc_samples_for_param.std().item() # Standard deviation of the PyMC samples
    return (
        analytical_mean,
        analytical_se,
        param_to_plot_name,
        pymc_mean,
        pymc_samples_for_param,
        pymc_se,
        weight_to_plot,
    )


@app.cell
def _(
    analytical_mean,
    analytical_se,
    az,
    param_to_plot_name,
    plot_analytical_distribution,
    plt,
    pymc_mean,
    pymc_samples_for_param,
    pymc_se,
    weight_to_plot,
):
    fig1, axes_fig1 = plt.subplots(2, 1, figsize=(12, 10), sharex=True, gridspec_kw={'height_ratios': [3, 1]}) 
    ax_dist = axes_fig1[0] # Axes for distribution plot
    ax_errorbar = axes_fig1[1] # Axes for error bar plot

    az.plot_dist(
        pymc_samples_for_param,
        ax=ax_dist,
        label=f'PyMC Posterior (w={weight_to_plot})', 
        color='C0',
        hist_kwargs={'alpha': 0.6}, 
    )

    # 2. Overlay the analytical distribution on the same axes
    plot_analytical_distribution(
        ax=ax_dist,
        posterior_mean=analytical_mean,
        posterior_se=analytical_se,
        label_dist=f'Analytical Posterior (w={weight_to_plot})'
    )

    # 3. Add vertical lines for means on the first subplot
    ax_dist.axvline(pymc_mean, color='C0', linestyle=':', linewidth=2, label=f'PyMC Mean: {pymc_mean:.3f}')
    ax_dist.axvline(
        analytical_mean, 
        color='red', 
        linestyle=':', 
        linewidth=2, 
        label=f'Analytical Mean: {analytical_mean:.3f}'
    )

    # 4. Add text labels for means on the first subplot
    ylim_dist = ax_dist.get_ylim()
    text_y_pos_dist = ylim_dist[1] * 0.9
    ax_dist.text(pymc_mean, text_y_pos_dist, f'{pymc_mean:.3f}', color='C0', ha='center', va='bottom', fontsize=9, backgroundcolor='white')
    ax_dist.text(analytical_mean, text_y_pos_dist * 0.95, f'{analytical_mean:.3f}', color='red', ha='center', va='bottom', fontsize=9, backgroundcolor='white')

    ax_dist.set_title(f"Posterior for '{param_to_plot_name}' (Prior Weight = {weight_to_plot})", fontsize=14)
    ax_dist.set_ylabel("Density", fontsize=12)
    ax_dist.tick_params(axis='y', which='major', labelsize=10) # Only y-axis for top plot
    ax_dist.legend(fontsize=9) 
    ax_dist.grid(True, linestyle=':', alpha=0.7)

    # 5. Create error bar plot on the second subplot
    errorbar_y_positions = [0.6, 0.3] # To separate the two error bars vertically
    ax_errorbar.errorbar(x=pymc_mean, y=errorbar_y_positions[0], xerr=pymc_se, fmt='o', color='C0', 
                         capsize=5, markersize=8, label=f'PyMC: {pymc_mean:.3f} ± {pymc_se:.3f}')
    ax_errorbar.errorbar(x=analytical_mean, y=errorbar_y_positions[1], xerr=analytical_se, fmt='s', color='red', 
                         capsize=5, markersize=8, label=f'Analytical: {analytical_mean:.3f} ± {analytical_se:.3f}')

    ax_errorbar.set_yticks(errorbar_y_positions)
    ax_errorbar.set_yticklabels(['PyMC', 'Analytical'])
    ax_errorbar.set_xlabel(param_to_plot_name, fontsize=12)
    ax_errorbar.set_title("Mean ± Standard Error", fontsize=12)
    ax_errorbar.tick_params(axis='x', which='major', labelsize=10)
    ax_errorbar.legend(fontsize=9, loc='upper center', bbox_to_anchor=(0.5, -0.25), ncol=2) # Adjust legend position
    ax_errorbar.grid(True, linestyle=':', alpha=0.5, axis='x') # Grid only on x-axis
    ax_errorbar.set_ylim(0, 1) # Adjust y-limit for clarity

    plt.tight_layout(rect=[0, 0.03, 1, 0.97]) # Adjust layout to prevent overlap
    plt.show()
    return


@app.cell
def _(bayesian_results, np, pd, plt, pymc_results, x_cols_table2):
    def _():
        # --- 1. Data Preparation ---
        param_names = x_cols_table2
        weights = sorted(bayesian_results.keys())
        num_params = len(param_names)
        num_weights = len(weights)

        plot_data = []
        for i, param in enumerate(param_names):
            for w in weights:
                analytical_res = bayesian_results[w]
                pymc_trace = pymc_results[w]

                pymc_samples = pymc_trace.posterior['betas'].sel(betas_dim_0=i)

                plot_data.append({
                    'parameter': param,
                    'weight': w,
                    'analytical_mean': analytical_res['posterior_mean'][param],
                    'analytical_se': analytical_res['posterior_se'][param],
                    'pymc_mean': pymc_samples.mean().item(),
                    'pymc_se': pymc_samples.std().item(),
                })

        df = pd.DataFrame(plot_data)

        # --- 2. Visualization ---
        # Create subplots with independent x-axes by setting sharex=False
        fig, axes = plt.subplots(
            nrows=num_params, 
            ncols=1, 
            figsize=(10, num_params * 3), # Increased vertical space slightly
            sharex=False # Each subplot will have its own x-axis scale
        )
        if num_params == 1:
            axes = [axes]

        y_ticks = np.arange(num_weights)
        y_offset = 0.1

        for i, param in enumerate(param_names):
            ax = axes[i]
            param_df = df[df['parameter'] == param]

            ax.errorbar(
                x=param_df['analytical_mean'],
                y=y_ticks - y_offset,
                xerr=param_df['analytical_se'],
                fmt='s',
                color='red',
                capsize=4,
                label='Analytical'
            )

            ax.errorbar(
                x=param_df['pymc_mean'],
                y=y_ticks + y_offset,
                xerr=param_df['pymc_se'],
                fmt='o',
                color='C0',
                capsize=4,
                label='PyMC'
            )

            ax.axvline(0, color='black', linestyle='--', linewidth=1, alpha=0.7)
            ax.set_yticks(y_ticks)
            ax.set_yticklabels([f'w={w}' for w in weights])
            ax.set_ylabel('Prior Weight')
            ax.set_xlabel('Coefficient Value (Mean ± SE)') # Add label to each subplot
            ax.set_title(f"Posterior Estimates for '{param}'")
            ax.grid(axis='x', linestyle=':', alpha=0.6)

        # --- 3. Final Touches ---
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(0.98, 0.98))
        fig.suptitle('Comparison of Posteriors Across All Weights and Parameters', fontsize=16, y=1.0)
        plt.tight_layout(rect=[0, 0, 1, 0.98]) # Adjust layout to fit suptitle
        return plt.show()


    _()
    return


@app.cell
def _(bayesian_results, np, pd, plt, pymc_results, sns, x_cols_table2):
    def _():
        # --- 1. Consolidate All Samples into a DataFrame ---
        # This is the most robust way to prepare data for Seaborn.
        all_samples_list = []
        param_names = x_cols_table2
        weights = sorted(bayesian_results.keys())
        n_analytical_samples = 200000 # Number of samples to draw for the analytical normal distribution

        for i, param in enumerate(param_names):
            for w in weights:
                # Get PyMC samples
                pymc_trace = pymc_results[w]
                pymc_samples = pymc_trace.posterior['betas'].sel(betas_dim_0=i).values.flatten()
                all_samples_list.append(pd.DataFrame({
                    'value': pymc_samples,
                    'weight': w,
                    'parameter': param,
                    'method': 'PyMC'
                }))

                # Generate samples for the analytical distribution
                analytical_res = bayesian_results[w]
                analytical_mean = analytical_res['posterior_mean'][param]
                analytical_se = analytical_res['posterior_se'][param]
                analytical_samples = np.random.normal(analytical_mean, analytical_se, n_analytical_samples)
                all_samples_list.append(pd.DataFrame({
                    'value': analytical_samples,
                    'weight': w,
                    'parameter': param,
                    'method': 'Analytical'
                }))

        samples_df = pd.concat(all_samples_list, ignore_index=True)

        # --- 2. Create the Visualization ---
        num_params = len(param_names)
        fig, axes = plt.subplots(
            nrows=num_params,
            ncols=1,
            figsize=(12, num_params * 4.5),
            sharex=False # Allow independent x-axes for each parameter
        )
        if num_params == 1:
            axes = [axes]

        # Define a consistent color palette
        palette = {'PyMC': 'C0', 'Analytical': 'red'}

        for i, param in enumerate(param_names):
            ax = axes[i]
            param_df = samples_df[samples_df['parameter'] == param]

            # Use a split violin plot to compare distributions directly
            sns.violinplot(
                data=param_df,
                x='weight',
                y='value',
                hue='method',
                split=True,
                ax=ax,
                palette=palette,
                inner='quartiles', # Shows the 25th, 50th (median), and 75th percentiles
                linewidth=1.2
            )

            # --- 3. Customization and Annotation ---
            ax.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.7)
            ax.set_title(f"Posterior Distributions for '{param}'", fontsize=14, pad=15)
            ax.set_xlabel('Prior Weight', fontsize=12)
            ax.set_ylabel('Coefficient Value', fontsize=12)
            ax.grid(axis='y', linestyle=':', alpha=0.6)

            # Clean up the legend
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles, labels, title='Method', loc='upper right')


        fig.suptitle('Side-by-Side Posterior Distributions', fontsize=18, y=1.0)
        plt.tight_layout(rect=[0, 0, 1, 0.98])
        return plt.show()


    _()
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 8. Out of sample forecast using estimated parameters

    1. Get historical inflation
    2. Get conditioning path of unemployment
    3. Plug into Phillips curve for each parameter, get numbers out and plot.
    """
    )
    return


@app.cell
def _(us_data):
    # Replicate fig 5
    us_data

    # We're using y_col as our target for all this.
    return


@app.cell
def _(deque, np, pd):
    def iterative_forecast(coeffs, initial_inf_lags, unemployment_lags):
        """
        Generates multi-step ahead forecasts for inflation as a pandas Series.

        Args:
            coeffs (pd.Series): Model coefficients.
            initial_inf_lags (list or np.array): The 12 most recent actual inflation values.
            unemployment_lags (pd.Series): Assumed unemployment rates for the forecast horizon,
                                           with a DatetimeIndex.

        Returns:
            pd.Series: A Series of forecasted inflation values with a DatetimeIndex.
        """
        num_lags = len(initial_inf_lags)
        inflation_lags = deque(initial_inf_lags, maxlen=num_lags)
        forecasts = {}

        inf_coeff, unemp_coeff = coeffs

        for date, u_lag in unemployment_lags.items():
            avg_inf_lag = np.mean(inflation_lags)
            forecast = (inf_coeff * avg_inf_lag) + (unemp_coeff * u_lag)

            forecasts[date] = forecast
            inflation_lags.append(forecast)

        return pd.Series(forecasts, name='inflation_forecast')
    return (iterative_forecast,)


@app.cell
def _(
    bayesian_results,
    iterative_forecast,
    pd,
    us_data,
    weights_to_test,
    y_col,
):
    # Jan 2022. Then May 2023, then full data.
    end_of_history = pd.to_datetime('2021-12')
    forecast_steps = 12

    initial_lags_end = end_of_history
    initial_lags_start = end_of_history - pd.DateOffset(months=11)
    initial_inf_lags = us_data.loc[initial_lags_start:initial_lags_end, y_col].values

    # Set our unemployment assumption (constant at last value)
    last_known_unemployment = us_data.loc[end_of_history, 'unemployment_rate']

    # Create the date range for the 12-month forecast
    forecast_dates = pd.date_range(
        start=end_of_history, 
        periods=forecast_steps + 1, 
        freq='MS'
    )[1:]

    # Create the unemployment Series for the forecast period
    unemployment_assumption = pd.Series(
        data=last_known_unemployment, 
        index=forecast_dates
    )

    forecasts = {}
    for w_val_2 in weights_to_test:
        coeffs_to_use = bayesian_results[w_val_2]['posterior_mean']

        print(w_val_2)
        print(coeffs_to_use)

        fcst = iterative_forecast(
            coeffs_to_use,
            initial_inf_lags,
            unemployment_assumption
        )

        forecasts[w_val_2] = fcst


    return forecasts, initial_inf_lags


@app.cell
def _(forecasts, go, us_data, y_col):
    def plot_inflation_forecasts_plotly(historical_data, forecast_dict):
        """
        Creates an interactive forecast chart using Plotly.

        Args:
            historical_data (pd.Series): A Series of actual historical inflation data.
            forecast_dict (dict): A dictionary where keys are weights (float) and
                                  values are the corresponding forecast Series.
        """
        # last_hist_date = historical_data.index[-1]
        # print(last_hist_date)
        # print()

        fig = go.Figure()

        # 1. Add the historical data trace
        fig.add_trace(go.Scatter(
            x=historical_data.index,
            y=historical_data,
            name='Historical Data',
            mode='lines',
            line=dict(color='#1f77b4', width=4),
        ))

        # 2. Add a trace for each forecast series
        for weight, forecast_series in forecast_dict.items():

            label = f"Forecast (Weight = {weight:.2f})"
            if weight == 0:
                label = "Forecast (Zero Weight)"

            fig.add_trace(go.Scatter(
                x=forecast_series.index,
                y=forecast_series.values,
                name=label,
                mode='lines',
                line=dict(width=2.5, dash='dot'),
            ))

        # 3. Update layout
        fig.update_layout(
            title=dict(
                text='<b> Forecast for Core CPI Inflation using estimated Phillips curve</b>',
                x=0.5,
                font=dict(size=22, family='Times New Roman')
            ),
            yaxis_title='Year-over-Year Percent Change',
            yaxis_ticksuffix='%',
            template='plotly_white',
            legend=dict(
                orientation='h',
                yanchor='bottom',
                y=1.0,
                xanchor='right',
                x=1,
                font=dict(size=12, family="Times New Roman")
            ),
            hovermode='x unified'
        )

        # 4. Add a shaded rectangle for the forecast period
        fig.add_vrect(
            x0=min(s.index[0] for s in forecast_dict.values()),
            x1=max(s.index[-1] for s in forecast_dict.values()),
            annotation_text="Forecast Period",
            annotation_position="top left",
            fillcolor="green",
            opacity=0.1,
            line_width=0
        )

        fig.show()

    plot_inflation_forecasts_plotly(
        us_data.loc["January 2019":"December 2021", y_col],
        forecasts
    )
    return


@app.cell
def _(us_data, y_col):
    us_data.loc["January 2021":"January 2022",y_col]
    return


@app.cell
def _(initial_inf_lags):
    initial_inf_lags
    return


if __name__ == "__main__":
    app.run()
