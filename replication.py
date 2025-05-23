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
    import statsmodels.api as sm
    return go, make_subplots, mo, np, pd, px, sm


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

    us_data['core_cpi_1m_change'] = us_data['core_cpi'].pct_change(periods=1) * 100
    us_data['core_cpi_log_1m_change'] = np.log(us_data['core_cpi']).pct_change(periods=1) * 100

    us_data['core_cpi_12m_change'] = us_data['core_cpi'].pct_change(periods=12) * 100
    us_data['core_cpi_log_12m_change'] = np.log(us_data['core_cpi']).pct_change(periods=12) * 100

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
def create_lag_features_MONTHLY(df_input):
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

    inflation_col = 'core_cpi_log_12m_change'
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

    # --- Add a constant term for the regression ---
    df['constant'] = 1.0

    return df


@app.cell
def _(full_sample, post_1999_sample, pre_2000_sample):
    full_sample_prepared = create_lag_features_MONTHLY(full_sample)
    pre_2000_sample_prepared = create_lag_features_MONTHLY(pre_2000_sample)
    post_1999_sample_prepared = create_lag_features_MONTHLY(post_1999_sample)
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
def _(estimate_prior, pre_2000_sample_prepared):
    y_col = 'core_cpi_log_12m_change'
    x_cols_table2 = ['constant', 'avg_1_12_lagged_inf', 'unemployment_rate_lag1']
    gamma_tilde, V = estimate_prior(pre_2000_sample_prepared, y_col, x_cols_table2)
    return V, gamma_tilde, x_cols_table2, y_col


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 6. Perform Bayesian Estimation

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

        model_post = sm.OLS(Y_post, X_post).fit()
        gamma_ls_post = model_post.params
        sigma2_post = model_post.mse_resid  # This is sigma^2 for the likelihood
        XtX_post = X_post.T @ X_post

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
    weights_to_test = [0.5, 0.2, 0.05]
    bayesian_results = run_bayesian_estimation(
        post_1999_sample_prepared, y_col, x_cols_table2, gamma_tilde, V, weights_to_test
    )
    return (bayesian_results,)


@app.cell
def _(mo):
    mo.md(r"""## Display results""")
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


if __name__ == "__main__":
    app.run()
