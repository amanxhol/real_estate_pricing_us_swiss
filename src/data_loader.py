import pandas as pd
from statsmodels.tsa.stattools import grangercausalitytests
import seaborn as sns
import matplotlib.pyplot as plt

# Install openpyxl to read .xlsx files


# Data loader

def load_data_swiss(path):
    """
    Loads the Excel file and prepares the columns.
    Column A = date
    Column G = real estate (converted to USD)
    Column H to M = materials
    Column N = daily USD/CHF exchange rate
    """
    df = pd.read_excel(path)

    """print("=== Preview of the loaded file ===")
    print(df.head())"""

    # Convert the date column
    df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0])

    # Define columns
    dates = df.iloc[:, 0]                 # Column A
    immobilier = df.iloc[:, 6] * df.iloc[:, 15]  # Column G converted to USD
    # real estate = df.iloc[:, 17] * df.iloc[:, 15]
    materials = df.iloc[:, 7:13]          # Column H to M

    # Drop rows with NaN
    data = pd.concat([dates, immobilier, materials], axis=1).dropna()

    # Rename columns
    data.columns = ["date", "immobilier_usd"] + list(materials.columns)

    """print("\n=== Preview after selection and conversion to USD ===")
    print(data.head())"""

    return data

def load_data_us(path, immobilier_col_index=17, immobilier_date_col_index=16):
    """
    Loads data from Excel and prepares a consistent DataFrame.

    Parameters:
    - path: path to the Excel file
    - property_col_index: index of the property price column (0-based)
    - immobilier_date_col_index: index of the date column corresponding to the property (0-based)

    Notes:
    - Material dates are in column 0
    - Materials are in columns 7 to 12 (H to M)
    """
    df = pd.read_excel(path)

    # --- Materials columns ---
    materials_dates = pd.to_datetime(df.iloc[:, 0])      # column 0 = materials date
    materials = df.iloc[:, 7:13]                         # columns H to M

    # --- Real estate columns ---
    immobilier_dates = pd.to_datetime(df.iloc[:, immobilier_date_col_index])
    immobilier = df.iloc[:, immobilier_col_index]        # real estate price
    
    # immobilier = df.iloc[:, immobilier_col_index] * df.iloc[:, 15]

    # --- Monthly aggregation of materials to match real estate ---
    materials.index = materials_dates
    materials_monthly = materials.resample('M').mean()   # monthly average
    materials_monthly = materials_monthly.reset_index()

    # --- Merge with the property price ---
    data = pd.concat(
        [
            immobilier_dates.reset_index(drop=True),
            immobilier.reset_index(drop=True),
            materials_monthly.iloc[:, 1:].reset_index(drop=True)
        ],
        axis=1
    )

    # Drop NaN rows
    data = data.dropna()

    # Rename columns
    data.columns = ["date", "immobilier_usd"] + list(materials.columns)

    return data



# Monthly aggregation

def aggregate_monthly(data):
    """
    Aggregates daily data into monthly averages.
    """
    data_monthly = data.copy()
    data_monthly = data_monthly.resample('ME', on='date').mean().reset_index()

    """print("\n=== Overview after monthly aggregation ===")
    print(data_monthly.head())"""

    return data_monthly


# Create monthly lags

def create_monthly_lags(data, lags=[1, 2, 3]):
    """
    Creates lagged columns for each material.
    """
    data_lagged = data.copy()
    material_cols = data_lagged.columns[2:8]  # Columns H to M

    for lag in lags:
        for col in material_cols:
            data_lagged[f"{col}_lag{lag}m"] = data_lagged[col].shift(lag)

    # Remove rows containing NaNs due to lags
    data_lagged = data_lagged.dropna()

    """print("\n=== Preview after creating lags ===")
    print(data_lagged.head())"""

    return data_lagged


def select_features_by_lag(data, lag):
    """
    Selects only the columns corresponding to the given lag.
    """
    feature_cols = [col for col in data.columns if f"_lag{lag}m" in col]
    return data[feature_cols]


def compute_percentage_variation(data, exclude_cols=["date"]):
    """
    Computes the monthly percentage change for all numeric columns
    (except the date).

    percentage_variation = (P_t - P_{t-1}) / P_{t-1}
    """
    data_var = data.copy()

    value_cols = [col for col in data_var.columns if col not in exclude_cols]

    data_var[value_cols] = data_var[value_cols].pct_change()

    # Drop the first row (NaN)
    data_var = data_var.dropna().reset_index(drop=True)

    """print("\n Preview of monthly percentage changes (%)")
    print(data_var.head())"""

    return data_var



def compute_correlations(data):
    """
    Computes the correlation between real estate prices and lagged materials.
    """
    target = "immobilier_usd"

    # Drop the date column if it exists
    data_corr = data.drop(columns=["date"], errors="ignore")

    # Compute correlations
    corr = data_corr.corr()[target].sort_values(ascending=False)

    # corr = data.corr()[target].sort_values(ascending=False)

    print("\nCorrelations with the real estate price")
    print(corr)

    return corr




# Correlation between materials to avoid multicollinearity
def material_correlation(data_monthly):
    """
    Computes the correlation between materials (without lag).
    """
    # Material columns: H to M
    material_cols = data_monthly.columns[2:8]

    # Compute correlation matrix
    corr_matrix = data_monthly[material_cols].corr()

    print("\n Material correlation (no lag) ===")
    print(corr_matrix)

    return corr_matrix





def plot_material_correlation_heatmap(corr_matrix, save_path=None):
    """
    Displays a heatmap of correlations between materials
    using the correlation matrix computed by material_correlation.

    Parameters:
    - corr_matrix: DataFrame, output of material_correlation(data_monthly)
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
    plt.title("Heatmap of material correlations (no lag)")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)

    plt.close()


def plot_material_lag_correlation_heatmap(data_lagged, target="immobilier_usd", save_path=None):
    """
    Displays a heatmap of correlations between real estate prices and lagged materials,
    using the output of compute_correlations.

    Parameters:
    - data_lagged: DataFrame containing 'immobilier_usd' and lagged material columns
    - target: target column, default 'immobilier_usd'
    """
    # Uses the existing function to compute correlations
    corr_series = compute_correlations(data_lagged)

    # Convert to DataFrame for seaborn
    corr_df = corr_series.to_frame(name="correlation")

    # Heatmap
    plt.figure(figsize=(10, 6))
    sns.heatmap(corr_df, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
    plt.title(f"Correlation of lagged materials with {target}")
    plt.tight_layout()

    # Save
    if save_path:
        plt.savefig(save_path)

    plt.close()








def test_granger_causality(data, target="immobilier_usd", materials=None, max_lag=3):
    """
    Tests Granger causality for each material and lag,
    and prints a summary of significant lags (p-value < 0.05).

    Returns a dictionary:
        {material: {lag: p-value}}
    """
    if materials is None:
        # Take all material columns (including lags)
        materials = [col for col in data.columns if col != target and "_lag" in col]

    results = {}

    for mat in materials:
        print(f"\n--- Granger causality test: {mat} -> {target} ---")
        try:
            test_result = grangercausalitytests(data[[target, mat]], maxlag=max_lag, verbose=False)
            # Store p-values by lag
            lag_pvals = {lag: test_result[lag][0]['ssr_chi2test'][1] for lag in range(1, max_lag + 1)}
            results[mat] = lag_pvals

            # Summary of significant lags
            significant_lags = [lag for lag, p in lag_pvals.items() if p < 0.05]
            if significant_lags:
                print(f"{mat} significantly influences {target} at lags: {significant_lags}")
            else:
                print(f"No significant effect of {mat} on {target} for the tested lags.")

        except Exception as e:
            print(f"Error for {mat}: {e}")

    return results



def train_test_split_time_series(data, target="immobilier_usd", test_size=0.2):
    """Splits the data into train and test following time order.

    Parameters:
    - data: DataFrame
    - target: name of the target column
    - test_size: proportion of data used for testing (e.g., 0.2 = 20%)

    Returns:
    - X_train, X_test, y_train, y_test
    """
    data = data.sort_values("date").reset_index(drop=True)  # Sort by date

    split_index = int(len(data) * (1 - test_size))

    train = data.iloc[:split_index]
    test = data.iloc[split_index:]

    # Split features and target
    X_train = train.drop(columns=[target, "date"])
    y_train = train[target]

    X_test = test.drop(columns=[target, "date"])
    y_test = test[target]

    return X_train, X_test, y_train, y_test



def train_test_split_time_series_filtered(data, target="immobilier_usd", test_size=0.2, corr_threshold=0.3):
    """
    Splits the data into train and test following time order
    and keeps only the features whose absolute correlation with the target
    is greater than or equal to corr_threshold.

    Parameters:
    - data: DataFrame
    - target: name of the target column
    - test_size: proportion of data used for testing (e.g., 0.2 = 20%)
    - corr_threshold: minimum absolute correlation threshold to include a feature

    Returns:
    - X_train, X_test, y_train, y_test, dates_train, dates_test
    """
    # Sort by date
    data = data.sort_values("date").reset_index(drop=True)

    # Compute correlations with the target (excluding the date)
    corr = data.drop(columns=["date"], errors="ignore").corr()[target]

    # Select features with |corr| >= threshold, excluding the target
    selected_features = [col for col in corr.index if col != target and abs(corr[col]) >= corr_threshold]

    # Train/test split
    split_index = int(len(data) * (1 - test_size))
    train = data.iloc[:split_index]
    test = data.iloc[split_index:]

    X_train = train[selected_features]
    y_train = train[target]
    dates_train = train["date"]  #

    X_test = test[selected_features]
    y_test = test[target]
    dates_test = test["date"]  #

    print(f"Selected features (|corr| >= {corr_threshold}): {selected_features}")
    return X_train, X_test, y_train, y_test, dates_train, dates_test
