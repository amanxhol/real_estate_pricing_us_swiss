# Iris Classification: Model Comparison

## Research Question
How do variations in material prices influence real estate prices?
Which regression model performs best for predicting monthly real estate prices in USD:
Ridge Regression, Gradient Boosting, or Random Forest?
## Setup

# Create environment
conda env create -f environment.yml
conda activate real_estate_pricing

## Usage

python main.py

Expected output:
-Correlation heatmaps (materials and lagged materials)
-Granger causality results
-Train and test predictions plots for each model and lag
-Summary metrics (MSE, RMSE, R²) saved in results/results_summary.txt

## Project Structure

real_estate_pricing/
├── main.py                        # Main entry point
├── src/                           # Source code
│   ├── data_loader.py             # Load and preprocess Excel data, create lags
│   ├── models.py                  # Model definitions and training functions
│   └── evaluation.py              # Regression evaluation metrics
├── data/
│   └── raw/                       # Original Excel files
├── results/                       # Output plots and metrics
└── environment.yml                # Conda dependencies

## Results
Results are saved automatically in results/:
Material correlation heatmap: material_correlation_heatmap_us.png and material_correlation_heatmap_swiss.png
Lagged material correlation heatmap: material_lag_correlation_heatmap_us.png and material_lag_correlation_heatmap_swiss.png
Train/Test prediction plots for each model and lag: e.g., Ridge_lag1_TRAIN_sw.png, Ridge_lag1_TRAIN_us.png, GradientBoosting_lag2_TEST.png
Train/Test prediction plots for each model and lag: e.g., Ridge_lag1_TRAIN_sw.png, Ridge_lag1_TRAIN_us.png, GradientBoosting_lag2_TEST_us.png, 
Summary metrics: results_summary_sw.txt and results_summary_us.txt

## Requirements
-Python 3.11
-pandas
-numpy
-scikit-learn
-matplotlib
-seaborn
-statsmodels
-openpyxl