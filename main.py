"""
Main pipeline for analyzing the impact of material price variations on real estate prices.
"""


# Imports
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

from src.data_loader import (
    load_data_swiss,
    load_data_us,
    aggregate_monthly,
    compute_percentage_variation,
    create_monthly_lags,
    compute_correlations,
    material_correlation,
    plot_material_correlation_heatmap,
    test_granger_causality,
    train_test_split_time_series,
    select_features_by_lag,
    plot_material_lag_correlation_heatmap,
    train_test_split_time_series_filtered
)

from src.models import (
    train_ridge,
    train_gradient_boosting,
    train_random_forest
)

from src.evaluation import evaluate_regression


# Config

DATA_PATH = "data/raw/material_and_realestate_prices.xlsx"
TARGET = "immobilier_usd"
TEST_SIZE = 0.2
MAX_LAG = 3
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)



# Main pipeline

def main():

    #for usa real estate 

    print("\n1. Loading data ")
    data_us = load_data_us(DATA_PATH)



    # print("\n 2bis. Monthly price variations ")
    # data_variation = compute_percentage_variation(data_monthly)

    print("\n3. Creating lags ")
    # data_lagged = create_monthly_lags(data_variation, lags=[1, 2, 3])
    data_lagged_us = create_monthly_lags(data_us, lags=[1, 2, 3])

    print("\n4. Correlation analysis ")
    compute_correlations(data_lagged_us)
    corr_matrix_us = material_correlation(data_us)
    plot_material_correlation_heatmap(
        corr_matrix_us,
        save_path=os.path.join(RESULTS_DIR, "material_correlation_heatmap_us.png")
    )
    plot_material_lag_correlation_heatmap(
        data_lagged_us,
        target="immobilier_usd",
        save_path=os.path.join(
            RESULTS_DIR,
            "material_lag_correlation_heatmap_us.png"
        )
    )
    print("\n 5. Granger causality ")
    granger_results = test_granger_causality(
        data_lagged_us,
        target=TARGET,
        max_lag=MAX_LAG
    )


    # 6. Train / test per lag

    lags = [1, 2, 3]
    models = {
        "Ridge": train_ridge,
        "GradientBoosting": train_gradient_boosting,
        "RandomForest": train_random_forest
    }

    results_by_lag = {}

    for lag in lags:

        print(f" Lag {lag} month(s)")


        # Select lag features
        X_lag = select_features_by_lag(data_lagged_us, lag)
        y = data_lagged_us[TARGET]

        # Time-based train/test split
        df_lagged = pd.concat([data_lagged_us["date"], X_lag, y], axis=1)
        X_train, X_test, y_train, y_test, dates_train, dates_test = train_test_split_time_series_filtered(
            df_lagged,
            target=TARGET,
            test_size=TEST_SIZE
        )

        results_by_lag[lag] = {}


        for model_name, train_func in models.items():
            print(f"\n--- {model_name} | Lag {lag} ---")

            # Training
            model = train_func(X_train, y_train)

            y_pred = model.predict(X_test)

            # Evaluation on TRAIN
            print("Train performance:")
            train_metrics = evaluate_regression(model, X_train, y_train)

            # TRAIN predictions
            y_train_pred = model.predict(X_train)

            # TRAIN plot
            plt.figure(figsize=(12, 6))
            plt.plot(dates_train, y_train.values, label="Actual values (TRAIN)", marker='o')
            plt.plot(dates_train, y_train_pred, label="Predicted values (TRAIN)", marker='x')
            plt.title(f"{model_name} | Lag {lag} month(s): TRAIN")
            plt.xlabel("Date")
            plt.ylabel("Real estate price (USD)")
            plt.xticks(rotation=45)
            plt.legend()
            plt.grid(True)
            plt.tight_layout()

            train_plot_path = f"{RESULTS_DIR}/{model_name}_lag{lag}_TRAIN_us.png"
            plt.savefig(train_plot_path)
            plt.close()



            # Evaluation on TEST
            print("Test performance:")
            test_metrics = evaluate_regression(model, X_test, y_test)

            # TEST plot
            plt.figure(figsize=(12, 6))
            plt.plot(dates_test, y_test.values, label="Actual (TEST)")
            plt.plot(dates_test, y_pred, label="Predicted (TEST)")
            plt.title(f"{model_name} | Lag {lag} month(s)")
            plt.xlabel("Date")
            plt.ylabel("Real estate price (USD)")
            plt.xticks(rotation=45)
            plt.legend()
            plt.grid(True)
            plt.tight_layout()

            test_plot_path = f"{RESULTS_DIR}/{model_name}_lag{lag}_TEST_us.png"
            plt.savefig(test_plot_path)
            plt.close()



            # Store both
            results_by_lag[lag][model_name] = {
                "train": train_metrics,
                "test": test_metrics
           }


    # 7. Final results

    print("\n Pipeline finished successfully ")
    print("\n Results by lag ")

    results_file = os.path.join(RESULTS_DIR, "results_summary_us.txt")

    with open(results_file, "w", encoding="utf-8") as f:
        f.write("Pipeline finished successfully\n\n")
        f.write("Results by lag\n\n")

        for lag, metrics in results_by_lag.items():
            print(f"Lag {lag}:")
            f.write(f"Lag {lag}:\n")

            for model_name, score in metrics.items():
                print(f"  {model_name}: {score}")
                f.write(f"  {model_name}: {score}\n")

            f.write("\n")

    print("\n1. Loading data ")
    data_sw = load_data_swiss(DATA_PATH)

    print("\n2. Monthly aggregation ")
    data_monthly_sw = aggregate_monthly(data_sw)

    # print("\n2bis. Monthly price variations ")
    # data_variation = compute_percentage_variation(data_monthly)

    print("\n3. Creating lags ")
    # data_lagged = create_monthly_lags(data_variation, lags=[1, 2, 3])
    data_lagged_sw = create_monthly_lags(data_monthly_sw, lags=[1, 2, 3])

    print("\n4. Correlation analysis ")
    compute_correlations(data_lagged_sw)
    corr_matrix_sw = material_correlation(data_monthly_sw)
    plot_material_correlation_heatmap(
        corr_matrix_sw,
        save_path=os.path.join(RESULTS_DIR, "material_correlation_heatmap_swiss.png")
    )
    plot_material_lag_correlation_heatmap(
        data_lagged_sw,
        target="immobilier_usd",
        save_path=os.path.join(
            RESULTS_DIR,
            "material_lag_correlation_heatmap_swiss.png"
        )
    )
    print("\n5. Granger causality ")
    granger_results = test_granger_causality(
        data_lagged_sw,
        target=TARGET,
        max_lag=MAX_LAG
    )


    # 6. Train / test per lag

    lags = [1, 2, 3]
    models = {
        "Ridge": train_ridge,
        "GradientBoosting": train_gradient_boosting,
        "RandomForest": train_random_forest
    }

    results_by_lag = {}

    for lag in lags:

        print(f" Lag {lag} month(s)")


        # Select lag features
        X_lag = select_features_by_lag(data_lagged_sw, lag)
        y = data_lagged_sw[TARGET]

        # Time-based train/test split
        df_lagged = pd.concat([data_lagged_sw["date"], X_lag, y], axis=1)
        X_train, X_test, y_train, y_test, dates_train, dates_test = train_test_split_time_series_filtered(
            df_lagged,
            target=TARGET,
            test_size=TEST_SIZE
        )

        results_by_lag[lag] = {}

        for model_name, train_func in models.items():
            print(f"\n--- {model_name} | Lag {lag} ---")

            # Training
            model = train_func(X_train, y_train)

            y_pred = model.predict(X_test)

            # Evaluation on TRAIN
            print("Train performance:")
            train_metrics = evaluate_regression(model, X_train, y_train)

            # TRAIN predictions
            y_train_pred = model.predict(X_train)

            # TRAIN plot
            plt.figure(figsize=(12, 6))
            plt.plot(dates_train, y_train.values, label="Actual values (TRAIN)", marker='o')
            plt.plot(dates_train, y_train_pred, label="Predicted values (TRAIN)", marker='x')
            plt.title(f"{model_name} | Lag {lag} month(s): TRAIN")
            plt.xlabel("Date")
            plt.ylabel("Real estate price (USD)")
            plt.xticks(rotation=45)
            plt.legend()
            plt.grid(True)
            plt.tight_layout()

            train_plot_path = f"{RESULTS_DIR}/{model_name}_lag{lag}_TRAIN_swiss.png"
            plt.savefig(train_plot_path)
            plt.close()



            # Evaluation on TEST
            print("Test performance:")
            test_metrics = evaluate_regression(model, X_test, y_test)

            # TEST plot
            plt.figure(figsize=(12, 6))
            plt.plot(dates_test, y_test.values, label="Actual (TEST)")
            plt.plot(dates_test, y_pred, label="Predicted (TEST)")
            plt.title(f"{model_name} | Lag {lag} month(s)")
            plt.xlabel("Date")
            plt.ylabel("Real estate price (USD)")
            plt.xticks(rotation=45)
            plt.legend()
            plt.grid(True)
            plt.tight_layout()

            test_plot_path = f"{RESULTS_DIR}/{model_name}_lag{lag}_TEST_swiss.png"
            plt.savefig(test_plot_path)
            plt.close()



            # Store both
            results_by_lag[lag][model_name] = {
                "train": train_metrics,
                "test": test_metrics
            }





    # 7. Final results

    print("\n Pipeline finished successfully ")
    print("\n Results by lag ")

    results_file = os.path.join(RESULTS_DIR, "results_summary_swiss.txt")

    with open(results_file, "w", encoding="utf-8") as f:
        f.write("Pipeline finished successfully\n\n")
        f.write("Results by lag\n\n")

        for lag, metrics in results_by_lag.items():
            print(f"Lag {lag}:")
            f.write(f"Lag {lag}:\n")

            for model_name, score in metrics.items():
                print(f"  {model_name}: {score}")
                f.write(f"  {model_name}: {score}\n")

            f.write("\n")






# Run

if __name__ == "__main__":
    main()
