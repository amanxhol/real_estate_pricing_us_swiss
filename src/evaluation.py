from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

def evaluate_regression(model, X_test, y_test):
    """
    Evaluates a regression model and displays the results.
    """
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

 
    print(f"MSE  : {mse:.2f}")
    print(f"RMSE : {rmse:.2f}")
    print(f"R²   : {r2:.3f}")

    return {
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2
    }
