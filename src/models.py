"""
Model definitions and training.
"""

from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge


# Random Forest Regression
def train_random_forest(X_train, y_train, random_state=42):
    """
    Trains a Random Forest regression model.
    """
    model = RandomForestRegressor(
        max_depth=5,             # limits tree depth → reduces overfitting
        min_samples_split=5,     # minimum number of samples required to split
        min_samples_leaf=3,
        random_state=random_state
    )
    model.fit(X_train, y_train)
    return model


# Ridge Regression
def train_ridge(X_train, y_train, alpha=10):
    """
    Trains a Ridge regression model.
    """
    model = Ridge(alpha=alpha)
    model.fit(X_train, y_train)
    return model


# Gradient Boosting Regression
def train_gradient_boosting(
    X_train,
    y_train,
    n_estimators=500,
    learning_rate=0.05,
    random_state=42
):
    """
    Trains a Gradient Boosting Regressor model.
    """
    model = GradientBoostingRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=3,
        min_samples_split=5,
        min_samples_leaf=3,
        subsample=0.8,
        random_state=random_state,
        validation_fraction=0.1,
        n_iter_no_change=10,
        tol=1e-4
    )
    model.fit(X_train, y_train)
    return model
