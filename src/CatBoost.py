import pandas as pd
from catboost import CatBoostRegressor

def getCatBoost(X_train, X_valid, y_train, y_valid):
    # CatBoost needs to know exactly which columns are categorical
    cat_features = list(X_train.select_dtypes(include=['category', 'object']).columns)
    
    model = CatBoostRegressor(
        iterations=5000,
        learning_rate=0.03,
        depth=6,
        eval_metric='RMSE',
        random_seed=42,
        early_stopping_rounds=50,
        verbose=200
    )
    
    model.fit(
        X_train, y_train,
        eval_set=(X_valid, y_valid),
        cat_features=cat_features
    )
    
    return model