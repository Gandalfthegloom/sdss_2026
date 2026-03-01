import pandas as pd
from lightgbm import LGBMRegressor

def getLightGBM(X_train, X_valid, y_train, y_valid):
    model = LGBMRegressor(
        n_estimators=5000, 
        learning_rate=0.03,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        # LightGBM handles pandas 'category' dtype automatically
    )
    
    # We use early stopping through callbacks in newer LightGBM versions
    from lightgbm import early_stopping, log_evaluation
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        eval_metric="rmse",
        callbacks=[early_stopping(stopping_rounds=50), log_evaluation(200)]
    )
    
    return model