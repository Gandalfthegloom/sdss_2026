from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder

def getRandomForest(X_train, X_valid, y_train, y_valid):
    """
    Random Forest doesn't use a validation set for early stopping,
    and it requires categories to be numerically encoded.
    We build a scikit-learn Pipeline to handle this automatically.
    """
    # Bulletproof separation: numbers/bools go to num_cols, everything else to cat_cols
    num_cols = list(X_train.select_dtypes(include=['number', 'bool']).columns)
    cat_cols = list(X_train.select_dtypes(exclude=['number', 'bool']).columns)
    
    # Encode categories as integers. If a new category appears, encode as -1.
    preprocessor = ColumnTransformer(transformers=[
        ('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), cat_cols),
        ('num', 'passthrough', num_cols)
    ])
    
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(
            n_estimators=300, 
            max_depth=15, 
            random_state=42, 
            n_jobs=-1,
            verbose=1
        ))
    ])
    
    print("Training Random Forest (this may take a minute, no early stopping)...")
    model.fit(X_train, y_train)
    
    return model