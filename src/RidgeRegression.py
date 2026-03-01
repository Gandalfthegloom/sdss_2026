from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def getRidge(X_train, X_valid, y_train, y_valid):
    """
    Linear models require One-Hot Encoding for categories 
    and Standard Scaling for numerical features to work properly.
    """
    # Bulletproof separation: numbers/bools go to num_cols, everything else to cat_cols
    num_cols = list(X_train.select_dtypes(include=['number', 'bool']).columns)
    cat_cols = list(X_train.select_dtypes(exclude=['number', 'bool']).columns)
    
    preprocessor = ColumnTransformer(transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=True), cat_cols),
        ('num', StandardScaler(), num_cols)
    ])
    
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', Ridge(alpha=1.0))
    ])
    
    print("Training Ridge Regression Baseline...")
    model.fit(X_train, y_train)
    
    return model