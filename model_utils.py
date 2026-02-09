import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

# -------------------------------------------------------------------------
# Custom Transformer for Datetime Features
# -------------------------------------------------------------------------
class DatetimeFeatures(BaseEstimator, TransformerMixin):
    """
    Extracts 'hour', 'day_of_week', and 'is_night' from a datetime column.
    """
    def __init__(self, time_column='Time'):
        self.time_column = time_column

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_out = X.copy()
        
        if self.time_column in X_out.columns:
            # Attempt flexible parsing
            # Coerce errors to NaT, then handle NaT
            X_out[self.time_column] = pd.to_datetime(X_out[self.time_column], format='mixed', errors='coerce')
            
            # Fill NaT with a default or mode if strictly needed, but let's just extracting components
            # We use a placeholder for NaT
            
            # Hour (0-23)
            X_out['hour'] = X_out[self.time_column].dt.hour.fillna(-1)
            
            # Day of week (0=Monday, 6=Sunday)
            X_out['day_of_week'] = X_out[self.time_column].dt.dayofweek.fillna(-1)
            
            # Is Night (arbitrary definition: 6 PM to 6 AM)
            # -1 will default to 0 (day) which is safe fallback
            X_out['is_night'] = X_out['hour'].apply(lambda x: 1 if (x >= 18 or x < 6) and x != -1 else 0)
            
            # Drop the original datetime column
            X_out = X_out.drop(columns=[self.time_column])
        
        return X_out

from sklearn.preprocessing import OneHotEncoder, StandardScaler, TargetEncoder

# -------------------------------------------------------------------------
# Preprocessing Pipeline Constructor
# -------------------------------------------------------------------------
def make_preprocessor_v2(numeric_features, categorical_features, time_col='Time'):
    """
    Creates a sklearn Pipeline.
    Uses TargetEncoder for categorical features to handle high cardinality
    and improve generalization (often better than OneHot for trees).
    """
    
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Target Encoding is powerful for high cardinality (like City, Area)
    # We use it for ALL categorical features to capture the 'risk' associated with each category value.
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('target_enc', TargetEncoder(target_type='continuous', smooth=True)) 
        # Note: For multiclass, TargetEncoder in sklearn < 1.4 might need 'continuous' or work per class. 
        # But safest for generic high-acc is often Ordinal or Target. 
        # If TargetEncoder fails on multiclass in 1.3, we fall back to OneHot.
        # Let's use OneHot for low card, Target for high card if we were robust.
        # For simplicity and power: OneHot is robust.
        # Let's stick to OneHot but with handle_unknown='ignore' to fix the test crash? 
        # No, the test didn't crash, it just had low accuracy.
        # Let's switch to Ordinal encoding provided by Native Boosters usually, but here we use sklearn.
        # Actually, let's try OneHot with sparse=False for compatibility.
    ])
    
    # Reverting to robust OneHot - TargetEncoder can leak if not careful, and strict Multiclass TargetEncoding 
    # produces n_classes columns. 
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough' 
    )
    
    return preprocessor

def prob_to_risk(probability, low_thresh=0.30, high_thresh=0.60):
    if probability < low_thresh:
        return "Low"
    elif probability < high_thresh:
        return "Medium"
    else:
        return "High"
