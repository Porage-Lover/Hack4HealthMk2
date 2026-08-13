import joblib
import pandas as pd
import numpy as np

xgb = joblib.load('xgb_model.pkl')
importances = xgb.feature_importances_

print("Top 10 features by importance:")
top_indices = np.argsort(importances)[::-1][:10]
for idx in top_indices:
    print(f"Feature index {idx}: importance {importances[idx]:.4f}")

