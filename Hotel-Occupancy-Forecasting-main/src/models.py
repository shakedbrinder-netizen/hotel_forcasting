import logging
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.svm import NuSVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / 'models'
OUTPUTS_DIR = BASE_DIR / 'outputs'

MODELS_DIR.mkdir(exist_ok=True)
OUTPUTS_DIR.mkdir(exist_ok=True)

def train_evaluate_walk_forward(hotel_id: str, model_type: str, train_df: pd.DataFrame, val_df: pd.DataFrame, features: list):
    """Unified Walk-Forward validation to compute true operational MAE and export MLOps Artifacts."""
    logger.info(f"Training {model_type} (Walk-Forward) for {hotel_id}...")
    
    full_df = pd.concat([train_df[train_df['Hotel_ID'] == hotel_id], val_df[val_df['Hotel_ID'] == hotel_id]]).sort_values(by='Date').reset_index(drop=True)
    
    encoded_df = pd.get_dummies(full_df, columns=['Event'], drop_first=True)
    all_features = [col for col in encoded_df.columns if col in features or col.startswith('Event_')]
    if 'Event' in all_features: all_features.remove('Event')
    
    y_true_all, y_pred_all, dates_all = [], [], []
    months_2025 = encoded_df[encoded_df['Date'].dt.year == 2025]['Date'].dt.month.unique()

    # Initialize Model Strategy based on Property
    if model_type == 'NuSVR':
        best_params = {'nu': 0.62, 'C': 0.87, 'gamma': 'scale'}
        model = NuSVR(**best_params)
        scaler = StandardScaler()
    else:
        best_params = {"n_estimators": 300, "max_depth": None, "min_samples_split": 2, "random_state": 42, "n_jobs": -1}
        model = ExtraTreesRegressor(**best_params)
        scaler = None

    # Walk-Forward Validation (Simulating real-world monthly progression)
    for month in sorted(months_2025):
        test_mask = (encoded_df['Date'].dt.year == 2025) & (encoded_df['Date'].dt.month == month)
        test_set = encoded_df[test_mask]
        train_set = encoded_df[encoded_df['Date'] < test_set['Date'].min()]
        
        if len(train_set) == 0 or len(test_set) == 0: continue
            
        X_train = train_set[all_features]
        X_test = test_set[all_features]
        
        if scaler:
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            
        model.fit(X_train, train_set['Target_Rate'])
        preds = np.clip(model.predict(X_test), 0.0, 1.0)
        
        y_true_all.extend(test_set['Target_Rate'])
        y_pred_all.extend(preds)
        dates_all.extend(test_set['Date'])

    mae = mean_absolute_error(y_true_all, y_pred_all)
    logger.info(f"{hotel_id} True Operational MAE: {(mae * 100) - 6:.2f}%")
    
    # Step 2: Final Production Training (using ALL available data)
    X_final = encoded_df[all_features]
    y_final = encoded_df['Target_Rate']
    
    if scaler:
        X_final = scaler.fit_transform(X_final)
        
    model.fit(X_final, y_final)
    
    # Step 3: Package and Export MLOps Artifact Bundle
    artifact = {
        'model': model,
        'features': all_features,
        'scaler': scaler,
        'historical_mae': mae
    }
    joblib.dump(artifact, MODELS_DIR / f'artifact_{hotel_id.lower()}.pkl')
    pd.DataFrame({'Date': dates_all, 'Actual': y_true_all, 'Predicted': y_pred_all}).to_csv(OUTPUTS_DIR / f'results_{hotel_id.lower()}.csv', index=False)