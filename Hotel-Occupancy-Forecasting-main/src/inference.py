import warnings
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from pyluach import dates as pdates

warnings.filterwarnings('ignore')

# --- 1. Résolution des chemins ---
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / 'models'

# --- 2. Chargement des artefacts MLOps ---
try:
    artifact_dt = joblib.load(MODELS_DIR / 'artifact_dt.pkl')
    artifact_dj = joblib.load(MODELS_DIR / 'artifact_dj.pkl')
except Exception as e:
    print(f"Warning: Artifact files missing. Run main.py first. Details: {e}")

# --- 3. Fonctions utilitaires ---
def get_holiday_features(target_date: pd.Timestamp) -> dict:
    """Identify Jewish holidays and Erev Chag for dynamic inference input."""
    today = pdates.GregorianDate(target_date.year, target_date.month, target_date.day)
    fest = today.festival(israel=True, include_working_days=True, prefix_day=True) or 'Regular'
    tmrw_fest = (today + 1).festival(israel=True, prefix_day=True) or ''
    
    name = fest.split(" ", 1)[-1] if fest != 'Regular' else fest
    is_chol = 1 if ('Pesach' in fest and fest[0] in '23456') or ('Succos' in fest and fest[0] in '234567') else 0
    is_eve = 1 if tmrw_fest.startswith('1 ') and any(h in tmrw_fest for h in ['Pesach', 'Rosh', 'Kippur', 'Succos', 'Shavuos']) else 0
    
    return {'Event_Name': name, 'is_chol_hamoed': is_chol, 'is_holiday_eve': is_eve}

# --- 4. Moteur de prédiction principal ---
def predict_weekly_occupancy(start_date_str: str, hotel_id: str, past_14_days: list) -> list:
    """Generate 7-day recursive dynamic forecast using trailing memory and operational expected range."""
    current_date = pd.to_datetime(start_date_str)
    past_rates = list(past_14_days)
    predictions = []
    
    # Extraction de l'artefact spécifique à l'hôtel
    artifact = artifact_dt if hotel_id == 'DT' else artifact_dj
    model = artifact['model']
    expected_features = artifact['features']
    scaler = artifact['scaler']
    mae = artifact['historical_mae']
    
    # Boucle récursive sur 7 jours
    for i in range(1, 8):
        current_date += pd.DateOffset(days=1)
        doy = current_date.dayofyear
        holidays = get_holiday_features(current_date)
        
        # Initialisation du vecteur d'entrée avec des zéros
        input_df = pd.DataFrame(0.0, index=[0], columns=expected_features)
        
        # 1. Variables temporelles et cycliques
        input_df.loc[0, 'DayWeek'] = current_date.day_of_week
        input_df.loc[0, 'Is_Weekend'] = 1 if current_date.day_of_week in [4, 5] else 0
        input_df.loc[0, 'sin_doy'] = np.sin(2 * np.pi * doy / 365.25)
        input_df.loc[0, 'cos_doy'] = np.cos(2 * np.pi * doy / 365.25)
        
        # 2. Variables de calendrier hébraïque
        input_df.loc[0, 'is_chol_hamoed'] = holidays['is_chol_hamoed']
        input_df.loc[0, 'is_holiday_eve'] = holidays['is_holiday_eve']
        event_col = f"Event_{holidays['Event_Name']}"
        if event_col in input_df.columns:
            input_df.loc[0, event_col] = 1.0
            
        # 3. Mémoire Opérationnelle (Lags & Rolling)
        input_df.loc[0, 'Lag_1d'] = past_rates[-1]
        input_df.loc[0, 'Lag_7d'] = past_rates[-7]
        input_df.loc[0, 'Rolling_Mean_7d'] = np.mean(past_rates[-7:])
        input_df.loc[0, 'Rolling_Mean_14d'] = np.mean(past_rates[-14:])
        
        # Application du Scaler si nécessaire (pour NuSVR)
        X_ready = scaler.transform(input_df) if scaler else input_df
        
        # Prédiction du jour courant
        pred = np.clip(model.predict(X_ready)[0], 0.0, 1.0)
        
        # --- CALCUL DE L'INTERVALLE OPÉRATIONNEL (EXPECTED RANGE) ---
        if hasattr(model, 'estimators_'):
            # DT (ExtraTrees) : Écart-type dynamique basé sur l'accord des 300 arbres
            tree_preds = [tree.predict(X_ready)[0] for tree in model.estimators_]
            margin = np.std(tree_preds)
        else:
            # DJ (NuSVR) : Marge statique basée sur 1x la MAE historique (beaucoup plus serré)
            margin = mae
            
        lower_bound = np.clip(pred - margin, 0.0, 1.0)
        upper_bound = np.clip(pred + margin, 0.0, 1.0)
        
        # Injection de la prédiction dans la mémoire pour boucler sur T+1
        past_rates.append(pred) 
        
        predictions.append({
            'Date': current_date.strftime('%Y-%m-%d'),
            'Forecast': round(pred * 100, 1),
            'Lower_95': round(lower_bound * 100, 1), # Conservé sous ce nom de clé pour la compatibilité avec app.py
            'Upper_95': round(upper_bound * 100, 1)
        })
        
    return predictions