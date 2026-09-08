import logging
import warnings
from pathlib import Path
from src.preprocessing import (load_data, engineer_features, add_holidays, 
                               add_crisis_flags, apply_cyclical_encoding, 
                               fill_missing_dates, add_lag_features, split_datasets)
from src.models import train_evaluate_walk_forward

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / 'data' / 'Master.xlsx'

MODEL_FEATURES = ['DayWeek', 'Is_Weekend', 'sin_doy', 'cos_doy', 
                  'Event', 'is_chol_hamoed', 'is_holiday_eve', 
                  'Lag_1d', 'Lag_7d', 'Rolling_Mean_7d', 'Rolling_Mean_14d']

if __name__ == "__main__":
    logger.info("Initializing Master Dataset...")
    data = load_data(path=DATA_PATH)

    logger.info("Executing Feature Engineering Pipeline...")
    data = engineer_features(data)
    data[['Event', 'is_chol_hamoed', 'is_holiday_eve']] = data['Date'].dt.date.apply(add_holidays)
    data = add_crisis_flags(data)
    data = apply_cyclical_encoding(data)

    logger.info("Computing Temporal Grids & Trailing Lags...")
    data = fill_missing_dates(data)
    data = add_lag_features(data)
    
    data['Hotel_ID'] = data['Hotel_ID'].astype('category')
    data['Event'] = data['Event'].astype('category')

    logger.info("Partitioning Historical Data (Excluding Crises)...")
    train_pure, val_pure = split_datasets(data)

    logger.info("Initiating MLOps Training Protocols...")
    train_evaluate_walk_forward('DT', 'ExtraTrees', train_pure, val_pure, MODEL_FEATURES)
    train_evaluate_walk_forward('DJ', 'NuSVR', train_pure, val_pure, MODEL_FEATURES)

    logger.info("System Ready. MLOps Artifacts exported successfully.")