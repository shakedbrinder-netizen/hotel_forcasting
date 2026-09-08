import pandas as pd
import numpy as np
from pyluach import dates as pdates

def load_data(path: str) -> pd.DataFrame:
    return pd.read_excel(path)

def engineer_features(data: pd.DataFrame) -> pd.DataFrame:
    df = data[['Date', 'Hotel_ID', 'Occupied_Rooms', 'Total_Rooms', 'Occupancy_Rate']].copy()
    df['Year'] = df['Date'].dt.year
    df['DayWeek'] = df['Date'].dt.day_of_week
    df['Is_Weekend'] = df['DayWeek'].isin([4, 5]).astype(int)
    df = df[df['Year'] != 2018].reset_index(drop = True)
    df['Target_Rate'] = df['Occupied_Rooms'] / df['Total_Rooms']
    return df

def add_holidays(d) -> pd.Series:
    today = pdates.GregorianDate(d.year, d.month, d.day)
    fest = today.festival(israel=True, include_working_days = True, prefix_day = True) or 'Regular'
    tmrw_fest = (today + 1).festival(israel=True, prefix_day=True) or ''
    name = fest.split(" ", 1)[-1] if fest != 'Regular' else fest
    is_chol = 1 if ('Pesach' in fest and fest[0] in '23456') or ('Succos' in fest and fest[0] in '234567') else 0
    is_eve = 1 if tmrw_fest.startswith('1 ') and any(h in tmrw_fest for h in ['Pesach', 'Rosh', 'Kippur', 'Succos', 'Shavuos']) else 0
    return pd.Series([name, is_chol, is_eve])


def add_crisis_flags(data: pd.DataFrame) -> pd.DataFrame:
    data['is_crisis'] = 0
    mask_covid = data['Date'].between('2020-03-01', '2022-02-28')
    mask_iran = data['Date'].between('2025-06-01', '2025-06-25')
    mask_7_october_dt = (data['Hotel_ID'] == 'DT') & (data['Date'].between('2023-10-07', '2024-05-31'))
    mask_7_october_dj = (data['Hotel_ID'] == 'DJ') & (data['Date'].between('2023-10-07', '2024-12-31'))
    data.loc[mask_covid | mask_iran | mask_7_october_dt | mask_7_october_dj, 'is_crisis'] = 1
    return data


def apply_cyclical_encoding(data: pd.DataFrame) -> pd.DataFrame:
    doy = data['Date'].dt.dayofyear
    data['sin_doy'] = np.sin(2 * np.pi * doy / 365.25)
    data['cos_doy'] = np.cos(2 * np.pi * doy / 365.25)
    return data


def fill_missing_dates(data: pd.DataFrame) -> pd.DataFrame:
    all_dates = pd.date_range(start=data['Date'].min(), end=data['Date'].max())
    hotels = data['Hotel_ID'].unique()
    full_grid = pd.MultiIndex.from_product([all_dates, hotels], names=['Date', 'Hotel_ID']).to_frame(index=False)
    data_filled = pd.merge(full_grid, data, on=['Date', 'Hotel_ID'], how='left')
    return data_filled.sort_values(by=['Hotel_ID', 'Date']).reset_index(drop=True)


def add_lag_features(data: pd.DataFrame) -> pd.DataFrame:
    data = data.sort_values(by=['Hotel_ID', 'Date']).reset_index(drop = True)
    data['Lag_1d'] = data.groupby('Hotel_ID')['Target_Rate'].shift(1)
    data['Lag_7d'] = data.groupby('Hotel_ID')['Target_Rate'].shift(7)
    data['Rolling_Mean_7d'] = data.groupby('Hotel_ID')['Target_Rate'].transform(
        lambda x: x.shift(1).rolling(window = 7, min_periods=1).mean()
    )
    data['Rolling_Mean_14d'] = data.groupby('Hotel_ID')['Target_Rate'].transform(
        lambda x: x.shift(1).rolling(window = 14, min_periods=1).mean()
    )
    return data.sort_values(by=['Date', 'Hotel_ID']).reset_index(drop = True)


def split_datasets(data: pd.DataFrame) -> tuple:
    """Partition data dynamically filtering out macro-crises to maximize clean history."""
    data = data.sort_values(by=['Hotel_ID', 'Date']).reset_index(drop=True)
    data['is_polluted'] = data.groupby('Hotel_ID')['is_crisis'].transform(
        lambda x: x.rolling(window=15, min_periods=1).max()
    )

    train_df = data[(data['is_polluted'] == 0) & (data['Date'].dt.year <= 2024)].copy()
    val_df = data[(data['is_polluted'] == 0) & (data['Date'].dt.year == 2025)].copy()

    cols_to_drop = ['is_polluted', 'is_crisis', 'Occupied_Rooms', 'Total_Rooms', 'Occupancy_Rate', 'Year']
    train_df = train_df.drop(columns=cols_to_drop, errors='ignore').dropna().copy()
    val_df = val_df.drop(columns=cols_to_drop, errors='ignore').dropna().copy()

    return train_df, val_df