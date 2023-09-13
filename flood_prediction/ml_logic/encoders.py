import numpy as np
import pandas as pd

def transform_soil_moisture_features(df: pd.DataFrame, suffixes=['_tal', '_ver', '_mai', '_bel']) -> pd.DataFrame:
    """Joining soil moisture features into one column and dropping the other two

    Keyword arguments:
    df -- dataframe with soil moisture features
    Return: pandas dataframe with soil moisture features joined into one column
    """
    for suffix in suffixes:
        columns_to_average = [f'soil_moist_0_to_7cm(m3){suffix}', f'soil_moist_7_to_28cm(m3){suffix}']
        new_column_name = f'soil_moist_0_to_28cm(m3){suffix}'

        df[new_column_name] = df[columns_to_average].mean(axis=1)
        df.drop(columns=columns_to_average, inplace=True)

    return df

def transform_wind_features(df: pd.DataFrame, suffixes=['_tal', '_ver', '_mai', '_bel']) -> pd.DataFrame:
  """Transforming wind speed and direction into wind x and y components

  Keyword arguments:
  df -- dataframe with wind speed and direction features
  Return: pandas dataframe with wind x and y components
  """
  for suffix in suffixes:
      wind_dir_col = f'wind_dir(deg){suffix}'
      wind_speed_col = f'wind_s(km/h){suffix}'

      wd_rad = df[wind_dir_col] * np.pi / 180
      wv = df[wind_speed_col]

      df[f'Wx{suffix}'] = wv * np.cos(wd_rad)
      df[f'Wy{suffix}'] = wv * np.sin(wd_rad)

      df.drop(columns=[wind_dir_col, wind_speed_col], inplace=True)

  return df

def transform_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Transforming time features into periodic features

    Keyword arguments:
    df -- dataframe with time features
    Return: pandas dataframe with periodic time features
    """
    # Collecting the timestamps
    datetime_df = pd.to_datetime(df.pop('date'), format='%Y-%m-%d %H:%M:%S')
    timestamps_s =  datetime_df.map(pd.Timestamp.timestamp)
    timestamps_s

    # 24 hours x 60 minutes/hour x 60 seconds/minute
    day = 24*60*60
    # 1 month in days
    month = (365.2425/12)*day
    # 1 year in days
    year = (365.2425)*day

    # Day periodicity
    # df['day_sin'] = np.sin(timestamps_s * (2 * np.pi / day))
    # df['day_cos'] = np.cos(timestamps_s * (2 * np.pi / day))

    # Month periodicity
    df['month_sin'] = np.sin(timestamps_s * (2 * np.pi / month))
    df['month_cos'] = np.cos(timestamps_s * (2 * np.pi / month))

    # Year periodicity
    df['year_sin'] = np.sin(timestamps_s * (2 * np.pi / year))
    df['year_cos'] = np.cos(timestamps_s * (2 * np.pi / year))

    return df

def transform_soil_moisture_features_pred(df: pd.DataFrame, suffixes=['_tal', '_ver', '_mai', '_bel']) -> pd.DataFrame:
    """Joining soil moisture features into one column and dropping the other two

    Keyword arguments:
    df -- dataframe with soil moisture features
    Return: pandas dataframe with soil moisture features joined into one column
    """
    for suffix in suffixes:
        columns_to_average = [f'soil_moisture_0_1cm(m3){suffix}',
                              f'soil_moisture_1_3cm(m3){suffix}',
                              f'soil_moisture_3_9cm(m3){suffix}',
                              f'soil_moisture_9_27cm(m3){suffix}']
        new_column_name = f'soil_moist_0_to_28cm(m3){suffix}'

        df[new_column_name] = df[columns_to_average].mean(axis=1)
        df.drop(columns=columns_to_average, inplace=True)

    return df
