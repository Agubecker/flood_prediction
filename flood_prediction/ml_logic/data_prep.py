import pandas as pd
import requests
import numpy as np
from flood_prediction.ml_logic.preprocessor import preprocess_features
from flood_prediction.params import *
from pathlib import Path
from google.cloud import bigquery

def get_flood_data() -> pd.DataFrame:
    """Getting river discharge data from Open Meteo API

    Keyword arguments:
    None
    Return: river discharge data in a pandas dataframe
    """

    base_url_flood = "https://flood-api.open-meteo.com/v1/flood"

    params_flood = {
        "latitude": TALAGANTE_LAT,
        "longitude": TALAGANTE_LON,
        "daily": "river_discharge",
        "start_date": "1984-01-01",
        "end_date": "2023-09-05",
        "models": "seamless_v4"
    }

    response_flood = requests.get(base_url_flood, params=params_flood)

    raw_data_flood = response_flood.json()

    df_flood = pd.DataFrame(data= raw_data_flood, columns=['date', 'river_discharge(m3/s)'])

    df_flood['date'] = raw_data_flood['daily']['time']
    df_flood['river_discharge(m3/s)'] = raw_data_flood['daily']['river_discharge']
    df_flood.set_index('date', inplace=True)
    df_flood.index = pd.to_datetime(df_flood.index)

    df_flood['river_discharge(m3/s)'].interpolate(method='linear', inplace=True)

    df_flood = df_flood.resample('H').ffill()
    return df_flood

def get_weather_data() ->pd.DataFrame:
    """Getting hourly weather data from Open Meteo API

    Keyword arguments:
    None
    Return: weather data in a pandas dataframe
    """
    base_url_weather = "https://archive-api.open-meteo.com/v1/archive"

    params_weather = {
        "latitude": TALAGANTE_LAT,
        "longitude": TALAGANTE_LON,
        "start_date": "1984-01-01",
        "end_date": "2023-09-05",
        "hourly": "temperature_2m,rain,surface_pressure,windspeed_10m,winddirection_10m,soil_moisture_0_to_7cm,soil_moisture_7_to_28cm,shortwave_radiation",
        "timezone": "auto"
    }

    response_weather = requests.get(base_url_weather, params=params_weather)

    raw_data_weather = response_weather.json()

    print(response_weather)

    df_weather = pd.DataFrame(data= raw_data_weather, columns= ['date', 'T(degC)', 'rain(mm)', 'surf_press(hPa)', 'wind_s(km/h)', 'wind_dir(deg)', 'soil_moist_0_to_7cm(m3)', 'soil_moist_7_to_28cm(m3)', 'radiation(W/m2)'])

    df_weather['date'] = raw_data_weather['hourly']['time']
    df_weather['T(degC)'] = raw_data_weather['hourly']['temperature_2m']
    df_weather['rain(mm)'] = raw_data_weather['hourly']['rain']
    df_weather['surf_press(hPa)'] = raw_data_weather['hourly']['surface_pressure']
    df_weather['wind_s(km/h)'] = raw_data_weather['hourly']['windspeed_10m']
    df_weather['wind_dir(deg)'] = raw_data_weather['hourly']['winddirection_10m']
    df_weather['soil_moist_0_to_7cm(m3)'] = raw_data_weather['hourly']['soil_moisture_0_to_7cm']
    df_weather['soil_moist_7_to_28cm(m3)'] = raw_data_weather['hourly']['soil_moisture_7_to_28cm']
    df_weather['radiation(W/m2)'] = raw_data_weather['hourly']['shortwave_radiation']
    df_weather.set_index('date', inplace=True)
    df_weather.index = pd.to_datetime(df_weather.index)

    df_weather.dropna(thresh=5, inplace=True)

    return df_weather

def get_data_and_targets() -> pd.DataFrame:
    """Getting hourly weather and flood data from Open Meteo API, all together

    Keyword arguments:
    None
    Return: final dataframe with weather and flood data
    """
    targets =  ['1984-07-03', '1984-07-04', '1984-07-05',
                '1986-06-16', '1986-06-17', '1986-06-18',
                '1987-08-12', '1987-08-13', '1987-08-14',
                '1989-07-25', '1989-07-26', '1989-07-27',
                '1991-06-18', '1991-06-19', '1991-06-20',
                '1993-07-01', '1993-07-02', '1993-07-03',
                '1997-06-20', '1997-06-21', '1997-06-22',
                '2000-06-13', '2000-06-14', '2000-06-15',
                '2005-08-26', '2005-08-27', '2005-08-28',
                '2006-07-11', '2006-07-12', '2006-07-13',
                '2015-08-05', '2015-08-06', '2015-08-07',
                '2016-04-16', '2016-04-17', '2016-04-18',
                '2023-06-22', '2023-06-23', '2023-06-24']
    print("Getting Weather Data")
    df_weather = get_weather_data()

    print("Getting Flood Data")
    df_flood = get_flood_data()

    df = pd.merge(df_weather, df_flood, how='left', left_index=True, right_index=True)
    df.reset_index(inplace=True)
    df['target'] = 0
    df['target'] = df['date'].dt.date.astype('str').isin(targets).astype('int')

    return df

def get_folds(
    df: pd.DataFrame,
    fold_length: int,
    fold_stride: int) -> list[pd.DataFrame]:
    """
    This function slides through the Time Series dataframe of shape (n_timesteps, n_features) to create folds
    - of equal `fold_length`
    - using `fold_stride` between each fold

    Args:
        df (pd.DataFrame): Overall dataframe
        fold_length (int): How long each fold should be in rows
        fold_stride (int): How many timesteps to move forward between taking each fold

    Returns:
        List[pd.DataFrame]: A list where each fold is a dataframe within
    """

    folds = []
    for idx in range(0, len(df), fold_stride):
        # Exits the loop as soon as the last fold index would exceed the last index
        if (idx + fold_length) > len(df):
            break
        fold = df.iloc[idx:idx + fold_length, :]
        folds.append(fold)
    return folds

def train_test_split(fold:pd.DataFrame,
                     train_test_ratio: float,
                     input_length: int) -> tuple[pd.DataFrame]:
    """From a fold dataframe, take a train dataframe and test dataframe based on
    the split ratio.
    - df_train should contain all the timesteps until round(train_test_ratio * len(fold))
    - df_test should contain all the timesteps needed to create all (X_test, y_test) tuples

    Args:
        fold (pd.DataFrame): A fold of timesteps
        train_test_ratio (float): The ratio between train and test 0-1
        input_length (int): How long each X_i will be

    Returns:
        Tuple[pd.DataFrame]: A tuple of two dataframes (fold_train, fold_test)
    """

    # TRAIN SET
    last_train_idx = round(float(train_test_ratio) * len(fold))
    fold_train = fold.iloc[0:last_train_idx, :]

    # TEST SET
    first_test_idx = last_train_idx - int(input_length)
    fold_test = fold.iloc[first_test_idx:, :]


    return (fold_train, fold_test)

def get_data_with_cache(
        cache_path:Path,
        data_has_header=True
    ) -> pd.DataFrame:
    """
    Retrieve data from API, or from `cache_path` if the file exists
    Store at `cache_path` if retrieved from API for future use
    """
    if cache_path.is_file():
        print("\nLoad data from local CSV...")
        df = pd.read_csv(cache_path, header='infer' if data_has_header else None, dtype=DTYPES_RAW)
    else:
        print("\nLoad data from API server...")
        df = get_data_and_targets()
        #df = preprocess_features()

        # Store as CSV if the API returned at least one valid line
        if df.shape[0] > 1:
            df.to_csv(cache_path, header=data_has_header, index=False)

    print(f"✅ Data loaded, with shape {df.shape}")

    return df

def load_data_to_db(
        data: pd.DataFrame,
        gcp_project:str,
        bq_dataset:str,
        table: str,
        truncate: bool
    ) -> None:
    """
    - Save the DataFrame to BigQuery
    - Empty the table beforehand if `truncate` is True, append otherwise
    """

    assert isinstance(data, pd.DataFrame)
    full_table_name = f"{gcp_project}.{bq_dataset}.{table}"
    print("\nSave data to BigQuery @ {full_table_name}...:")

    # Load data onto full_table_name

    # 🎯 HINT for "*** TypeError: expected bytes, int found":
    # After preprocessing the data, your original column names are gone (print it to check),
    # so ensure that your column names are *strings* that start with either
    # a *letter* or an *underscore*, as BQ does not accept anything else

    # TODO: simplify this solution if possible, but students may very well choose another way to do it
    # We don't test directly against their own BQ tables, but only the result of their query
    data.columns = [f"_{column}" if not str(column)[0].isalpha() and not str(column)[0] == "_" else str(column) for column in data.columns]

    client = bigquery.Client()

    # Define write mode and schema
    write_mode = "WRITE_TRUNCATE" if truncate else "WRITE_APPEND"
    job_config = bigquery.LoadJobConfig(write_disposition=write_mode)

    print(f"\n{'Write' if truncate else 'Append'} {full_table_name} ({data.shape[0]} rows)")

    # Load data
    job = client.load_table_from_dataframe(data, full_table_name, job_config=job_config)
    result = job.result()  # wait for the job to complete

    print(f"✅ Data saved to bigquery, with shape {data.shape}")

def get_X_y_strides(fold: pd.DataFrame, input_length: int, output_length: int, horizon: int,
    sequence_stride: int) -> tuple[np.array]:
    """slides through a `fold` Time Series (2D array) to create sequences of equal
        * `input_length` for X,
        * `output_length` for y,
    using a temporal gap `sequence_stride` between each sequence

    Args:
        fold (pd.DataFrame): One single fold dataframe
        input_length (int): Length of each X_i
        output_length (int): Length of each y_i
        sequence_stride (int): How many timesteps to take before taking the next X_i

    Returns:
        Tuple[np.array]: A tuple of numpy arrays (X, y)
    """
    X, y = [], []

    for i in range(0, len(fold), int(sequence_stride)):
        # Exits the loop as soon as the last fold index would exceed the last index
        if (i + int(input_length) + int(output_length) + int(horizon)) >= len(fold):
            break
        X_i = fold.iloc[i:i + int(input_length), :]
        y_i = fold.iloc[i + int(input_length) + int(horizon) - 1:i + int(input_length) + int(horizon) - 1 + int(output_length), :]#[[TARGET]]

        # Asumo que el 13 es river_discharge
        y_i = y_i.iloc[:, [13]]

        X.append(X_i)
        y.append(y_i)

    return (np.array(X), np.array(y))

def api_request_pred():
    url = "https://api.open-meteo.com/v1/forecast"

    params = {
        "latitude": TALAGANTE_LAT,
        "longitude": TALAGANTE_LON,
        "hourly": "temperature_2m,rain,surface_pressure,windspeed_10m,winddirection_10m,soil_moisture_0_1cm,soil_moisture_1_3cm,soil_moisture_3_9cm,soil_moisture_9_27cm,shortwave_radiation",
        "past_days": "14",
        "timezone": "auto"
    }

    response = requests.get(url, params=params)

    data = response.json()

    df_weather = pd.DataFrame(data= data, columns= ['date', 'T(degC)', 'rain(mm)', 'surf_press(hPa)', 'wind_s(km/h)', 'wind_dir(deg)', 'soil_moisture_0_1cm(m3)', 'soil_moisture_1_3cm(m3)','soil_moisture_3_9cm(m3)','soil_moisture_9_27cm(m3)', 'radiation(W/m2)'])

    df_weather['date'] = data['hourly']['time']
    df_weather['T(degC)'] = data['hourly']['temperature_2m']
    df_weather['rain(mm)'] = data['hourly']['rain']
    df_weather['surf_press(hPa)'] = data['hourly']['surface_pressure']
    df_weather['wind_s(km/h)'] = data['hourly']['windspeed_10m']
    df_weather['wind_dir(deg)'] = data['hourly']['winddirection_10m']
    df_weather['soil_moisture_0_1cm(m3)'] = data['hourly']['soil_moisture_0_1cm']
    df_weather['soil_moisture_1_3cm(m3)'] = data['hourly']['soil_moisture_1_3cm']
    df_weather['soil_moisture_3_9cm(m3)'] = data['hourly']['soil_moisture_3_9cm']
    df_weather['soil_moisture_9_27cm(m3)'] = data['hourly']['soil_moisture_9_27cm']
    df_weather['radiation(W/m2)'] = data['hourly']['shortwave_radiation']

    df_weather['date'] = pd.to_datetime(df_weather['date'])

    return df_weather
