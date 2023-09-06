import pandas as pd
import requests
from params import *

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
    df_weather = get_weather_data()
    df_flood = get_flood_data()

    df = pd.merge(df_weather, df_flood, how='left', left_index=True, right_index=True)
    df['target'] = 0
    df.loc[targets, 'target'] = 1

    return df
