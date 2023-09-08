import numpy as np
import pandas as pd

from flood_prediction.ml_logic.encoders import transform_soil_moisture_features, transform_wind_features, transform_time_features, transform_soil_moisture_features_pred

def preprocess_features(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocessing features

    Keyword arguments:
    df -- dataframe with features
    Return: pandas dataframe with preprocessed features
    """
    df = transform_soil_moisture_features(df)
    df = transform_wind_features(df)
    df = transform_time_features(df)

    return df

def preprocess_features_pred(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocessing prediction features (soil moisture is the only different function)

    Keyword arguments:
    df -- dataframe with features
    Return: pandas dataframe with preprocessed features
    """
    df = transform_soil_moisture_features_pred(df)
    df = transform_wind_features(df)
    df = transform_time_features(df)


    return df
