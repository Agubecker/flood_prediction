import pandas as pd
import numpy as np

from flood_prediction.params import *
from pathlib import Path

# add this files to ml_logic folder
from flood_prediction.ml_logic.data_prep import get_folds, train_test_split, get_data_with_cache
from flood_prediction.ml_logic.model import init_model, compile_model, fit_model, evaluate_model
from flood_prediction.ml_logic.preprocessor import preprocess_features
#from flood_prediction.registry import load_model, save_model, save_results
#from flood_prediction.registry import mlflow_run, mlflow_transition_model

def preprocess() -> None:
    """
    - Getting raw dataset from data_prep.py (In a future should be from BQ)
    - Process data
    """
    # Retriving data
    data_query_cache_path = Path(LOCAL_DATA_PATH).joinpath("raw", f"data_raw.csv")
    df = get_data_with_cache(cache_path=data_query_cache_path)

    # Splitting data into X and y
    y = df[["river_discharge(m3/s)"]]
    X = df.drop(columns=["target", "river_discharge(m3/s)"], axis=1)

    # Preprocessing data
    X_processed = preprocess_features(X)

    # Creating dataframe with processed data
    data_processed = pd.concat([X_processed, y], axis=1)

    # Storing processed data
    data_processed_path = Path(LOCAL_DATA_PATH).joinpath("processed", f"data_processed.csv")
    data_processed.to_csv(data_processed_path, index=False)

    print("✅ preprocess() done \n")

# def train()

# def evaluate()

# def pred()

if __name__ == "__main__":
    preprocess()
    #train()
    #evaluate()
    #pred()
