import pandas as pd
import numpy as np
from dateutil.parser import parse
import tensorflow as tf

from flood_prediction.params import *
from pathlib import Path

from flood_prediction.ml_logic.data_prep import api_request_pred, get_X_y_strides, get_folds, train_test_split, get_data_with_cache
from flood_prediction.ml_logic.model import init_model, fit_model
from flood_prediction.ml_logic.preprocessor import preprocess_features, preprocess_features_pred
from flood_prediction.ml_logic.registry import load_model, save_model, save_results

def preprocess() -> None:
    """
    - Getting raw dataset from data_prep.py
    - Process data
    """
    print("\n⭐️ Use case: preprocess")
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

def train() -> float:
    """
    - Train on the preprocessed dataset (which should be ordered by date)
    - Store training results and model weights

    return
    """
    print("\n⭐️ Use case: train")
    print("\nLoading preprocessed validation data...")

    # Load processed data using `get_data_with_cache` in chronological order
    data_processed_cache_path = Path(LOCAL_DATA_PATH).joinpath("processed", f"data_processed.csv")

    data_processed = get_data_with_cache(
        cache_path=data_processed_cache_path,
        data_has_header=False
    )

    if data_processed.shape[0] < 10:
        print("❌ Not enough processed data retrieved to train on")
        return None

    # Create (X_train, y_train, X_test, y_val)
    (whole_train, whole_test) = train_test_split(data_processed, TRAIN_TEST_RATIO, INPUT_LENGTH)
    X_train, y_train = get_X_y_strides(whole_train, INPUT_LENGTH, OUTPUT_LENGTH, HORIZON, SEQUENCE_STRIDE)
    # X_test, y_test = get_X_y_strides(whole_test, INPUT_LENGTH, OUTPUT_LENGTH, HORIZON, SEQUENCE_STRIDE)

    print(X_train.dtype)
    print(y_train.dtype)

    # Train model using `model.py`
    model = load_model()

    if model is None:
        print("\nModel not found, starting initialization")

        # init and compile
        model = init_model(X_train, y_train)


    # fitting model
    model, history = fit_model(model, X_train, y_train)

    val_mae = np.min(history.history['val_mae'])

    # Saving metrics and params locally
    # save_results(params=params, metrics=dict(mae=val_mae))

    # Save model weight on the hard drive (and optionally on GCS too!)
    save_model(model=model)

    print("✅ train() done \n")

    return val_mae

def evaluate():
    pass

def pred() -> np.ndarray:
    """
    Make a prediction using the latest trained model
    """

    print("\n⭐️ Use case: predict")

    # connect it to front end and API
    X_pred = api_request_pred()

    model = load_model()
    assert model is not None

    X_processed = preprocess_features_pred(X_pred)
    X_processed = tf.expand_dims(X_processed, axis=0)

    print(X_processed.shape)

    y_pred = model.predict(X_processed)

    print("\n✅ prediction done: ", y_pred, y_pred.shape, "\n")

if __name__ == "__main__":
    # preprocess()
    # train()
    # evaluate()
    pred()
