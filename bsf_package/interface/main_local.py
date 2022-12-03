import numpy as np
import pandas as pd
import os

from tests.test_base import write_result

from taxifare.ml_logic.data import clean_data
from taxifare.ml_logic.model import initialize_model, compile_model, train_model
from taxifare.ml_logic.preprocessor import preprocess_features
from taxifare.ml_logic.registry import save_model, load_model

from taxifare.ml_logic.params import (
    CHUNK_SIZE,
    DTYPES_RAW_OPTIMIZED_HEADLESS,
    DTYPES_RAW_OPTIMIZED,
    DTYPES_PROCESSED_OPTIMIZED,
    COLUMN_NAMES_RAW,
    DATASET_SIZE,
    VALIDATION_DATASET_SIZE,
    LOCAL_DATA_PATH
)


def preprocess_and_train():
    """
    Load historical data in memory, clean and preprocess it, train a Keras model on it,
    save the model, and finally compute & save a performance metric
    on a validation set holdout at the `model.fit()` level
    """

    print("\n⭐️ Use case: preprocess and train basic")


    # Retrieve raw data
    data_raw_path = os.path.join(LOCAL_DATA_PATH, "raw", f"train_{DATASET_SIZE}.csv")
    data = pd.read_csv(data_raw_path, dtype=DTYPES_RAW_OPTIMIZED)

    # Clean data using ml_logic.data.clean_data
    # $CODE_BEGIN
    data_cleaned = clean_data(data)
    # $CODE_END

    # Create X, y
    # $CODE_BEGIN
    X = data_cleaned.drop("fare_amount", axis=1)
    y = data_cleaned[["fare_amount"]]
    # $CODE_END

    # Preprocess X using `preprocessor.py`
    # $CODE_BEGIN
    X_processed = preprocess_features(X)
    # $CODE_END

    # Train model on X_processed and y, using `model.py`
    model = None
    learning_rate = 0.001
    batch_size = 256
    patience = 2

    # $CODE_BEGIN
    model = initialize_model(X_processed)
    model = compile_model(model, learning_rate)
    model, history = train_model(model, X_processed, y, batch_size=batch_size, patience=patience, validation_split=0.3)
    # $CODE_END

    # Compute the validation metric (min val mae of the holdout set)
    metrics = dict(mae=np.min(history.history['val_mae']))

    # Save trained model
    params = dict(
        learning_rate=learning_rate,
        batch_size=batch_size,
        patience=patience
    )

    save_model(model, params=params, metrics=metrics)

    # 🧪 Write outputs so that they can be tested by make test_train_at_scale (do not remove)
    write_result(name="test_preprocess_and_train", subdir="train_at_scale", metrics=metrics)

    print("✅ preprocess_and_train() done")

# $ERASE_BEGIN
def preprocess(source_type='train'):
    """
    Preprocess the dataset iteratively by loading data in chunks fitting in memory,
    processing each chunk, appending each of them to a final, preprocessed dataset,
    and saving that dataset in CSV format.

    Parameter:
    - source_type could be 'train' or 'val'
    """

    print("\n⭐️ Use case: preprocess")

    # Local saving paths given to you (do not overwrite these data_path variables)
    source_name = f"{source_type}_{DATASET_SIZE}.csv"
    destination_name = f"{source_type}_processed_{DATASET_SIZE}.csv"

    data_raw_path = os.path.abspath(os.path.join(LOCAL_DATA_PATH, "raw", source_name))
    data_processed_path = os.path.abspath(os.path.join(LOCAL_DATA_PATH, "processed", destination_name))

    # Iterate over the dataset, in chunks
    chunk_id = 0

    # Let's loop until we reach the end of the dataset, then `break` out
    while (True):
        print(f"Processing chunk n°{chunk_id}...")

        try:
            # Load the chunk numbered `chunk_id` of size `CHUNK_SIZE` into memory 

            # 🎯 Hint: check out pd.read_csv(skiprows=..., nrows=..., headers=...)
            # We advise you to always load data with `header=None`, and add back column names using COLUMN_NAMES_RAW

            # $CODE_BEGIN
            data_raw_chunk = pd.read_csv(
                data_raw_path,
                header=None,
                skiprows=(chunk_id * CHUNK_SIZE) + 1, # first chunk has headers, we don't want them
                nrows=CHUNK_SIZE,
                dtype=DTYPES_RAW_OPTIMIZED_HEADLESS,
            )

            assert dict(data_raw_chunk.dtypes) == DTYPES_RAW_OPTIMIZED_HEADLESS # read_csv(dtypes=...) silently fails to convert dtypes if column names don't match the dictionary key provided

            data_raw_chunk.columns = COLUMN_NAMES_RAW
            # $CODE_END

        except pd.errors.EmptyDataError:
            # 🤔 Question: what should you do when you reach the end of the CSV?
            # $CODE_BEGIN
            data_raw_chunk = None  # end of data
            # $CODE_END

        # $DEL_END
        # Break out of while loop if data is `None`
        if data_raw_chunk is None:
            break
        # $DEL_END

        # Clean chunk
        # $CODE_BEGIN
        data_clean_chunk = clean_data(data_raw_chunk)

        # Break out of while loop if cleaning removed all rows
        if len(data_clean_chunk) == 0:
            break
        # $CODE_END

        # Create X_chunk, y_chunk
        # $CODE_BEGIN
        X_chunk = data_clean_chunk.drop("fare_amount", axis=1)
        y_chunk = data_clean_chunk[["fare_amount"]]
        # $CODE_END

        # Create X_processed_chunk and concatenate (X_processed_chunk, y_chunk) into data_processed_chunk
        # $CODE_BEGIN
        X_processed_chunk = preprocess_features(X_chunk)

        data_processed_chunk = pd.DataFrame(
            np.concatenate((X_processed_chunk, y_chunk), axis=1))
        # $CODE_END

        # Save and append the chunk of the preprocessed dataset to a local CSV
        # Keep headers on the first chunk
        # By convention, we'll always save CSVs with headers in this challenge
        # 🎯 Hint: check out pd.to_csv(mode=...)
        # $CODE_BEGIN
        data_processed_chunk.to_csv(
            data_processed_path,
            mode="w" if chunk_id==0 else "a",
            header=chunk_id == 0, # Header only for first chunk
            index=False
        )
        # $CODE_END

        chunk_id += 1

    # 🧪 Write outputs so that they can be tested by make test_train_at_scale (do not remove)
    data_processed = pd.read_csv(data_processed_path, header=None, skiprows=1, dtype=DTYPES_PROCESSED_OPTIMIZED).to_numpy()
    write_result(name="test_preprocess", subdir="train_at_scale", data_processed_head=data_processed[0:10])

    print("✅ data_processed saved entirely")


def train():
    """
    Train on the full (already preprocessed) dataset, by loading it
    chunk-by-chunk, and updating the weight of the model for each chunk.
    Save model, compute validation metrics on a holdout validation set that is
    common to all chunks.
    """
    print("\n ⭐️ Use case: train")

    # Validation set: load a validation set common to all chunks and create X_val, y_val
    data_val_processed_path = os.path.abspath(os.path.join(
        LOCAL_DATA_PATH, "processed", f"val_processed_{VALIDATION_DATASET_SIZE}.csv"
    ))

    data_val_processed = pd.read_csv(
        data_val_processed_path,
        skiprows= 1, # skip header
        header=None,
        dtype=DTYPES_PROCESSED_OPTIMIZED
    ).to_numpy()

    X_val = data_val_processed[:, :-1]
    y_val = data_val_processed[:, -1]

    # Iterate over the full training dataset in chunks.
    # Break out of the loop if you receive no more data to train upon!
    model = None
    chunk_id = 0
    metrics_val_list = []  # store each metrics_val_chunk

    while (True):
        print(f"Loading and training on preprocessed chunk n°{chunk_id}")

        # Load chunk of preprocess data and create (X_train_chunk, y_train_chunk)
        path = os.path.abspath(os.path.join(
            LOCAL_DATA_PATH, "processed", f"train_processed_{DATASET_SIZE}.csv"))

        try:
            data_processed_chunk = pd.read_csv(
                path,
                skiprows=(chunk_id * CHUNK_SIZE) + 1, # skip header
                header=None,
                nrows=CHUNK_SIZE,
                dtype=DTYPES_PROCESSED_OPTIMIZED,
            ).to_numpy()

        except pd.errors.EmptyDataError:
            data_processed_chunk = None  # end of data

        # Break out of while loop if we have no data to train upon
        if data_processed_chunk is None:
            break

        X_train_chunk = data_processed_chunk[:, :-1]
        y_train_chunk = data_processed_chunk[:, -1]

        learning_rate = 0.001
        batch_size = 256
        patience=2

        # Train a model *incrementally*, and store the val MAE of each chunk in `metrics_val_list`
        # $CODE_BEGIN
        if model is None:
            model = initialize_model(X_train_chunk)

        model = compile_model(model, learning_rate)

        model, history = train_model(
            model,
            X_train_chunk,
            y_train_chunk,
            batch_size=batch_size,
            patience=patience,
            validation_data=(X_val, y_val)
        )

        metrics_val_chunk = np.min(history.history['val_mae'])
        metrics_val_list.append(metrics_val_chunk)

        print(metrics_val_chunk)
        # $CODE_END

        chunk_id += 1

    # Return the last value of the validation MAE
    val_mae = metrics_val_list[-1]

    # Save model and training params
    params = dict(
        learning_rate=learning_rate,
        batch_size=batch_size,
        patience=patience,
        incremental=True,
        chunk_size=CHUNK_SIZE
    )

    print(f"\n✅ Trained with MAE: {round(val_mae, 2)}")

    save_model(model=model, params=params, metrics=dict(mae=val_mae))

    print("✅ Model trained and saved")

# $ERASE_END

def pred(X_pred: pd.DataFrame = None) -> np.ndarray:

    if X_pred is None:
        X_pred = pd.DataFrame(dict(
            key=["2013-07-06 17:18:00"],  # useless but the pipeline requires it
            pickup_datetime=["2013-07-06 17:18:00 UTC"],
            pickup_longitude=[-73.950655],
            pickup_latitude=[40.783282],
            dropoff_longitude=[-73.984365],
            dropoff_latitude=[40.769802],
            passenger_count=[1]
        ))

    model = load_model()

    # Preprocess the new data
    # $CODE_BEGIN
    X_processed = preprocess_features(X_pred)
    # $CODE_END

    # Make a prediction
    # $CODE_BEGIN
    y_pred = model.predict(X_processed)
    # $CODE_END

    # 🧪 Write outputs so that they can be tested by make test_train_at_scale (do not remove)
    write_result(name="test_pred", subdir="train_at_scale", y_pred=y_pred)
    print("✅ prediction done: ", y_pred, y_pred.shape)

    return y_pred


if __name__ == '__main__':
    try:
        preprocess_and_train()
        pred()
    except:
        import ipdb, traceback, sys
        extype, value, tb = sys.exc_info()
        traceback.print_exc()
        ipdb.post_mortem(tb)
