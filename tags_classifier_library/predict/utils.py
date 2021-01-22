import json
import logging

import pandas as pd
from tensorflow.keras.preprocessing import sequence

from tags_classifier_library.predict.preprocess import \
    preprocess_for_prediction
from tags_classifier_library.settings import MAX_SEQUENCE_LENGTH

logger = logging.getLogger(__name__)


def transform_X(X_text, tokenizer):
    X = tokenizer.texts_to_sequences(X_text)
    X = sequence.pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH, padding="post", truncating="post")
    return X


def convert_data_frames_to_json(df, context):
    df_json = df.to_json(orient="records")
    return json.loads(df_json)


def fetch_prediction_data(rows, id_column_name, text_column_name):
    logger.info("starting fetching data")

    transformed_rows = transform_rows(rows, id_column_name, text_column_name)
    df = pd.DataFrame(transformed_rows)

    processed_df = preprocess_for_prediction(df)

    logger.info(f"check df shape: {df.shape}")
    logger.info(f"df head: {df.head()}")

    return processed_df


def transform_rows(rows, id_column_name, text_column_name):
    for row in rows:
        yield {
            "id": row[id_column_name],
            "sentence": row[text_column_name],
        }


def update_prediction_dict(d1, d2):
    d = d1.copy()
    if not isinstance(d2, float):
        d.update(d2)
    return d


def convert_prob_dict(a_dict):
    a_dict = {k: v for k, v in sorted(a_dict.items(), key=lambda item: item[1] * -1)}
    converted_dict = {}
    for i, (k, v) in enumerate(a_dict.items()):
        converted_dict["tag_" + str(i + 1)] = k
        converted_dict["probability_score_tag_" + str(i + 1)] = v
    for i in range(1, 6):
        if "tag_" + str(i) not in converted_dict:
            converted_dict["tag_" + str(i)] = None
            converted_dict["probability_score_tag_" + str(i)] = None
    return converted_dict


def update_df_column_with_prob(df):
    df["prediction_prob_top_5_converted"] = df["prediction_prob_top_5"].apply(
        lambda x: convert_prob_dict(x)
    )
    for i in range(1, 6):
        df["tag_" + str(i)] = df["prediction_prob_top_5_converted"].apply(
            lambda x: x["tag_" + str(i)]
        )
        df["probability_score_tag_" + str(i)] = df["prediction_prob_top_5_converted"].apply(
            lambda x: x["probability_score_tag_" + str(i)], 2
        )
    del df["prediction_prob_top_5_converted"]
    del df["prediction_prob_top_5"]
    return df
