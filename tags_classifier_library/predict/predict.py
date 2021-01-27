import json
import logging
import operator
import os
from itertools import compress
from typing import List

import numpy as np
import pandas as pd
from keras_preprocessing.text import tokenizer_from_json

from tags_classifier_library.predict.model import ModelInfo, get_model
from tags_classifier_library.predict.utils import (
    transform_X,
    update_df_column_with_prob,
    update_prediction_dict,
)
from tags_classifier_library.settings import PROBABILITY_THRESHOLD

logger = logging.getLogger(__name__)


def top_5_labels(probabilities: dict, threshold=0.5):
    probabilities_final = {k: np.round(v, 2) for k, v in probabilities.items() if v > threshold}
    covid_tags = [i for i in probabilities_final.keys() if i.lower().startswith("covid")]
    if len(covid_tags) == 1:
        probabilities_final = {
            k.replace("Covid-19", "Covid-19 General"): v for k, v in probabilities_final.items()
        }
    elif len(covid_tags) > 1:
        probabilities_final.pop("Covid-19", None)
    probabilities_final = dict(
        sorted(probabilities_final.items(), key=operator.itemgetter(1), reverse=True)[:5]
    )
    if isinstance(probabilities_final, float) or len(probabilities_final) == 0:
        probabilities_final = {"General": 0}
    return probabilities_final


def predict_data_hub_tags(df, models_info: List[ModelInfo], covid_tag="Covid-19"):
    models_covid = [tag for tag in models_info if "models_covid" in tag.group]
    models_general = [tag for tag in models_info if "models_general" in tag.group]

    logger.info(f"check general models: {os.listdir(models_general[0].group_path)}")
    logger.info(f"check covid models: {os.listdir(models_covid[0].group_path)}")

    prediction_on_general_data = predict(
        X_to_predict=df,
        tokenizer=None,
        models_info=models_general,
    )

    covid_to_predict = prediction_on_general_data[
        prediction_on_general_data["prediction"].apply(lambda x: covid_tag in x)
    ]
    logger.info(f"check covid df shape: {covid_to_predict.shape[0]}")

    if covid_to_predict.shape[0] > 0:
        prediction_on_covid_data = predict(
            X_to_predict=covid_to_predict[["id", "sentence"]],
            tokenizer=None,
            models_info=models_covid,
        )

        predictions = prediction_on_general_data.merge(
            prediction_on_covid_data, left_on="id", right_on="id", how="left"
        )
        predictions["prediction_prob"] = predictions.apply(
            lambda x: update_prediction_dict(x["prediction_prob_x"], x["prediction_prob_y"]),
            axis=1,
        )

    else:
        predictions = prediction_on_general_data.copy()

    predictions["prediction_prob_top_5"] = predictions.apply(
        lambda x: top_5_labels(x["prediction_prob"], threshold=PROBABILITY_THRESHOLD),
        axis=1,
    )
    predictions["tags_prediction"] = predictions.apply(
        lambda x: list(x["prediction_prob_top_5"].keys()), axis=1
    )
    predictions["tags_prediction"] = predictions["tags_prediction"].apply(lambda x: ",".join(x))

    if "sentence_y" in predictions.columns:
        del predictions["sentence_y"]
    predictions = predictions.rename(columns={"sentence_x": "sentence"})

    predictions = predictions[["id", "sentence", "tags_prediction", "prediction_prob_top_5"]]
    predictions = update_df_column_with_prob(predictions)
    return predictions


def predict(X_to_predict, tokenizer, models_info: List[ModelInfo]):
    logger.info("Start making prediction")

    # assume that all models come from a single model group (directory)
    group_path = models_info[0].group_path

    if not tokenizer:
        logger.info("Tokenizer not provided, loading the default one.")
        tokenizer_path = os.path.join(group_path, "cnn_tokenizer.json")
        with open(tokenizer_path) as f:
            tokenizer_json = json.load(f)
            tokenizer = tokenizer_from_json(tokenizer_json)

    ids = X_to_predict["id"]
    X_to_predict = X_to_predict["sentence"]
    text_to_predict = X_to_predict.copy()

    X_to_predict = transform_X(X_to_predict.values, tokenizer)
    Y_test_predict = np.zeros((X_to_predict.shape[0], len(models_info)))
    Y_test_predict_prob = np.zeros((X_to_predict.shape[0], len(models_info)))

    for ind, tag_i in enumerate(models_info):
        logger.info(f"Predicting for tag {ind}, {tag_i.name}")
        model = get_model(tag_i.path)
        test_predictions_prob_tag = model.predict(X_to_predict)
        test_predictions_class_tag = (test_predictions_prob_tag > PROBABILITY_THRESHOLD) + 0
        Y_test_predict_prob[:, ind] = np.concatenate(test_predictions_prob_tag)
        Y_test_predict[:, ind] = np.concatenate(test_predictions_class_tag)

    predict = []
    sentence = []
    predict_prob = []

    tag_labels = [model.name for model in models_info]
    for i in np.arange(0, X_to_predict.shape[0]):
        sentence.append(X_to_predict[i])
        predict.append(list(compress(tag_labels, Y_test_predict_prob[i] > PROBABILITY_THRESHOLD)))
        predict_prob.append(dict(zip(tag_labels, Y_test_predict_prob[i])))

    prediction_on_data = pd.DataFrame(
        {
            "id": ids,
            "sentence": text_to_predict,
            "prediction": predict,
            "prediction_prob": predict_prob,
        }
    )

    return prediction_on_data
