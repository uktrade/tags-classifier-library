import json
import logging
import os
from itertools import compress

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Conv1D, Dense, Dropout, Embedding, GlobalMaxPool1D
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.preprocessing.text import Tokenizer

from tags_classifier_library.settings import (
    EMBEDDING_DIM,
    MAX_NB_WORDS,
    MAX_SEQUENCE_LENGTH,
    PROBABILITY_THRESHOLD,
)

logger = logging.getLogger(__name__)


def report_metric_per_model(actual, predict, average_type="binary"):
    precisions = precision_score(actual, predict, average=average_type)
    recalls = recall_score(actual, predict, average=average_type)
    f1 = f1_score(actual, predict, average=average_type)
    accuracy = accuracy_score(actual, predict)
    auc = roc_auc_score(actual, predict)
    logger.info(f"Precision = {precisions}")
    logger.info(f"Recall = {recalls}")
    logger.info(f"f1 = {f1}")
    logger.info(f"Accuracy = {accuracy}")
    logger.info(f"AUC = {auc}")

    return precisions, recalls, f1, accuracy, auc


def report_metrics_for_all(tag_size, tag_precisions, tag_recalls, tag_f1, tag_accuracy, tag_auc):
    size_df = pd.DataFrame.from_dict(tag_size, orient="index")
    size_df = size_df.rename(columns={0: "size"})
    size_df = size_df[["size"]]

    precisions_df = pd.DataFrame.from_dict(tag_precisions, orient="index")
    precisions_df = precisions_df.rename(columns={0: "precisions"})

    recalls_df = pd.DataFrame.from_dict(tag_recalls, orient="index")
    recalls_df = recalls_df.rename(columns={0: "recalls"})

    f1_df = pd.DataFrame.from_dict(tag_f1, orient="index")
    f1_df = f1_df.rename(columns={0: "f1"})

    accuracy_df = pd.DataFrame.from_dict(tag_accuracy, orient="index")
    accuracy_df = accuracy_df.rename(columns={0: "accuracy"})

    auc_df = pd.DataFrame.from_dict(tag_auc, orient="index")
    auc_df = auc_df.rename(columns={0: "auc"})

    metric_df = pd.concat([size_df, precisions_df, recalls_df, f1_df, accuracy_df, auc_df], axis=1)
    metric_df["model_for_tag"] = metric_df.index
    metric_df = metric_df[["model_for_tag"] + list(metric_df.columns[:-1])]

    return metric_df


def build_tokens(df, model_path):
    tokenizer = Tokenizer(
        num_words=MAX_NB_WORDS,
        filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
        lower=True,
    )

    tokenizer.fit_on_texts(df["sentence"].values)
    word_index = tokenizer.word_index
    logger.info(f"Found {len(word_index)} unique tokens")

    tokenizer_json = tokenizer.to_json()

    if not os.path.exists(model_path):
        os.makedirs(model_path)
    with open(f"{model_path}cnn_tokenizer.json", "w", encoding="utf-8") as f:
        f.write(json.dumps(tokenizer_json, ensure_ascii=False))

    return tokenizer, tokenizer_json


def build_train_set(df, tokenizer, tags):
    """
    Build train set
    """
    logger.info(f"MAX_SEQUENCE_LENGTH: {MAX_SEQUENCE_LENGTH}")
    X = tokenizer.texts_to_sequences(df["sentence"].values)
    X = sequence.pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH, padding="post", truncating="post")
    logger.info(f"Shape of data tensor: {X.shape}")

    Y = df[tags].values
    logger.info(f"Shape of label tensor: {Y.shape}")

    sent_train, sent_test, X_train, X_test, Y_train, Y_test = train_test_split(
        df["sentence"].values, X, Y, test_size=0.10, random_state=42, shuffle=True
    )

    logger.info(f"Shape of X_train: {X_train.shape}")
    logger.info(f"Shape of Y_train: {Y_train.shape}")
    logger.info(f"Shape of X_test: {X_test.shape}")
    logger.info(f"Shape of Y_test: {Y_test.shape}")

    sent_train, sent_val, X_train, X_val, Y_train, Y_val = train_test_split(
        sent_train, X_train, Y_train, test_size=0.10, random_state=42, shuffle=True
    )

    return (
        sent_train,
        sent_val,
        sent_test,
        X_train,
        X_val,
        X_test,
        Y_train,
        Y_val,
        Y_test,
    )


def cnn():
    model = Sequential()
    model.add(
        Embedding(
            MAX_NB_WORDS,
            EMBEDDING_DIM,
            input_length=MAX_SEQUENCE_LENGTH,
            trainable=True,
        )
    )
    model.add(Conv1D(128, 5, activation="relu"))
    model.add(GlobalMaxPool1D())
    model.add(Dense(8, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation="sigmoid"))

    return model


def train_model(model, X_train, Y_train, X_val, Y_val, class_weight):
    model.compile(
        loss="binary_crossentropy",
        optimizer="adam",
        metrics=[
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.BinaryCrossentropy(name="entropy"),
            tf.keras.metrics.TruePositives(name="tp"),
            tf.keras.metrics.TrueNegatives(name="tn"),
            tf.keras.metrics.FalsePositives(name="fp"),
            tf.keras.metrics.FalseNegatives(name="fn"),
        ],
    )

    epochs = 20
    batch_size = 64

    model.fit(
        X_train,
        Y_train,
        epochs=epochs,
        class_weight={0: 1, 1: class_weight},
        batch_size=batch_size,
        validation_data=(X_val, Y_val),
        callbacks=[
            EarlyStopping(
                monitor="val_entropy",
                patience=3,
                min_delta=0.0001,
                mode="min",
                restore_best_weights=True,
            )
        ],
        verbose=0,
    )

    return model


def build_models_for_tags(
    tags, sent_test, X_train, X_val, Y_train, Y_val, X_test, Y_test, model_path, overwrite=False
):
    tag_precisions = dict()
    tag_recalls = dict()
    tag_f1 = dict()
    tag_accuracy = dict()
    tag_auc = dict()
    tag_size = dict()

    Y_test_predict = np.zeros(Y_test.shape)
    Y_test_predict_prob = np.zeros(Y_test.shape)

    logger.info(f"build models for tags:{tags}")

    for tag in tags:
        logger.info("-------------" * 5)
        logger.info(f"model for {tag}")
        i = tags.index(tag)

        Y_train_tag = Y_train[:, i]
        logger.info(
            f"check Y_train_tag:  positive case:{Y_train_tag.sum()}; shape:{Y_train_tag.shape}"
        )
        Y_val_tag = Y_val[:, i]
        class_size = (
            sum(Y_test[:, i] == 1) + sum(Y_train[:, i] == 1),
            (sum(Y_test[:, i] == 1) + sum(Y_train[:, i] == 1))
            / (Y_test.shape[0] + Y_train.shape[0]),
        )
        class_weight = (sum(Y_test[:, i] == 0) + sum(Y_train[:, i] == 0)) / (
            sum(Y_test[:, i] == 1) + sum(Y_train[:, i] == 1)
        )
        logger.info(f"check class_weight:  {class_weight}")

        model = cnn()
        model = train_model(model, X_train, Y_train_tag, X_val, Y_val_tag, class_weight)
        path = os.path.join(model_path, "_".join(tag.split(" ")))
        model.save(path, overwrite=overwrite)

        data_to_evaluate = X_test
        sent_to_evaluate = sent_test

        test_predictions_prob_tag = model.predict(data_to_evaluate)
        test_predictions_class_tag = (test_predictions_prob_tag > PROBABILITY_THRESHOLD) + 0
        Y_test_predict[:, i] = np.concatenate((test_predictions_class_tag))
        Y_test_predict_prob[:, i] = np.concatenate((test_predictions_prob_tag))

        precisions = 0
        recalls = 0
        f1 = 0
        accuracy = 0
        auc = 0

        tag_precisions[tags[i]] = precisions
        tag_recalls[tags[i]] = recalls
        tag_f1[tags[i]] = f1
        tag_accuracy[tags[i]] = accuracy
        tag_auc[tags[i]] = auc
        tag_size[tags[i]] = class_size

    metric_df = report_metrics_for_all(
        tag_size, tag_precisions, tag_recalls, tag_f1, tag_accuracy, tag_auc
    )

    logger.info(f"check metric columns:  {metric_df.columns}")
    metric_df.to_csv(model_path + "/models_metrics_cnn.csv", index=False)

    actual = []
    predict = []
    sentence = []

    for i in np.arange(0, sent_to_evaluate.shape[0]):
        sentence.append(sent_to_evaluate[i])
        actual.append(list(compress(tags, Y_test[i])))
        predict.append(list(compress(tags, Y_test_predict_prob[i] > PROBABILITY_THRESHOLD)))

    return metric_df


def check_result(tag, tag_i, X_test, Y_test, sent_test, m):
    logger.info(f"check result for tag {tag}")

    test_predictions_prob_tag1 = m.predict(X_test)
    fp_sen = sent_test[
        (np.concatenate(test_predictions_prob_tag1) > PROBABILITY_THRESHOLD)
        & (Y_test[:, tag_i] == 0)
    ]
    fp_p = test_predictions_prob_tag1[
        np.concatenate((test_predictions_prob_tag1 >= PROBABILITY_THRESHOLD))
        & (Y_test[:, tag_i] == 0)
    ]

    tp_sen = sent_test[
        (np.concatenate(test_predictions_prob_tag1) > PROBABILITY_THRESHOLD)
        & (Y_test[:, tag_i] == 1)
    ]
    tp_p = test_predictions_prob_tag1[
        np.concatenate((test_predictions_prob_tag1 >= PROBABILITY_THRESHOLD))
        & (Y_test[:, tag_i] == 1)
    ]

    fn_sen = sent_test[
        (np.concatenate(test_predictions_prob_tag1) <= PROBABILITY_THRESHOLD)
        & (Y_test[:, tag_i] == 1)
    ]
    fn_p = test_predictions_prob_tag1[
        np.concatenate((test_predictions_prob_tag1 < PROBABILITY_THRESHOLD))
        & (Y_test[:, tag_i] == 1)
    ]

    tn_sen = sent_test[
        (np.concatenate(test_predictions_prob_tag1) <= PROBABILITY_THRESHOLD)
        & (Y_test[:, tag_i] == 0)
    ]
    tn_p = test_predictions_prob_tag1[
        np.concatenate((test_predictions_prob_tag1 < PROBABILITY_THRESHOLD))
        & (Y_test[:, tag_i] == 0)
    ]

    return fp_sen, tp_sen, fn_sen, tn_sen, fp_p, tp_p, fn_p, tn_p


def build_models_pipeline(
    df, path, tags_general, tags_covid, covid_tag="Covid-19", overwrite=False
):
    metric_df_general = None
    metric_df_covid = None

    if len(tags_general) > 0:
        model_path = _get_models_path(path, "models_general")
        tokenizer_general, tokenizer_json_general = build_tokens(df, model_path)
        (
            sent_train,
            sent_val,
            sent_test,
            X_train,
            X_val,
            X_test,
            Y_train,
            Y_val,
            Y_test,
        ) = build_train_set(df, tokenizer_general, tags_general)

        metric_df_general = build_models_for_tags(
            tags_general,
            sent_test,
            X_train,
            X_val,
            Y_train,
            Y_val,
            X_test,
            Y_test,
            model_path=model_path,
            overwrite=overwrite,
        )

    if len(tags_covid) > 0:
        model_path = _get_models_path(path, "models_covid")
        df_covid = df[df[covid_tag] > 0][["sentence"] + tags_covid]
        tokenizer_covid, tokenizer_json_covid = build_tokens(df_covid, model_path)
        (
            sent_train,
            sent_val,
            sent_test,
            X_train,
            X_val,
            X_test,
            Y_train,
            Y_val,
            Y_test,
        ) = build_train_set(df_covid, tokenizer_covid, tags_covid)

        metric_df_covid = build_models_for_tags(
            tags_covid,
            sent_test,
            X_train,
            X_val,
            Y_train,
            Y_val,
            X_test,
            Y_test,
            model_path=model_path,
            overwrite=overwrite,
        )

    return (metric_df_general, metric_df_covid)


def get_model_performance(today, **context):
    metrics1 = pd.read_csv("models_" + today + "/models_covid/models_metrics_cnn.csv")
    metrics2 = pd.read_csv("models_" + today + "/models_general/models_metrics_cnn.csv")

    metrics = pd.concat([metrics1, metrics2])
    metrics = metrics.reset_index()
    # metrics = context['task_instance'].xcom_pull(task_ids='build-model')

    metrics["model_version"] = "models_" + today
    cols = metrics.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    metrics = metrics[cols]

    metrics_json = metrics.to_json(orient="records")
    logger.info(f"performance for this model version is: {metrics}")

    return json.loads(metrics_json)


def _get_models_path(path, name):
    return os.path.join(path, name)
