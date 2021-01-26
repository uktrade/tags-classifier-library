import os
from tempfile import TemporaryDirectory

import pandas as pd

from tags_classifier_library.predict.model import ModelInfo, inspect_model
from tags_classifier_library.train.preprocess import preprocess_for_training
from tags_classifier_library.train.tests.constants import \
    TEST_TRAINING_FILES_PATH
from tags_classifier_library.train.train import build_models_pipeline

METRIC_COLUMNS = ["model_for_tag", "size", "precisions", "recalls", "f1", "accuracy", "auc"]

GENERAL_TAGS = [
    "Alpha",
    "Beta",
    "Charlie",
    "Delta",
    "Echo",
    "Foxtrot",
    "Golf",
    "Hotel",
    "India",
    "Juliett",
]
COVID_TAGS = [
    "Juliett",
    "Kilo",
    "Lima",
    "Mike",
    "November",
    "Oscar",
    "Papa",
    "Quebec",
    "Romeo",
    "Sierra",
]


def test_training_general():
    with TemporaryDirectory() as tmp_dir_path:
        models_path = os.path.join(tmp_dir_path, "models_01012021")

        df = pd.read_csv(os.path.join(TEST_TRAINING_FILES_PATH, "train.csv"))
        df, my_tags = preprocess_for_training(df, tags_limit=2)

        metric_df_general, metric_df_covid = build_models_pipeline(
            df, models_path, tags_general=my_tags, tags_covid=[], overwrite=False
        )

        assert metric_df_covid is None
        assert set(METRIC_COLUMNS).issubset(metric_df_general.columns)
        assert set(GENERAL_TAGS).issubset(metric_df_general["model_for_tag"])

        models_info = inspect_model(models_path, ["models_general"])
        models_info.sort(key=lambda x: x.name)
        group_path = os.path.join(models_path, "models_general")
        for index, model_info in enumerate(models_info):
            tag = GENERAL_TAGS[index]
            assert model_info == ModelInfo(
                name=GENERAL_TAGS[index],
                group="models_general",
                path=os.path.join(group_path, tag),
                group_path=group_path,
            )


def test_training_covid():
    with TemporaryDirectory() as tmp_dir_path:
        models_path = os.path.join(tmp_dir_path, "models_01012021")

        df = pd.read_csv(os.path.join(TEST_TRAINING_FILES_PATH, "train-covid.csv"))
        df, my_tags = preprocess_for_training(df, tags_limit=2)
        metric_df_general, metric_df_covid = build_models_pipeline(
            df,
            models_path,
            tags_general=[],
            tags_covid=my_tags,
            covid_tag="Juliett",
            overwrite=False,
        )
        assert metric_df_general is None
        assert set(METRIC_COLUMNS).issubset(metric_df_covid.columns)
        assert set(COVID_TAGS).issubset(metric_df_covid["model_for_tag"])

        models_info = inspect_model(models_path, ["models_covid"])
        models_info.sort(key=lambda x: x.name)
        group_path = os.path.join(models_path, "models_covid")
        for index, model_info in enumerate(models_info):
            tag = COVID_TAGS[index]
            assert model_info == ModelInfo(
                name=COVID_TAGS[index],
                group="models_covid",
                path=os.path.join(group_path, tag),
                group_path=group_path,
            )
