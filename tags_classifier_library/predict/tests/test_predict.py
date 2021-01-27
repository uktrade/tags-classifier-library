from tags_classifier_library.predict.model import inspect_model
from tags_classifier_library.predict.predict import predict, predict_data_hub_tags
from tags_classifier_library.predict.tests.constants import TEST_MODELS_PATH
from tags_classifier_library.predict.utils import fetch_prediction_data

INTERACTION_DATA = [
    {
        "id": 1,
        "policy_feedback_notes": "juliett quebec kilo juliett quebec kilo juliett kilo delta "
        "alpha delta delta alpha delta delta alpha delta delta alpha delta delta alpha juliett",
    },
    {
        "id": 2,
        "policy_feedback_notes": "quebec kilo juliett delta delta alpha delta delta alpha delta "
        "delta alpha delta delta alpha",
    },
    {"id": 3, "policy_feedback_notes": "delta"},
    {"id": 4, "policy_feedback_notes": "lorem ipsum"},
]

PREDICTION_COLUMNS = [
    "id",
    "sentence",
    "prediction",
    "prediction_prob",
]

DATA_HUB_PREDICTION_COLUMNS = [
    "id",
    "sentence",
    "tags_prediction",
    "tag_1",
    "probability_score_tag_1",
    "tag_2",
    "probability_score_tag_2",
    "tag_3",
    "probability_score_tag_3",
    "tag_4",
    "probability_score_tag_4",
    "tag_5",
    "probability_score_tag_5",
]


def test_predict():
    models_info = inspect_model(TEST_MODELS_PATH, ["models_general"])
    df = fetch_prediction_data(INTERACTION_DATA, "id", "policy_feedback_notes")

    prediction = predict(
        X_to_predict=df[["id", "sentence"]],
        tokenizer=None,
        models_info=models_info,
    )

    assert set(PREDICTION_COLUMNS).issubset(prediction.columns)

    for index, row in prediction.iterrows():
        assert row["sentence"] == INTERACTION_DATA[index]["policy_feedback_notes"]


def test_predict_data_hub_tags():
    models_info = inspect_model(TEST_MODELS_PATH, ["models_general", "models_covid"])

    assert len(models_info) == 20

    df = fetch_prediction_data(INTERACTION_DATA, "id", "policy_feedback_notes")

    # if first prediction results in `Juliett` tag being found, then
    # a second prediction is being run using `models_covid` and results are merged
    prediction = predict_data_hub_tags(df, models_info, covid_tag="Juliett")

    assert set(DATA_HUB_PREDICTION_COLUMNS).issubset(prediction.columns)

    for index, row in prediction.iterrows():
        assert row["sentence"] == INTERACTION_DATA[index]["policy_feedback_notes"]
