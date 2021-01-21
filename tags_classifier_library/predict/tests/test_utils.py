from tags_classifier_library.predict.model import ModelInfo, inspect_model
from tags_classifier_library.predict.tests.constants import TEST_MODELS_PATH


def test_inspect_model():
    models_info = inspect_model(TEST_MODELS_PATH, ["models_general"])
    models_info.sort(key=lambda x: x.name)

    assert models_info == [
        ModelInfo(
            name="Alpha",
            group="models_general",
            path="./tags_classifier_library/predict/tests/models_test/models_general/Alpha",
            group_path="./tags_classifier_library/predict/tests/models_test/models_general",
        ),
        ModelInfo(
            name="Beta",
            group="models_general",
            path="./tags_classifier_library/predict/tests/models_test/models_general/Beta",
            group_path="./tags_classifier_library/predict/tests/models_test/models_general",
        ),
        ModelInfo(
            name="Charlie",
            group="models_general",
            path="./tags_classifier_library/predict/tests/models_test/models_general/Charlie",
            group_path="./tags_classifier_library/predict/tests/models_test/models_general",
        ),
        ModelInfo(
            name="Delta",
            group="models_general",
            path="./tags_classifier_library/predict/tests/models_test/models_general/Delta",
            group_path="./tags_classifier_library/predict/tests/models_test/models_general",
        ),
        ModelInfo(
            name="Echo",
            group="models_general",
            path="./tags_classifier_library/predict/tests/models_test/models_general/Echo",
            group_path="./tags_classifier_library/predict/tests/models_test/models_general",
        ),
        ModelInfo(
            name="Foxtrot",
            group="models_general",
            path="./tags_classifier_library/predict/tests/models_test/models_general/Foxtrot",
            group_path="./tags_classifier_library/predict/tests/models_test/models_general",
        ),
        ModelInfo(
            name="Golf",
            group="models_general",
            path="./tags_classifier_library/predict/tests/models_test/models_general/Golf",
            group_path="./tags_classifier_library/predict/tests/models_test/models_general",
        ),
        ModelInfo(
            name="Hotel",
            group="models_general",
            path="./tags_classifier_library/predict/tests/models_test/models_general/Hotel",
            group_path="./tags_classifier_library/predict/tests/models_test/models_general",
        ),
        ModelInfo(
            name="India",
            group="models_general",
            path="./tags_classifier_library/predict/tests/models_test/models_general/India",
            group_path="./tags_classifier_library/predict/tests/models_test/models_general",
        ),
        ModelInfo(
            name="Juliett",
            group="models_general",
            path="./tags_classifier_library/predict/tests/models_test/models_general/Juliett",
            group_path="./tags_classifier_library/predict/tests/models_test/models_general",
        ),
    ]
