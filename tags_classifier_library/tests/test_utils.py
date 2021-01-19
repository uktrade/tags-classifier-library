import pandas as pd

from tags_classifier_library.utils import find_text


def test_find_text():
    data = [{"sentence": "lorem ipsum"}]

    df = pd.DataFrame(data)
    text, words = find_text(df.iloc[0])
    assert "lorem ipsum" == text
    assert ["lorem", "ipsum"] == words
