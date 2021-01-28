import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from tags_classifier_library.preprocess import (
    clean_tag,
    decontracted,
    exclude_notes,
    preprocess,
    relabel,
)
from tags_classifier_library.utils import find_text

TEST_COLUMN_RELABEL_MAP = [
    {
        "columns": ["Transition Period - General", "policy_issue_types"],
        "transform": lambda row: 1
        if row["policy_issue_types"] == '{"EU exit"}'
        else row["Transition Period - General"],
    },
    {
        "columns": ["Covid-19"],
        "transform": lambda row: 1
        if any(i in find_text(row)[1] for i in ["covid", "coronavirus"])
        else row["Covid-19"],
    },
]

TEST_TAG_REPLACE_MAP = {
    "Opportunities": "Opportunities",
    "Exports - other": "Exports",
    "Export": "Exports",
}

TEST_TAG_REMOVED = ["Reduced profit"]


@pytest.mark.parametrize(
    "test_input,expected",
    [
        ("they won't", "they will not"),
        ("they can't", "they can not"),
        ("coronavirus", "covid"),
        ("corona virus", "covid"),
        ("http://example", ""),
        ("don't", "do not"),
        ("we're", "we are"),
        ("'s", " is"),
        ("they'd", "they would"),
        ("they'll", "they will"),
        ("'t", " not"),
        ("'ve", " have"),
        ("'m", " am"),
    ],
)
def test_decontracted(test_input, expected):
    result = decontracted(test_input)
    assert expected == result


@pytest.mark.parametrize(
    "test_input,expected",
    [
        ("Please see email above.", None),
        (
            "Lorem ipsum. This is a longer note, please refer to above notes",
            "Lorem ipsum. This is a longer note, please refer to above notes",
        ),
    ],
)
def test_exclude_notes(test_input, expected):
    df = pd.DataFrame(data={"sentence": [test_input]})
    result = exclude_notes(df)
    if not expected:
        assert result.empty
    else:
        assert expected == result.at[0, "sentence"]


def test_relabel():
    data = [
        {
            "Transition Period - General": "",
            "policy_issue_types": '{"EU exit"}',
            "Covid-19": "",
            "sentence": "lorem ipsum",
        },
        {
            "Transition Period - General": "0",
            "policy_issue_types": "",
            "Covid-19": "",
            "sentence": "lorem ipsum covid",
        },
        {
            "Transition Period - General": "",
            "policy_issue_types": "",
            "Covid-19": "",
            "sentence": "lorem ipsum",
        },
        {
            "Transition Period - General": "0",
            "policy_issue_types": "",
            "Covid-19": "",
            "sentence": "lorem ipsum",
        },
    ]
    df = pd.DataFrame(data)
    result = relabel(df, relabel_map=TEST_COLUMN_RELABEL_MAP)

    assert 1 == result.at[0, "Transition Period - General"]
    assert 1 == result.at[1, "Covid-19"]
    assert "" == result.at[2, "Transition Period - General"]
    assert "" == result.at[3, "Covid-19"]


def test_clean_tag():
    data = [{"tags": "Opportunities,export,reduced profit"}]
    df = pd.DataFrame(data)
    result = clean_tag(df, replace_map=TEST_TAG_REPLACE_MAP, removed_tags=TEST_TAG_REMOVED)
    assert "Opportunities,Exports" == result.at[0, "tags"]


def test_preprocess():
    rename_map = {
        "content": "sentence",
        "category": "tags",
    }
    data = [
        # should be dropped
        {"content": "test", "category": "lorem ipsum"},
        # column should be renamed to "sentence" and carried over
        # category should be renamed to tags
        {"content": "lorem ipsum lorem ipsum lorem ipsum", "category": "lorem ipsum"},
        # they'll not should be replaced with they will not
        {"content": "lorem ipsum lorem ipsum lorem ipsum they'll not", "category": "lorem ipsum"},
        # should be dropped
        {"content": "see email details above.", "category": "lorem ipsum"},
    ]

    expected = [
        {
            "sentence": "lorem ipsum lorem ipsum lorem ipsum",
            "tags": "lorem ipsum",
            "length": 35,
        },
        {
            "sentence": "lorem ipsum lorem ipsum lorem ipsum they will not",
            "tags": "lorem ipsum",
            "length": 49,
        },
    ]

    df = pd.DataFrame(data)
    expected = pd.DataFrame(expected)

    result = preprocess(df, column_rename_map=rename_map)

    assert_frame_equal(expected, result)
