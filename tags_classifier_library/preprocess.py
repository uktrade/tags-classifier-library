import re

from tags_classifier_library.constants import (COLUMN_RELABEL_MAP,
                                               COLUMN_RENAME_MAP, TAG_REMOVED,
                                               TAG_REPLACE_MAP)
from tags_classifier_library.settings import MIN_SENTENCE_LENGTH


def decontracted(phrase):
    phrase = re.sub(r"won't", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)
    phrase = re.sub(r"coronavirus", "covid", phrase)
    phrase = re.sub(r"corona virus", "covid", phrase)
    phrase = re.sub(r"https?:.*?(?=$|\s)", "", phrase, flags=re.MULTILINE)

    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    return phrase


def exclude_notes(fb_all):
    excluded_list = [
        "see notes above",
        "see email above",
        "see notes box above",
        'see "as above"',
        "none",
        "feedback as above",
        '"As above"',
        "see notes",
        "see email in notes",
        "covid-19",
        "covid 19",
        "covis 19",
        "refer to above notes",
        "see email details above",
        "see email detail above",
        "included in notes above",
        "please see above",
        "cbils",
        "feedback in above notes",
        "see interaction notes",
        "",
        "no additional notes",
        "refer to above notes",
        "please see the notes above",
    ]
    excluded_list_dot = [i + "." for i in excluded_list]
    excluded_list.extend(excluded_list_dot)
    fb_all = fb_all[fb_all["sentence"].str.len() > 25]
    fb_all = fb_all[~fb_all["sentence"].isin(excluded_list)]
    fb_all = fb_all[~fb_all["sentence"].str.lower().str.startswith("file:///c:/users/")]
    fb_all = fb_all[~fb_all["sentence"].str.lower().str.startswith("see detail above")]
    fb_all = fb_all[~fb_all["sentence"].str.lower().str.startswith("detail above")]
    fb_all = fb_all[~fb_all["sentence"].str.lower().str.startswith("see above")]

    return fb_all


def relabel(source_df, relabel_map=None):
    if not relabel_map:
        relabel_map = COLUMN_RELABEL_MAP

    df = source_df.copy()

    for transform in relabel_map:
        if all(column in df.columns for column in transform["columns"]):
            df[transform["columns"][0]] = df.apply(transform["transform"], axis=1)

    return df


def clean_tag(source_df, replace_map=None, removed_tags=None):
    if not replace_map:
        replace_map = TAG_REPLACE_MAP
    if not removed_tags:
        removed_tags = TAG_REMOVED

    df = source_df.copy()

    replace_map = {k.title(): v.title() for k, v in replace_map.items()}

    removed_tags = [i.lower() for i in removed_tags]
    df["tags"] = df["tags"].apply(
        lambda x: [
            replace_map.get(i.strip().title(), i.strip().title())
            for i in x.split(",")
            if i.lower() not in removed_tags
        ]
    )
    df["tags"] = df["tags"].apply(lambda x: ",".join(x))
    return df


def preprocess(fb_all, column_rename_map=None, min_sentence_length=MIN_SENTENCE_LENGTH):
    if not column_rename_map:
        column_rename_map = COLUMN_RENAME_MAP

    fb_all = fb_all.rename(columns=column_rename_map)
    fb_all = fb_all.dropna(subset=["sentence"])

    fb_all = exclude_notes(fb_all)
    fb_all["sentence"] = fb_all["sentence"].apply(lambda x: decontracted(x))
    fb_all["length"] = fb_all["sentence"].str.len()
    fb_all = fb_all[fb_all["length"] > min_sentence_length]
    fb_all = fb_all.reset_index(drop=True)
    return fb_all
