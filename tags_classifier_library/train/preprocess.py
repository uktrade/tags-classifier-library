import logging

from tags_classifier_library.preprocess import clean_tag, preprocess, relabel

logger = logging.getLogger(__name__)


def preprocess_for_training(fb_all, tags_limit=200):
    df = preprocess(fb_all)
    fb_all = preprocess(fb_all, column_rename_map=None, min_sentence_length=5)
    fb_all = fb_all[fb_all["tags"] != "Not Specified"]

    fb_all = fb_all[["id", "sentence", "tags"]]
    fb_all = fb_all.dropna(subset=["sentence", "tags"])
    fb_all["tags"] = fb_all["tags"].apply(
        lambda x: x.replace(";", ",").replace("\u200b", "").replace("â€‹", "").replace("Â", "")
    )
    fb_all["tags"] = fb_all["tags"].apply(
        lambda x: x + ",covid-19".title() if "covid" in x.lower() else x
    )

    fb_all = clean_tag(fb_all)

    fb_tag = fb_all["tags"].str.strip().str.get_dummies(sep=",")

    tags_count = fb_tag.sum().sort_values(ascending=False)
    fb_tag.columns = [i.strip().title() for i in fb_tag.columns]
    df = fb_all.merge(fb_tag, left_index=True, right_index=True)
    df = relabel(df)

    tags_200 = list(tags_count[tags_count > tags_limit].index)
    tags_200 = [
        i
        for i in tags_200
        if i.lower() not in ["general", "not specified", "other", "others", "covid-19 general"]
    ]
    logger.info(f"tags counts: {tags_count}")
    logger.info(f"ordered tags counts: {tags_count.sort_index()}")
    logger.info(f"tags with more than {tags_limit} counts: {tags_200}")
    logger.info(f"train model for these tags (more than {tags_limit} samples): {tags_200}")

    select_columns = ["id", "sentence"]
    select_columns.extend(tags_200)
    if "bert_vec_cleaned" in fb_all.columns:
        select_columns.append("bert_vec_cleaned")

    df = df[select_columns]
    return df, tags_200
