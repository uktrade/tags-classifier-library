from tags_classifier_library.preprocess import preprocess


def preprocess_for_prediction(fb_all):
    df = preprocess(fb_all)
    df = fb_all[["id", "sentence"]]
    df = df.dropna()
    return df
