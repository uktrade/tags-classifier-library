import re


def find_text(y, column_name="sentence"):
    text = y[column_name].lower()
    text_list = re.split(r"\W+", text)
    return text, text_list
