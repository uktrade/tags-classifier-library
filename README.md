# Tags Classifier Library

A library for classifying text that helps training prediction models and run prediction on an arbitrary text.

## Local development

Create a new local environment using for example `pyenv`.
Install development dependencies using `pip install -r requirements-dev.txt`.
Make changes to the library.
Run `pip install .` - this will install library in your current environment.

### Running tests

To run tests use `make pytest`.

## Prediction

To run prediction on a given sentence, the prediction models need to be inspected first.

```
    models_info = inspect_model(path_to_models, ["models_group1", ...])
```

This will return an information about all the models in the given path.

Then a data source needs to be created:

```
    rows = [
        {
            "id": 1,
            "text": "lorem ipsum...",
        },
        ...
    }

    df = fetch_prediction_data(rows, "id", "text")
```

This will return a Pandas data frame with normalised data for prediction function.

Finally, a prediction can be peformed with:

```
    prediction = predict(
        X_to_predict=df[["id", "sentence"]],
        tokenizer=None,
        models_info=models_info,
    )
```

The function should return a data frame with predictions with following columns:


| Column | Description | Example |
| -------- | ------------- | --------- |
| id | id of the row | 1
| sentence | sentence a prediction was run against | Lorem ipsum... |
| prediction | List of predicted tags | ["Alpha", "Beta", "Charlie", ...] |
| prediction_prob | Probabilities for each tag | {"Alpha": 0.39499, "Beta": 0.12415, "Charlie": 0.1102, ...} |

#### Predictions for Data Hub

The model for Data Hub has to contain two model groups - `models_general` and `models_covid`.

The function to run prediction requires parameters that can be obtained in the same way as in the previous example.

```
    prediction = predict_data_hub_tags(df, models_info, covid_tag="Covid-19")
```

The prediction will be split into two parts - first it will run for models from `models_general` and then if resulting prediction contains `covid_tag`, then the second prediction will run from `models_covid` and the results will be merged.
The function will then select top 5 tags for each row of source data and return in the following format:

| Column | Description | Example |
| -------- | ------------- | --------- |
| id | id of the row | 1 |
| sentence | sentence a prediction was run against | Lorem ipsum... |
| prediction | List of predicted tags | ["Alpha", "Beta", "Charlie", ...] |
| tag_1 | first tag from the top 5 | Alpha |
| probability_score_tag_1 | probability score of the first tag | 0.39499 |
| tag_2 | second tag | Beta |
| probability_score_tag_2 | second probability score | 0.12415 |
| tag_X | X tag | X tag |
| probability_score_tag_X | X probability score | a number |
...

## Training

To train a model, a training file has to be provided.

The training file should be in a `.csv` format and should contain following columns:

| Column | Description | Example |
| -------- | ------------- | --------- |
| id | id of the row | 1 |
| sentence | A sentence to train from | "Lorem ipsum..." |
| tags | A list of tags that the `sentence` can be categorised with  | Trade,Tax |

Then the training file should be loaded into data frames:

```
        df = pd.read_csv(path_to_training_file)
```        

The resulting data frames need to be preprocessed and a list of tags extracted:

```
        df, my_tags = preprocess_for_training(df, tags_limit=2)
```
`tags_limit` describes how many times a tag needs to occur in the training file to be considered.

Finally, we can run the training:

```
        metric_df_general, metric_df_covid = build_models_pipeline(
            df, models_path, tags_general=my_tags, tags_covid=[], overwrite=False
        )
```

The `overwrite` parameter can be set to `True`, to overwrite the previous model.

The training function will train separately for general and Covid and that should result in two models being produced, if both general and covid tags are present in the single training file.

The resulting models can now be used for prediction.

### Coding style

The library is using `black` to maintain consistent coding style, `flake8` for linting and `isort` to sort imports.
You can check your code by running `make checks`.

### Update dependencies

Add library dependencies to `requirements.in` and development dependencies to `requirements-dev.in`.

Update dependencies using `make compile_all_requirements` and then install from updated `requirements*.txt`.
