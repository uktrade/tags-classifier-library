# Tags Classifier Library

A library for classifying text that helps training prediction models and run prediction on an arbitrary text.

## Local development

Create a new local environment using for example `pyenv`.
Install development dependencies using `pip install -r requirements-dev.txt`.
Make changes to the library.
Run `pip install .` - this will install library in your current environment.

### Running tests

To run tests use `make pytest`.

### Coding style

The library is using `black` to maintain consistent coding style, `flake8` for linting and `isort` to sort imports.
You can check your code by running `make checks`.

### Update dependencies

Add library dependencies to `requirements.in` and development dependencies to `requirements-dev.in`.

Update dependencies using `make compile_all_requirements` and then install from updated `requirements*.txt`.
