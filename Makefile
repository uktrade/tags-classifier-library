build: test_requirements test

clean:
	-find . -type f -name "*.pyc" -delete
	-find . -type d -name "__pycache__" -delete

test_requirements:
	pip install -e .[test]

pytest:
	pytest . --cov=. --cov-config=.coveragerc $(pytest_args)

flake8:
	flake8 .

checks: flake8
	isort $(PWD) --check
	black $(PWD) --check --verbose

test: checks pytest

compile_requirements:
	pip-compile --output-file=requirements.txt requirements.in

compile_dev_requirements:
	pip-compile --output-file=requirements-dev.txt requirements-dev.in

compile_all_requirements: compile_requirements compile_dev_requirements

.PHONY: build clean test_requirements flake8 pytest test checks
