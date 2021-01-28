"""
Tags classifier library
"""

from setuptools import find_packages, setup

setup(
    name="tags-classifier-library",
    version="0.1.0",
    url="https://github.com/uktrade/tags-classifier-library",
    license="MIT",
    author="Department for International Trade",
    description="A library for classifying text.",
    packages=find_packages(exclude=["tests.*", "tests"]),
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    include_package_data=True,
    install_requires=[
        "tensorflow==2.3.1"
    ],
    extras_require={
        "test": [
            "pytest==6.1.0",
            "pytest-cov==2.10.1",
            "flake8==3.8.3",
            "wheel>=0.31.0,<1.0.0",
            "setuptools>=38.6.0,<39.0.0",
            "codecov",
            "twine",
        ]
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Environment :: Web Environment",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: Implementation :: PyPy",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
)
