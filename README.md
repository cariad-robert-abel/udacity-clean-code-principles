# Predict Customer Churn

This is the project for the _Machine Learning Engineer_ module from Udacity that I complete as part
of the _Clean Code Principles_ course on XX&nbsp;Mar 2026.

## Project Description

The original project, _Predict Customer Churn_, is part of the ML DevOps Engineer Nanodegree and
its purpose here is that I clean it up and adhere to clean code practices.

> [!NOTE]
> The original project has been updated from scikit-learn v0.22.x to v1.8.x, which means the models
> benefit from improvements within that library in the meantime.
> Unfortunately, the original Logistic Regression Classifier (LRC) performed much more poorly albeit
> having been trained much more extensively due to a bug in scikit-learn v0.22.x (`max_iter`).
> The original code used the LRC's poor performance to showcase that ensemble classifiers such as
> Random Forest Classifier (RFC) can work much better.  
> I included the original LRC model as well as the updated model and adjusted the plots to show what
> they originally looked like in order to preserve this intent.

## Files and data description

- **churn_library.py**:  
  Refactored library code from original customer attrition (churn) notebook.
- **churn_notebook.ipynb**:  
  The original customer attrition (churn) notebook, which is supposed to be improved using clean
  code techniques.
- **model/logistic_model.orig.pkl**:  
  Best trained Logistic Regression Classifier Model from scikit-learn v0.22.x extended with feature
  names (`feature_names_in_`) and saved in scikit-learn v1.8.x format.
- **model/logistic_model.pkl**:  
  Best trained Logistic Regression Classifier Model from scikit-learn v1.8.x.
- **model/rfc_model.pkl**:  
  Best trained Random Forest Classifier Model from scikit-learn v1.8.x.
- **tests/test_churn_library.py**:  
  Unit tests for customer attrition (churn) library.

## Running Files

Install run-time dependencies using `pip install -e .`.
Install group `dev` when using Jupyter to develop notebook code.
Install group `test` when running unit tests is required.

Install everything:

    pip install -e .[dev,test]

Run the main executable, which performs exploratory data analysis, trains models, and outputs result
pictures using:

    python churn_library.py

Run unit tests using:

    pytest

## Packaging

While this project isn't packaged per se, I'm using the modern Python package and dependency manager
[PDM](https://pdm-project.org) to manage dependencies.
I recommend installing PDM via [pipx](https://pipx.pypa.io) into its own dedicated virtual environment.

## License

Original files Copyright 2012–2020 Udacity, Inc.
My additions to documentation and code are [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).
See [LICENSE-Udacity](LICENSE-Udacity) resp. [LICENSE](LICENSE).
