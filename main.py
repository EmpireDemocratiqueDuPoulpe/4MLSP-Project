import sys
from timeit import default_timer as timer
from datetime import datetime, timedelta
import colorama
from colorama import Style, Fore
import pandas
import numpy
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
import utils


class StdOutCapture(list):
    def __enter__(self):
        self._orig_stdout = sys.stdout
        sys.stdout = ShellClock(self._print)

        return self

    def _print(self, text):
        self._orig_stdout.write(text)

    def __exit__(self, *args):
        sys.stdout = self._orig_stdout


class ShellClock(object):
    def __init__(self, stdout_write):
        self._orig_write = stdout_write

    def write(self, text):
        if text.strip():
            self._orig_write(f"{Fore.GREEN}{datetime.now().strftime('%H:%M:%S')} >>>{Fore.RESET} {text}\n")


def main():
    program_start = timer()

    # Read the CSV
    data = pandas.read_csv("./data/MP.csv", delimiter=";", dtype={
        "age": int,
        "job": "category",
        "marital": "category",
        "education": "category",
        "default": "category",
        "housing": "category",
        "loan": "category",
        "contact": "category",
        "month": "category",
        "day_of_week": "category",
        "duration": int,
        "campaign": int,
        "pdays": int,
        "previous": int,
        "emp.var.rate": numpy.float64,
        "cons.conf.idx": numpy.float64,
        "cons.price.idx": numpy.float64,
        "euribor3m": numpy.float64,
        "nr.employed": numpy.float64,
        "y": "category"
    })

    # First look
    utils.print.title("First look")
    utils.dataframe.first_look(data)

    # Missing values
    utils.print.title("Missing values")
    utils.dataframe.missing_values(data, keep_zeros=False)

    # Splitting dataset
    utils.print.title("Splitting dataset")
    model_data = data.drop(["duration"], axis=1)
    x_train, x_test, y_train, y_test = utils.dataframe.split_train_test(model_data, y_label="y")
    print(f"Train data: {Fore.LIGHTGREEN_EX}{x_train.shape}")
    print(f"Test data: {Fore.LIGHTGREEN_EX}{x_test.shape}")

    # Transform numeric and categorical values
    utils.print.title("Transform numeric and categorical values")
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler())
    ])
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])
    # ranking_transformer = Pipeline(steps=[
    #    ("imputer", SimpleImputer(strategy="constant")),
    #    ("encoder", OrdinalEncoder())
    # ])

    numeric_features = [
        "age", "campaign", "pdays", "previous", "emp.var.rate", "cons.price.idx", "cons.conf.idx", "euribor3m",
        "nr.employed"
    ]
    categorical_features = [
        "job", "marital", "education", "default", "housing", "loan", "contact", "month", "day_of_week", "poutcome"
    ]
    # ranking_features = ["education"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, numeric_features),
            ("categorical", categorical_transformer, categorical_features),
            # ("ranking", ranking_transformer, ranking_features)
        ]
    )

    # Get models
    utils.print.title("Get best model")
    models = {
        "k_neighbors_classifier": {
            "model": KNeighborsClassifier(),
            "is_regression": False,
            "hyper_params": {
                "n_neighbors": numpy.arange(1, 20),
                "metric": ["euclidean", "manhattan", "minkowski"]
            }
        },
        "k_neighbors_regressor": {
            "model": KNeighborsRegressor(),
            "is_regression": True,
            "hyper_params": {
                "n_neighbors": numpy.arange(1, 20),
                "metric": ["euclidean", "manhattan", "minkowski"]
            }
        },
        "logistic_regression": {
            "model": LogisticRegression(),
            "is_regression": True,
            "hyper_params": {
                "solver": ["sag", "saga", "newton-cg", "lbfgs"],
                "max_iter": numpy.arange(100, 200)
            }
        },
        "linear_regression": {
            "model": LinearRegression(),
            "is_regression": True,
            "hyper_params": None
        },
        "linear_discriminant_analysis": {
            "model": LinearDiscriminantAnalysis(),
            "is_regression": False,
            "hyper_params": {
                "solver": ["svd", "lsqr", "eigen"]
            }
        },
        "tree_classifier": {
            "model": DecisionTreeClassifier(),
            "is_regression": False,
            "hyper_params": {
                "criterion": ["gini", "entropy"],
                "min_samples_split": numpy.arange(2, 3)
            }
        },
        "tree_regressor": {
            "model": DecisionTreeRegressor(),
            "is_regression": True,
            "hyper_params": {
                "criterion": ["squared_error", "friedman_mse", "absolute_error", "poisson"],
                "min_samples_split": numpy.arange(2, 3)
            }
        },
        "svc": {
            "model": SVC(),
            "is_regression": False,
            "hyper_params": {
                "kernel": ["linear", "poly", "rbf", "sigmoid", "precomputed"]
            }
        },
        "linear_svc": {
            "model": LinearSVC(),
            "is_regression": False,
            "hyper_params": {
                "loss": ["hinge", "squared_hinge"]
            }
        },
    }

    with StdOutCapture() as output:
        for model_key in models:
            # Start model processing
            model_start = timer()
            model_infos = models[model_key]

            # Get model
            print(f"{Fore.LIGHTBLUE_EX}Processing the dataset with \"{model_key}\" model...")
            pipeline = Pipeline(steps=[
                ("preprocessor", preprocessor),
                (model_key, model_infos["model"])
            ])

            # TEMPORARY #
            if model_infos["is_regression"]:
                continue
            #############

            if model_infos["is_regression"]:
                model, scores = utils.model.process_regression_model(
                    pipeline,
                    x_train=x_train, y_train=y_train,
                    x_test=x_test, y_test=y_test,
                    verbose=True
                )
            else:
                model, scores = utils.model.process_model(
                    pipeline,
                    x_train=x_train, y_train=y_train,
                    x_test=x_test, y_test=y_test,
                    verbose=True
                )

            utils.model.print_scores(scores)

            if model_infos["hyper_params"] is not None:
                print(f"Finding the best params using Grid Search CV")
                grid = GridSearchCV(
                    pipeline,
                    {f"{model_key}__{k}": v for k, v in model_infos["hyper_params"].items()},
                    verbose=2
                )
                model, scores = utils.model.best_model(
                    model, is_regression=model_infos["is_regression"], search=grid,
                    x_train=x_train, y_train=y_train,
                    x_test=x_test, y_test=y_test,
                    scores=scores
                )

            model_end = timer()
            model_elapsed_time = timedelta(seconds=model_end - model_start)
            print(f"\n{Fore.LIGHTBLUE_EX}Finished in {model_elapsed_time}.\n")

    # Program end
    program_end = timer()
    program_elapsed_time = timedelta(seconds=program_end - program_start)
    print(f"\n{Fore.GREEN}Successful processing of \"MP.csv\" in {program_elapsed_time}.")


if __name__ == "__main__":
    colorama.init(autoreset=True)
    main()
