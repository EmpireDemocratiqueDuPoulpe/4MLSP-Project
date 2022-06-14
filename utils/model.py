# #################################################################################################################### #
#       model.py                                                                                                       #
#           Model processing.                                                                                          #
# #################################################################################################################### #

from colorama import Style, Fore
import numpy
import sklearn
import scipy.stats
import warnings


def process_model(model, x_train, y_train, x_test, y_test, verbose=False):
    model.fit(x_train, y_train)
    prediction = model.predict(x_test)

    if verbose:
        print("NOT IMPLEMENTED")

    acc_score, train_score, test_score = get_score(
        model,
        x_train=x_train, y_train=y_train,
        x_test=x_test, y_test=y_test,
        prediction=prediction
    )
    return model, {"accuracy": acc_score, "train": train_score, "test": test_score}


def process_regression_model(model, x_train, y_train, x_test, y_test, verbose=False):
    model.fit(x_train, y_train)
    prediction = model.predict(x_test)

    train_score, test_score, mae, rmse, mape, statistic, pvalue = get_regression_score(
        model,
        x_train=x_train, y_train=y_train,
        x_test=x_test, y_test=y_test,
        prediction=prediction
    )
    return model, {
        "train": train_score, "test": test_score, "mae": mae, "rmse": rmse, "mape": mape,
        "statistic": statistic, "pvalue": pvalue
    }


def get_score(model, x_train, y_train, x_test, y_test, prediction=None):
    if prediction is None:
        prediction = model.predict(x_test)

    accuracy_score = round(sklearn.metrics.accuracy_score(y_test, prediction), 2)
    train_score = round(model.score(x_train, y_train) * 100, 2)
    test_score = round(model.score(x_test, y_test) * 100, 2)

    return accuracy_score, train_score, test_score


def get_regression_score(model, x_train, y_train, x_test, y_test, prediction=None):
    if prediction is None:
        prediction = model.predict(x_test)

    train_score = round(model.score(x_train, y_train) * 100, 2)
    test_score = round(model.score(x_test, y_test) * 100, 2)
    mae = round(sklearn.metrics.mean_absolute_error(y_true=y_test, y_pred=prediction), 3)
    rmse = round(numpy.sqrt(sklearn.metrics.mean_squared_error(y_true=y_test, y_pred=prediction)), 3)
    mape = round(sklearn.metrics.mean_absolute_percentage_error(y_true=y_test, y_pred=prediction) * 100, 3)

    with warnings.catch_warnings():
        warnings.filterwarnings("error")

        try:
            statistic, pvalue = scipy.stats.shapiro(prediction - y_test)
        except UserWarning as warn:
            print(f"{Fore.YELLOW}Warning: {warn}")
            warnings.filterwarnings("ignore")
            statistic, pvalue = scipy.stats.shapiro(prediction - y_test)

    return train_score, test_score, mae, rmse, mape, statistic, pvalue


def print_scores(scores):
    if all(key in scores for key in ["mae", "rmse", "mape", "pvalue", "statistic"]):
        print((
            f"Best achieved accuracy:"
            f" (train: {Fore.LIGHTGREEN_EX}{scores['train']}%{Fore.RESET}"
            f", test: {Fore.LIGHTGREEN_EX}{scores['test']}%{Fore.RESET})\n"

            f"Regression model scores:"
            f" (MAE: {Fore.LIGHTGREEN_EX}{scores['mae']}{Fore.RESET}"
            f", RMSE: {Fore.LIGHTGREEN_EX}{scores['rmse']}{Fore.RESET}"
            f", MAPE: {Fore.LIGHTGREEN_EX}{scores['mape']}%{Fore.RESET})\n"

            f"Normality test:"
            f" (pvalue: {Fore.LIGHTGREEN_EX}{round(scores['pvalue'], 1)}{Fore.RESET}"
            f", statistic: {Fore.LIGHTGREEN_EX}{round(scores['statistic'], 3)}{Fore.RESET})"
        ))
    else:
        print((
            f"Best achieved accuracy: {Fore.LIGHTGREEN_EX}{scores['accuracy']}"
            f"{Fore.WHITE}{Style.DIM} (train: {scores['train']}%"
            f", test: {scores['test']}%){Style.RESET_ALL}"
        ))


def best_model(orig_model, search, x_train, y_train, x_test, y_test, scores=None):
    if scores is None:
        train_score_orig = round(orig_model.score(x_train, y_train) * 100, 2)
        test_score_orig = round(orig_model.score(x_test, y_test) * 100, 2)
    else:
        train_score_orig = scores["train"]
        test_score_orig = scores["test"]

    search.fit(x_train, y_train)
    new_model = search.best_estimator_

    print(f"Best params found for model: {Fore.LIGHTGREEN_EX}{search.best_params_}")

    new_train_score = round(new_model.score(x_train, y_train) * 100, 2)
    new_test_score = round(new_model.score(x_test, y_test) * 100, 2)
    delta_train = round((new_train_score - train_score_orig) / 100, 2)
    delta_test = round((new_test_score - test_score_orig) / 100, 2)

    print((
        f"New model score:\n"
        f"\t{Fore.WHITE}{Style.DIM}∟ train → {new_train_score}% ({delta_to_str(delta_train)}{Fore.WHITE}{Style.DIM})\n"
        f"\t{Fore.WHITE}{Style.DIM}∟ test → {new_test_score}% ({delta_to_str(delta_test)}{Fore.WHITE}{Style.DIM})"
    ))

    if new_train_score >= train_score_orig and new_test_score >= test_score_orig:
        return new_model
    elif new_train_score < train_score_orig and new_test_score < test_score_orig:
        return orig_model
    else:
        if new_train_score < new_test_score:
            resolve = {"lower_score": new_train_score, "prev_score": train_score_orig}
        else:
            resolve = {"lower_score": new_test_score, "prev_score": test_score_orig}

        return new_model if ((resolve["lower_score"] - resolve["prev_score"]) >= 0) else orig_model


def delta_to_str(delta):
    color = Fore.GREEN
    sign = "+"

    if delta < 0:
        color = Fore.RED
        sign = "-"
        delta = abs(delta)

    return f"{color}{sign}{delta}%{Fore.RESET}"
