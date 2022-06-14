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

    scores = get_score(
        model,
        x_train=x_train, y_train=y_train,
        x_test=x_test, y_test=y_test,
        prediction=prediction
    )
    return model, scores


def process_regression_model(model, x_train, y_train, x_test, y_test, verbose=False):
    model.fit(x_train, y_train)
    prediction = model.predict(x_test)

    scores = get_regression_score(
        model,
        x_train=x_train, y_train=y_train,
        x_test=x_test, y_test=y_test,
        prediction=prediction
    )
    return model, scores


def get_score(model, x_train, y_train, x_test, y_test, prediction=None):
    if prediction is None:
        prediction = model.predict(x_test)

    accuracy_score = round(sklearn.metrics.accuracy_score(y_test, prediction), 2)
    train_score = round(model.score(x_train, y_train) * 100, 2)
    test_score = round(model.score(x_test, y_test) * 100, 2)

    return {"accuracy": accuracy_score, "train": train_score, "test": test_score}


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
            print(f"{Fore.YELLOW}Attention: {warn}")
            warnings.filterwarnings("ignore")
            statistic, pvalue = scipy.stats.shapiro(prediction - y_test)

    return {
        "train": train_score, "test": test_score, "mae": mae, "rmse": rmse, "mape": mape,
        "statistic": statistic, "pvalue": pvalue
    }


def print_scores(scores):
    if all(key in scores for key in ["mae", "rmse", "mape", "pvalue", "statistic"]):
        print((
            f"Précision du modèle :"
            f" (entraînement: {Fore.LIGHTGREEN_EX}{scores['train']}%{Fore.RESET}"
            f", test: {Fore.LIGHTGREEN_EX}{scores['test']}%{Fore.RESET})\n"

            f"Score du modèle de régression :"
            f" (MAE : {Fore.LIGHTGREEN_EX}{scores['mae']}{Fore.RESET}"
            f", RMSE : {Fore.LIGHTGREEN_EX}{scores['rmse']}{Fore.RESET}"
            f", MAPE : {Fore.LIGHTGREEN_EX}{scores['mape']}%{Fore.RESET})\n"

            f"Test de normalité :"
            f" (pvalue : {Fore.LIGHTGREEN_EX}{round(scores['pvalue'], 1)}{Fore.RESET}"
            f", statistic : {Fore.LIGHTGREEN_EX}{round(scores['statistic'], 3)}{Fore.RESET})"
        ))
    else:
        print((
            f"Précision du modèle : {Fore.LIGHTGREEN_EX}{scores['accuracy']}"
            f"{Fore.WHITE}{Style.DIM} (entraînement: {scores['train']}%"
            f", test: {scores['test']}%){Style.RESET_ALL}"
        ))


def best_model(orig_model, is_regression, search, x_train, y_train, x_test, y_test, scores=None):
    if scores is None:
        if is_regression:
            scores = get_regression_score(orig_model, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)
        else:
            scores = get_score(orig_model, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)

    search.fit(x_train, y_train)
    new_model = search.best_estimator_

    if is_regression:
        new_scores = get_regression_score(new_model, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)
    else:
        new_scores = get_score(new_model, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)

    print(f"Meilleurs paramètres trouvés pour le modèle : {Fore.LIGHTGREEN_EX}{search.best_params_}")

    delta_train = round(new_scores["train"] - scores["train"], 2)
    delta_test = round(new_scores["test"] - scores["test"], 2)

    print((
        f"Scores du nouveau modèle:\n"
        f"\t{Fore.WHITE}{Style.DIM}∟ entraînement → {new_scores['train']}% ({delta_to_str(delta_train)}{Fore.WHITE}{Style.DIM})\n"
        f"\t{Fore.WHITE}{Style.DIM}∟ test → {new_scores['test']}% ({delta_to_str(delta_test)}{Fore.WHITE}{Style.DIM})"
    ))

    if new_scores["train"] >= scores["train"] and new_scores["test"] >= scores["test"]:
        return new_model, new_scores
    elif new_scores["train"] < scores["train"] and new_scores["test"] < scores["test"]:
        return orig_model, scores
    else:
        if new_scores["train"] < new_scores["test"]:
            resolve = {"lower_score": new_scores['train'], "prev_score": scores['train']}
        else:
            resolve = {"lower_score": new_scores['test'], "prev_score": scores['test']}

        return (new_model, new_scores) if ((resolve["lower_score"] - resolve["prev_score"]) >= 0) else (orig_model, scores)


def delta_to_str(delta):
    color = Fore.GREEN
    sign = "+"

    if delta < 0:
        color = Fore.RED
        sign = "-"
        delta = abs(delta)

    return f"{color}{sign}{delta}%{Fore.RESET}"
