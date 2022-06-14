# #################################################################################################################### #
#       metrics.py                                                                                                     #
#           Where weird numbers become sweet charts.                                                                   #
# #################################################################################################################### #

from matplotlib import pyplot
import sklearn


def roc_curve(model, x_test, y_test, name="", prediction=None):
    if prediction is None:
        prediction = model.predict(x_test)

    fpr, tpr, threshold = sklearn.metrics.roc_curve(y_test, prediction)
    roc_auc = sklearn.metrics.auc(fpr, tpr)

    pyplot.title(f"Fonction d'efficacité du récepteur (ROC){f' - {name}' if name else ''}")
    pyplot.plot(fpr, tpr, "b", label="AUC = %0.2f" % roc_auc)
    pyplot.legend(loc="lower right")

    pyplot.plot([0, 1], [0, 1], "r--")
    pyplot.xlim([0, 1])
    pyplot.ylim([0, 1])
    pyplot.xlabel("Taux de faux positifs (FPR)")
    pyplot.ylabel("Taux de vrais positifs (TPR)")

    pyplot.show()


def precision_recall_curve(model, x_test, y_test, name="", prediction=None):
    if prediction is not None:
        prd = sklearn.metrics.PrecisionRecallDisplay.from_predictions(y_test, prediction, name=name)
    else:
        prd = sklearn.metrics.PrecisionRecallDisplay.from_estimator(model, x_test, y_test, name=name)

    prd.ax_.set_title("Courbe de précision-rappel à 2 classes")
    pyplot.show()
