from sklearn.metrics import classification_report, roc_auc_score
import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay

def evaluate_model(model, X_test, y_test):

    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:,1]

    print(classification_report(y_test, preds))

    auc = roc_auc_score(y_test, probs)
    print("ROC AUC:", auc)

    RocCurveDisplay.from_predictions(y_test, probs)
    plt.savefig("outputs/figures/roc_curve.png")

    with open("outputs/reports/metrics.txt","w") as f:
        f.write(classification_report(y_test,preds))
