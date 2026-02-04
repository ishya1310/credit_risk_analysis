

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)


def evaluate_model(model, X_test, y_test):
    
    

    y_pred = model.predict(X_test)

    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("Accuracy:")
    print(accuracy_score(y_test, y_pred))
