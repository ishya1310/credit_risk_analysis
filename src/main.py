

from data_loading import load_application_data, load_credit_data
from label_creation import create_type_of_client
from preprocessing import preprocess
from train import train_models
from evaluate import evaluate_model


APP_PATH = "data/application_record.csv"
CREDIT_PATH = "data/credit_record.csv"


def main():
    print("DATASETS")
    application_df = load_application_data(APP_PATH)
    credit_df = load_credit_data(CREDIT_PATH)

    print("Creation of target label")
    labels_df = create_type_of_client(credit_df)

    print("Preprocessing of data")
    X_scaled, X_unscaled, y = preprocess(application_df, labels_df)

    print("Training models")
    lr_model, rf_model, Xs_test, Xu_test, y_test = train_models(
        X_scaled, X_unscaled, y
    )

    print("\n Logistic Regression (Baseline model)")
    evaluate_model(lr_model, Xs_test, y_test)



if __name__ == "__main__":
    main()
