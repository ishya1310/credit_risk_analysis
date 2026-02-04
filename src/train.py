
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
def train_models(X_scaled, X_unscaled, y):
    Xs_train, Xs_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, stratify=y, random_state=42
    )

    Xu_train, Xu_test, _, _ = train_test_split(
        X_unscaled, y, test_size=0.2, stratify=y, random_state=42
    )

    lr = LogisticRegression(
        class_weight='balanced',
        max_iter=1000,
        random_state=42,
        solver='lbfgs'
    )
    lr.fit(Xs_train, y_train)

    rf = RandomForestClassifier(
        n_estimators=200,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    rf.fit(Xu_train, y_train)

    return lr, rf, Xs_test, Xu_test, y_test
