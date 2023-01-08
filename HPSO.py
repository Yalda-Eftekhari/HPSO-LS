import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


if __name__ == '__main__':
    dataset = pd.read_csv('./data/wine.csv',dtype=float)

    x = dataset.iloc[:, 0:13].values
    y = dataset.iloc[:, 13].values

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.4, shuffle=True, random_state=0)
    feature_scaler = StandardScaler()
    X_train = feature_scaler.fit_transform(X_train)
    X_test = feature_scaler.transform(X_test)
    FEATURES = len(X_train[0][:])

