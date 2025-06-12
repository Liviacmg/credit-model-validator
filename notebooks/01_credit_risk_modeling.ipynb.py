# Modelagem de Probability of Default (PD)
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit

# Preparação de dados
features = ['income', 'credit_score', 'debt_to_income']
target = 'default'

# Validação cruzada temporal
tscv = TimeSeriesSplit(n_splits=5)
for train_index, test_index in tscv.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Treinamento do modelo
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train[features], y_train)

    # Cálculo de PD
    y_pred_proba = model.predict_proba(X_test[features])[:, 1]
    X_test['PD'] = y_pred_proba