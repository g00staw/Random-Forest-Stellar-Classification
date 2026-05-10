from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

from read_data import load_data
from result_charts import conf_matrix, roc_auc

X, y, le = load_data()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

knn = KNeighborsClassifier()
param_grid = {
    'n_neighbors': [3, 5, 7, 9, 13, 15],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan']
}

grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy')

grid_search.fit(X_train, y_train)

# Add some charts for cross - val results etc. based on seminars materials

print("\nBest parameters: ")
print(grid_search.best_params_)

best_model = grid_search.best_estimator_

print("\nResult on test set: ")
y_pred = best_model.predict(X_test)

print(classification_report(y_test, y_pred, target_names=le.classes_))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))

conf_matrix(y_test, y_pred, le, output_prefix="knn_")
roc_auc(X_test, y_test, le, best_model, output_prefix="knn_")
