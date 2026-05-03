import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

print("Reading data...")
dataset = pd.read_csv('data/star_classification.csv')

dataset = dataset.drop(columns=[
    'obj_ID', 'run_ID', 'rerun_ID', 'cam_col',
    'field_ID', 'spec_obj_ID', 'plate', 'MJD', 'fiber_ID'
])

dataset['u_g'] = dataset['u'] - dataset['g']
dataset['g_r'] = dataset['g'] - dataset['r']
dataset['r_i'] = dataset['r'] - dataset['i']
dataset['i_z'] = dataset['i'] - dataset['z']

le = LabelEncoder()
dataset['class'] = le.fit_transform(dataset['class'])

X = dataset.drop('class', axis=1)
y = dataset['class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Start GridSearch for best parameters...")

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 15, 25],
    'min_samples_split': [2, 5],
}

rf_base = RandomForestClassifier(random_state=42)

grid_search = GridSearchCV(estimator=rf_base,
                           param_grid=param_grid,
                           cv=3,
                           n_jobs=-1,
                           verbose=2)

grid_search.fit(X_train, y_train)

print("\nBest parameters: ")
print(grid_search.best_params_)

best_model = grid_search.best_estimator_

print("\nResult on test set: ")
y_pred = best_model.predict(X_test)

print(classification_report(y_test, y_pred, target_names=le.classes_))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))