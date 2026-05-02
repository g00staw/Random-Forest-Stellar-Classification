import numpy as np
import pandas as pd

from sklearn import metrics, datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

dataset = pd.read_csv('data/star_classification.csv')

dataset.drop(columns=
             ['obj_ID',
              'run_ID',
              'rerun_ID',
              'cam_col',
              'field_ID',
              'spec_obj_ID',
              'plate',
              'MJD',
              'fiber_ID'], axis = 1)


dataset['u_g'] = dataset['u'] - dataset['g']
dataset['g_r'] = dataset['g'] - dataset['r']
dataset['r_i'] = dataset['r'] - dataset['i']
dataset['i_z'] = dataset['i'] - dataset['z']

le = LabelEncoder()

dataset = le.fit_transform(dataset)

X = dataset.drop('class', axis=1)
y = dataset['class']

model = RandomForestClassifier(random_state=42, n_jobs=-1, n_estimators=100)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)