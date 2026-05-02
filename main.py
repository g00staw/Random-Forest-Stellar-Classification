import numpy as np
import pandas as pd

from sklearn import metrics, datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

dataset = pd.read_csv('data.csv')
