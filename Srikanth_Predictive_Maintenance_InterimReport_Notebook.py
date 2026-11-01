import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as sts
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import RandomizedSearchCV
import os

#Used for google collab test run
#from google.colab import drive
#drive.mount('/content/drive')

#Used for google collab test run
#BASE_DIR = '/content/drive/MyDrive/Capstone_project_Predictive_Maintenance/'
#DATA_PATH =  f'{BASE_DIR}data/'
#ARTIFACTS_PATH =  f'{BASE_DIR}/artifacts/'
#MODEL_PATH =  f'{BASE_DIR}/model/'

#Used with Github actions
DATA_PATH = 'data/'
ARTIFACTS_PATH = 'artifacts/'
MODEL_PATH = 'model/'
df = pd.read_csv(f'{DATA_PATH}/engine_data.csv')

if not os.path.isdir(DATA_PATH):
  print(f"Data Directory does not exist {DATA_PATH}")
elif not os.path.exists(f'{DATA_PATH}/engine_data.csv'):
  print(f"Data csv does not exist {DATA_PATH}/engine_data.csv")

df = pd.read_csv(f'{DATA_PATH}/engine_data.csv')

os.makedirs('artifacts', exist_ok=True)
os.makedirs('model', exist_ok=True)

Expected_columns = ["Engine rpm",	"Lub oil pressure",	"Fuel pressure",	"Coolant pressure", "lub oil temp", "Coolant temp",	"Engine Condition"]
for col in df.columns:
  if col not in Expected_columns:
    print(f"Column {col} is not expected")
    df.drop(col, axis=1, inplace=True)
for exp_col in Expected_columns:
  if exp_col not in df.columns:
    print(f"Column {exp_col} is missing")

df.head()

df.shape

df.info()

df.describe()

for col in df.columns:
  plt.figure(figsize=(5,5))
  sns.boxplot(df[col], orient = 'v')
  plt.title(f'Distribution of {col}')
  plt.xlabel(f'{col}')
  plt.ylabel('count')
  plt.show()
  print("\n")
  plt.close()

for col in df.columns:
  sns.histplot(df[col], kde=True)
  plt.title(f'Distribution of {col}')
  plt.xlabel(f'{col}')
  plt.ylabel('count')
  plt.show()
  print("\n")
  plt.close()

for col in df.columns:
  plt.figure(figsize=(5,5))
  sns.boxplot(data=df, y = col, x='Engine Condition' ,hue='Engine Condition', orient='v')
  plt.show()
  print("\n")
  plt.close()

for col in df.columns:
  plt.figure(figsize=(5,5))
  sns.displot(data=df, x=col, col="Engine Condition", bins=20, kde=True)
  plt.show()
  print("\n")
  plt.close()

sns.heatmap(df.corr(), annot=True)
plt.show()
plt.close()

df.shape

df.dropna(inplace=True)

df.shape

df.isnull().sum()

df.shape

df.drop_duplicates(inplace=True)

df.shape

for col in df.columns:
  z_score = np.abs(sts.zscore(df[col]))
  outlier = df[z_score > 3]
  print(f"{col} outliers {outlier}")

x = df.drop('Engine Condition', axis=1)
y = df['Engine Condition']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
scaler = MinMaxScaler()
x_scaled = scaler.fit_transform(x_train)
x_train = pd.DataFrame(x_scaled, columns=x_train.columns)
x_scaled = scaler.transform(x_test)
x_test = pd.DataFrame(x_scaled, columns=x_test.columns)

x_train.head()

y_train.head()

f1 = open(f"{ARTIFACTS_PATH}/x_train.pkl", "wb").write(pickle.dumps(x_train))
f2 = open(f"{ARTIFACTS_PATH}/x_test.pkl", "wb").write(pickle.dumps(x_test))
f3 = open(f"{ARTIFACTS_PATH}/y_train.pkl", "wb").write(pickle.dumps(y_train))
f4 = open(f"{ARTIFACTS_PATH}/y_test.pkl", "wb").write(pickle.dumps(y_test))
f5 = open(f"{ARTIFACTS_PATH}/scaler.pkl", "wb").write(pickle.dumps(scaler))

f1 = open(f"{ARTIFACTS_PATH}/x_train.pkl", "rb")
f2 = open(f"{ARTIFACTS_PATH}/x_test.pkl", "rb")
f3 = open(f"{ARTIFACTS_PATH}/y_train.pkl", "rb")
f4 = open(f"{ARTIFACTS_PATH}/y_test.pkl", "rb")

x_train = pickle.load(f1)
x_test = pickle.load(f2)
y_train = pickle.load(f3)
y_test = pickle.load(f4)

x_train.head()

x_test.head()

y_train.head()

y_test.head()

log_reg_model = LogisticRegression(max_iter=20)
log_reg_model.fit(x_train, y_train)

y_pred = log_reg_model.predict(x_test)

print(accuracy_score(y_test, y_pred))

print(classification_report(y_test, y_pred))

hyper_param_grid = {
    'penalty': [None, 'l2'],
    'C': [0.1, 1, 10, 100],
    'max_iter': [50, 100, 200],
    'class_weight': ['balanced']
}
GridSearch = GridSearchCV(estimator=LogisticRegression(), param_grid=hyper_param_grid, cv=5, scoring='f1_macro')
GridSearch.fit(x_train, y_train)

print(GridSearch.best_params_)

y_pred = GridSearch.predict(x_test)

print(classification_report(y_test, y_pred))

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

rf = RandomForestClassifier()
rf.fit(x_train, y_train)

y_pred = rf.predict(x_test)

print(classification_report(y_test, y_pred))

hyper_param_grid = {
    'n_estimators': [100, 200, 300,500],
    'criterion': ['gini','entropy'],
    'max_depth': [None, 10, 50, 100],
    'min_samples_split': [2, 10, 20],
    'min_samples_leaf': [1, 5, 10],
    'class_weight': ['balanced','balanced_subsample', None]
}

RandomSearch = RandomizedSearchCV(estimator=RandomForestClassifier(), param_distributions=hyper_param_grid, cv=5, scoring='f1_macro', n_iter=10, random_state=42)
RandomSearch.fit(x_train, y_train)

y_pred = RandomSearch.predict(x_test)

print(classification_report(y_test, y_pred))

print(RandomSearch.best_params_)

from xgboost import XGBClassifier
xgb = XGBClassifier()
xgb.fit(x_train, y_train)

y_pred = xgb.predict(x_test)
print(classification_report(y_test, y_pred))

hyper_param_grid = {
    'n_estimators': [100, 200, 300,500],
    'max_depth': [None, 3, 5, 10],
    'learning_rate': [0.01, 0.1, 0.2, 0.3],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'reg_alpha': [0, 0.1, 0.5],
    'reg_lambda': [0, 0.1, 0.5],
    'scale_pos_weight': [1, 2, 3]
}
xgb_random_search = RandomizedSearchCV(estimator=XGBClassifier(), param_distributions=hyper_param_grid, cv=5, scoring='f1_macro', n_iter=10, random_state=42)
xgb_random_search.fit(x_train, y_train)

y_pred = xgb_random_search.predict(x_test)

print(classification_report(y_test, y_pred))

print(xgb_random_search.best_estimator_)

def model_scores(tf_model, x_test, y_test):
   y_pred = tf_model.predict(x_test)
   y_pred = (y_pred > 0.5).astype(int)
   print(accuracy_score(y_test, y_pred))
   print(classification_report(y_test, y_pred))

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam

from sklearn.utils.class_weight import compute_class_weight
classes = np.unique(y_train)
weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
class_weights = dict(zip(classes, weights))
print(class_weights)

y_train = np.asarray(y_train).astype(np.float32)
y_test = np.asarray(y_test).astype(np.float32)
y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)

tf_model = Sequential([
    Input(shape=(x_train.shape[1],)),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

optimizer = Adam(learning_rate=0.001)
tf_model.compile(
    optimizer=optimizer,
    loss='binary_crossentropy',
    metrics=[
        'accuracy',
        tf.keras.metrics.Precision(),
        tf.keras.metrics.Recall()
    ]
)

tf_model.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

loss, acc, precision, recall = tf_model.evaluate(x_test, y_test)
print(f'Test Loss: {loss}')
print(f'Test Accuracy: {acc}')
print(f'Test Precision: {precision}')
print(f'Test Recall: {recall}')

model_scores(tf_model, x_test, y_test)

tf_model.fit(x_train, y_train, epochs=50, batch_size=32, validation_split=0.2, class_weight={0:1, 1:2})

model_scores(tf_model, x_test, y_test)

tf_model.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.2, class_weight={0:1, 1:2})

model_scores(tf_model, x_test, y_test)

tf_model.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.2, class_weight=class_weights)

model_scores(tf_model,x_test, y_test)

from tensorflow.keras.layers import Dropout
from tensorflow.keras.callbacks import EarlyStopping


tf_model = Sequential([
    Input(shape=(x_train.shape[1],)),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])


optimizer = Adam(learning_rate=0.001)
tf_model.compile(
    optimizer=optimizer,
    loss='binary_crossentropy',
    metrics=[
        'accuracy',
        tf.keras.metrics.Precision(),
        tf.keras.metrics.Recall()
    ]
)

es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
tf_model.fit(x_train, y_train, epochs=100, batch_size=32, validation_split=0.2, class_weight=class_weights, callbacks=[es])

model_scores(tf_model, x_test, y_test)

print(tf_model.summary())

print(scaler.feature_names_in_)

test_data = pd.DataFrame([{
            "Engine rpm": 500,
            "Lub oil pressure": 0,
            "Fuel pressure": 0,
            "Coolant pressure": 2,
            "lub oil temp": 80,
            "Coolant temp": 81
        }])
test_data_scaled = scaler.transform(test_data)
print(test_data_scaled)

prediction = tf_model.predict(test_data_scaled)
print(prediction)

import joblib
joblib.dump(scaler, f'{MODEL_PATH}/scaler.pkl')
joblib.dump(tf_model, f'{MODEL_PATH}/tf_model.pkl')
