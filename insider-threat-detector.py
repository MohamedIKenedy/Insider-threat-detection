# -*- coding: utf-8 -*-
"""Untitled36.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1wIyswNUg-7nEt_BHmijwhgDlFIgbW3ve
"""

!pip install numpy
!pip install ngrok
!pip install pandas
!pip install mlflow
!pip install scikit-learn
!pip install tensorflow
!pip install matplotlib
!pip install seaborn

import kagglehub
import os
import shutil

# Download latest version
path = kagglehub.dataset_download("nitishabharathi/cert-insider-threat")

# Create the 'content' directory if it doesn't exist
if not os.path.exists('/content/cert-insider-threat'):
    os.makedirs('/content/cert-insider-threat')

# Move the downloaded files to the 'content' directory
for file_name in os.listdir(path):
    source_path = os.path.join(path, file_name)
    destination_path = os.path.join('/content/cert-insider-threat', file_name)
    shutil.move(source_path, destination_path)

print("Dataset files saved to: /content/cert-insider-threat")

import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
import mlflow.keras
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest, RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.layers import Dense, GRU
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os

# Configure MLflow
def setup_mlflow_tracking():
    # Start ngrok tunnel (you'll need to run this in Colab first)
    # !pip install pyngrok
    # from pyngrok import ngrok
    # ngrok_tunnel = ngrok.connect(5000)
    # print('MLflow Tracking URI:', ngrok_tunnel.public_url)

    # Set MLflow tracking URI to your ngrok URL
    # mlflow.set_tracking_uri(ngrok_tunnel.public_url)
    experiment_name = "insider-threat-detection"
    try:
        mlflow.create_experiment(experiment_name)
    except:
        pass
    mlflow.set_experiment(experiment_name)

def plot_training_history(history, model_name):
    """Plot and save training history"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Plot accuracy
    ax1.plot(history.history['accuracy'])
    ax1.plot(history.history['val_accuracy'])
    ax1.set_title(f'{model_name} - Accuracy')
    ax1.set_ylabel('Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.legend(['Train', 'Validation'])

    # Plot loss
    ax2.plot(history.history['loss'])
    ax2.plot(history.history['val_loss'])
    ax2.set_title(f'{model_name} - Loss')
    ax2.set_ylabel('Loss')
    ax2.set_xlabel('Epoch')
    ax2.legend(['Train', 'Validation'])

    plt.tight_layout()
    return fig

def plot_feature_importance(model, feature_names):
    """Plot feature importance for a given model"""
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(10, 6))
    plt.title("Feature Importance")
    plt.bar(range(X_train.shape[1]), importances[indices], align="center")
    plt.xticks(range(X_train.shape[1]), feature_names[indices], rotation=90)
    plt.tight_layout()
    return plt.gcf()

def plot_roc_curve(y_true, y_pred_proba, model_name):
    """Plot ROC curve"""
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{model_name} - ROC Curve')
    plt.legend(loc='lower right')
    return plt.gcf()

def preprocess_data(data_path, sample_frac=0.1):
    # Load and sample data
    data = pd.read_csv(data_path)
    data_sample = data.sample(frac=sample_frac, random_state=42)

    # Basic preprocessing
    data_sample['date'] = pd.to_datetime(data_sample['date'])
    data_sample['cc'] = data_sample['cc'].fillna('')
    data_sample['bcc'] = data_sample['bcc'].fillna('')

    # Feature engineering
    data_sample['num_recipients'] = (data_sample['to'].str.count(';') +
                                   data_sample['cc'].str.count(';') +
                                   data_sample['bcc'].str.count(';') + 1)
    data_sample['hour'] = data_sample['date'].dt.hour
    data_sample['day_of_week'] = data_sample['date'].dt.dayofweek
    data_sample['is_weekend'] = data_sample['day_of_week'].isin([5, 6]).astype(int)
    data_sample['is_night'] = ((data_sample['hour'] < 6) | (data_sample['hour'] > 22)).astype(int)

    # TF-IDF vectorization
    tfidf = TfidfVectorizer(max_features=1000, stop_words='english')
    content_tfidf = tfidf.fit_transform(data_sample['content']).toarray()
    content_tfidf_df = pd.DataFrame(content_tfidf, columns=tfidf.get_feature_names_out())

    # Combine features
    features = ['size', 'attachments', 'num_recipients', 'hour', 'day_of_week',
               'is_weekend', 'is_night']
    X_numeric = data_sample[features]
    X = pd.concat([X_numeric.reset_index(drop=True),
                  content_tfidf_df.reset_index(drop=True)], axis=1)

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

    # Generate labels using Isolation Forest
    iso_forest = IsolationForest(contamination=0.01, random_state=42)
    labels = iso_forest.fit_predict(X_scaled)
    y = pd.Series(labels).map({1: 0, -1: 1})

    return train_test_split(X_scaled, y, test_size=0.2, random_state=42)

def create_lstm_model(input_shape):
    model = Sequential([
        LSTM(128, input_shape=input_shape, return_sequences=True),
        BatchNormalization(),
        Dropout(0.3),
        LSTM(64),
        BatchNormalization(),
        Dropout(0.3),
        Dense(32, activation='relu'),
        BatchNormalization(),
        Dense(2, activation='softmax')
    ])
    model.compile(optimizer='adam',
                 loss='categorical_crossentropy',
                 metrics=['accuracy'])
    return model

def create_gru_model(input_shape):
    model = Sequential([
        GRU(128, input_shape=input_shape, return_sequences=True),
        BatchNormalization(),
        Dropout(0.3),
        GRU(64),
        BatchNormalization(),
        Dropout(0.3),
        Dense(32, activation='relu'),
        BatchNormalization(),
        Dense(2, activation='softmax')
    ])
    model.compile(optimizer='adam',
                 loss='categorical_crossentropy',
                 metrics=['accuracy'])
    return model

def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    setup_mlflow_tracking()

    # Common callbacks for deep learning models
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        ModelCheckpoint('best_model.keras', monitor='val_loss', save_best_only=True)
    ]

    # Train and evaluate each model
    with mlflow.start_run(run_name="insider_threat_detection"):
        # Random Forest
        with mlflow.start_run(run_name="random_forest", nested=True):
            rf = RandomForestClassifier(n_estimators=200,
                                      max_depth=20,
                                      min_samples_split=10,
                                      random_state=42)
            rf.fit(X_train, y_train)
            y_pred_rf = rf.predict(X_test)
            y_pred_rf_proba = rf.predict_proba(X_test)[:, 1]

            # Log metrics and plots
            mlflow.log_metrics({
                "accuracy": accuracy_score(y_test, y_pred_rf),
                "precision": precision_score(y_test, y_pred_rf),
                "recall": recall_score(y_test, y_pred_rf),
                "f1": f1_score(y_test, y_pred_rf)
            })

            # Log ROC curve
            roc_plot = plot_roc_curve(y_test, y_pred_rf_proba, "Random Forest")
            mlflow.log_figure(roc_plot, "random_forest_roc.png")

            # Log feature importance
            feature_imp_plot = plot_feature_importance(rf, X_train.columns)
            mlflow.log_figure(feature_imp_plot, "random_forest_feature_importance.png")

            mlflow.sklearn.log_model(rf, "random_forest_model")

        # LSTM
        with mlflow.start_run(run_name="lstm", nested=True):
            X_train_lstm = X_train.values.reshape((X_train.shape[0], 1, X_train.shape[1]))
            X_test_lstm = X_test.values.reshape((X_test.shape[0], 1, X_test.shape[1]))
            y_train_cat = to_categorical(y_train)
            y_test_cat = to_categorical(y_test)

            lstm_model = create_lstm_model((X_train_lstm.shape[1], X_train_lstm.shape[2]))
            history = lstm_model.fit(X_train_lstm, y_train_cat,
                                   epochs=50,
                                   batch_size=64,
                                   validation_split=0.2,
                                   callbacks=callbacks,
                                   verbose=2)

            y_pred_lstm = np.argmax(lstm_model.predict(X_test_lstm), axis=1)
            y_pred_lstm_proba = lstm_model.predict(X_test_lstm)[:, 1]

            # Log metrics and plots
            mlflow.log_metrics({
                "accuracy": accuracy_score(y_test, y_pred_lstm),
                "precision": precision_score(y_test, y_pred_lstm),
                "recall": recall_score(y_test, y_pred_lstm),
                "f1": f1_score(y_test, y_pred_lstm)
            })

            # Log training history
            history_plot = plot_training_history(history, "LSTM")
            mlflow.log_figure(history_plot, "lstm_training_history.png")

            # Log ROC curve
            roc_plot = plot_roc_curve(y_test, y_pred_lstm_proba, "LSTM")
            mlflow.log_figure(roc_plot, "lstm_roc.png")

            mlflow.keras.log_model(lstm_model, "lstm_model")

        # Stacking Ensemble
        with mlflow.start_run(run_name="stacking_ensemble", nested=True):
            estimators = [
                ('rf', RandomForestClassifier(n_estimators=200, random_state=42)),
                ('gbdt', GradientBoostingClassifier(n_estimators=200, random_state=42)),
                ('mlp', MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=300, random_state=42))
            ]

            stacking = StackingClassifier(
                estimators=estimators,
                final_estimator=RandomForestClassifier(n_estimators=100, random_state=42),
                cv=5
            )

            stacking.fit(X_train, y_train)
            y_pred_stacking = stacking.predict(X_test)
            y_pred_stacking_proba = stacking.predict_proba(X_test)[:, 1]

            # Log metrics and plots
            mlflow.log_metrics({
                "accuracy": accuracy_score(y_test, y_pred_stacking),
                "precision": precision_score(y_test, y_pred_stacking),
                "recall": recall_score(y_test, y_pred_stacking),
                "f1": f1_score(y_test, y_pred_stacking)
            })

            # Log ROC curve
            roc_plot = plot_roc_curve(y_test, y_pred_stacking_proba, "Stacking Ensemble")
            mlflow.log_figure(roc_plot, "stacking_ensemble_roc.png")

            mlflow.sklearn.log_model(stacking, "stacking_ensemble_model")

def main():
    # Set random seed for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)

    # Load and preprocess data
    X_train, X_test, y_train, y_test = preprocess_data('/content/cert-insider-threat/email.csv')

    # Train and evaluate models
    train_and_evaluate_models(X_train, X_test, y_train, y_test)

if __name__ == "__main__":
    main()

