import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn  
import mlflow.tensorflow  
import matplotlib.pyplot as plt
import keras
import tensorflow as tf
from mlflow.models import infer_signature
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE


def train_logistic_regression(X_train, y_train, X_test, y_test):
    """
    Train and evaluate a Logistic Regression model.
    """
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("Logistic Regression Performance:")
    print(classification_report(y_test, y_pred))
    print("ROC-AUC Score:", roc_auc_score(y_test, y_pred))
    return model

def train_decision_tree(X_train, y_train, X_test, y_test):
    """
    Train and evaluate a Decision Tree model.
    """
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("Decision Tree Performance:")
    print(classification_report(y_test, y_pred))
    print("ROC-AUC Score:", roc_auc_score(y_test, y_pred))
    return model

def train_random_forest(X_train, y_train, X_test, y_test):
    """
    Train and evaluate a Random Forest model.
    """
    model = RandomForestClassifier(random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("Random Forest Performance:")
    print(classification_report(y_test, y_pred))
    print("ROC-AUC Score:", roc_auc_score(y_test, y_pred))
    return model

def train_xgboost(X_train, y_train, X_test, y_test):
    """
    Train and evaluate an XGBoost model.
    model.add(SimpleRNN(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(X_train_rnn, y_train_sm, epochs=15, batch_size=64, validation_split=0.2)
"""
    model = XGBClassifier(random_state=42, scale_pos_weight=(len(y_train) - sum(y_train)) / sum(y_train))
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("XGBoost Performance:")
    print(classification_report(y_test, y_pred))
    print("ROC-AUC Score:", roc_auc_score(y_test, y_pred))
    return model

def train_mlp(X_train, y_train, X_test, y_test):
    """
    Train and evaluate an MLP model.
    """
    model = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("MLP Performance:")
    print(classification_report(y_test, y_pred))
    print("ROC-AUC Score:", roc_auc_score(y_test, y_pred))
    return model

def train_rnn(X_train, y_train, X_test, y_test):
    """
    Train and evaluate an RNN model.
    """
   # Handle class imbalance using SMOTE (Apply on 2D data)
    smote = SMOTE(random_state=42)
    X_train_sm, y_train_sm = smote.fit_resample(X_train.reshape(X_train.shape[0], -1), y_train)

    # Debugging step
    print("X_train_sm shape after SMOTE:", X_train_sm.shape)
    print("X_test shape before reshaping:", X_test.shape)

    # Reshape data AFTER SMOTE
    X_train_rnn = X_train_sm.reshape((X_train_sm.shape[0], 1, -1))  # Fixed
    X_test_rnn = X_test.reshape((X_test.shape[0], 1, -1))  # Fixed

    # Initialize the RNN model
    model = Sequential()
    model.add(SimpleRNN(64, activation='relu', return_sequences=True, input_shape=(1, X_train_sm.shape[1])))
    model.add(SimpleRNN(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(X_train_rnn, y_train_sm, epochs=15, batch_size=64, validation_split=0.2)

    # Evaluate the model
    y_pred = (model.predict(X_test_rnn) > 0.5).astype(int)
    print("RNN Performance:")
    print(classification_report(y_test, y_pred))
    print("ROC-AUC Score:", roc_auc_score(y_test, y_pred))

    return model

def train_lstm(X_train, y_train, X_test, y_test):
    """
    Train and evaluate an LSTM model.
    """


    # Debug: Print shapes before SMOTE
    print("X_train shape before SMOTE:", X_train.shape)
    print("X_test shape before reshaping:", X_test.shape)

    # Handle class imbalance using SMOTE
    smote = SMOTE(random_state=42)
    X_train_sm, y_train_sm = smote.fit_resample(X_train.reshape(X_train.shape[0], -1), y_train)

    # Debug: Print shapes after SMOTE
    print("X_train_sm shape after SMOTE:", X_train_sm.shape)

    # Reshape Data
    X_train_lstm = X_train_sm.reshape((X_train_sm.shape[0], 1, -1))  # Fixed
    X_test_lstm = X_test.reshape((X_test.shape[0], 1, -1))  # Fixed

    # Initialize LSTM model
    model = Sequential()
    model.add(LSTM(64, activation='relu', return_sequences=True, input_shape=(1, X_train_sm.shape[1])))
    model.add(LSTM(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(X_train_lstm, y_train_sm, epochs=15, batch_size=64, validation_split=0.2)

    # Evaluate the model
    y_pred = (model.predict(X_test_lstm) > 0.5).astype(int)
    print("LSTM Performance:")
    print(classification_report(y_test, y_pred))
    print("ROC-AUC Score:", roc_auc_score(y_test, y_pred))

    return model
def plot_radar_chart(model_metrics, dataset_name):
    """
    Generate a radar chart to compare model performances.
    """
    labels = list(model_metrics[next(iter(model_metrics))].keys())  # Get metric names
    num_vars = len(labels)
    
    angles = [n / float(num_vars) * 2 * np.pi for n in range(num_vars)]
    angles += angles[:1]  # Complete the loop
    
    plt.figure(figsize=(8, 8))
    ax = plt.subplot(111, polar=True)
    
    for model_name, metrics in model_metrics.items():
        values = list(metrics.values())
        values += values[:1]  # Complete the loop
        ax.plot(angles, values, label=model_name, linewidth=2)
        ax.fill(angles, values, alpha=0.1)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    plt.title(f'Model Comparison - {dataset_name}')
    plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1))
    plt.show()
def evaluate_models(models, X_train, y_train, X_test, y_test, dataset_name):
    model_metrics = {}

    # Start a main experiment (parent run)
    with mlflow.start_run():
        mlflow.set_tag("dataset", dataset_name)  # Log dataset name as a tag

        for model_name, train_func in models.items():
            print(f"\nTraining {model_name}...")

            # Reshape Data for RNN/LSTM models
            if model_name in ["RNN", "LSTM"]:
                X_train_reshaped = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
                X_test_reshaped = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
            else:
                X_train_reshaped = X_train
                X_test_reshaped = X_test

            # Start a new nested run for each model
            with mlflow.start_run(nested=True):
                mlflow.log_param("model", model_name)  # Log model name

                # Train Model
                try:
                    model = train_func(X_train_reshaped, y_train, X_test_reshaped, y_test)
                    if model is None:
                        raise ValueError(f"{model_name} training function returned None")
                except Exception as e:
                    print(f"Error training {model_name}: {e}")
                    continue  # Skip to the next model if training fails

                # Predictions
                try:
                    if model_name in ["RNN", "LSTM"]:
                        y_pred_proba = model.predict(X_test_reshaped)
                        y_pred = (y_pred_proba > 0.5).astype(int)
                        input_example = X_test_reshaped[:1]  # 3D example for RNN/LSTM
                    else:
                        y_pred = model.predict(X_test_reshaped)
                        input_example = X_test_reshaped[:1]  # 2D example for others
                except Exception as e:
                    print(f"Error predicting with {model_name}: {e}")
                    continue  # Skip to the next model if prediction fails

                # Generate signature
                signature = infer_signature(X_test_reshaped, y_pred)

                # Log Model to MLflow
                try:
                    if isinstance(model, (LogisticRegression, DecisionTreeClassifier, RandomForestClassifier, XGBClassifier, MLPClassifier)):
                        # Log scikit-learn models
                        mlflow.sklearn.log_model(
                            model, 
                            f"{model_name}_model",
                            signature=signature,
                            input_example=input_example
                        )
                    elif isinstance(model, keras.Model):
                        # Log TensorFlow models
                        mlflow.tensorflow.log_model(
                            model,
                            f"{model_name}_model",
                            signature=signature,
                            input_example=input_example
                        )
                except Exception as e:
                    print(f"Error logging {model_name} model to MLflow: {e}")
                    continue  # Skip logging if it fails

                # Compute Metrics and Store in model_metrics
                try:
                    accuracy = accuracy_score(y_test, y_pred)
                    precision = precision_score(y_test, y_pred)
                    recall = recall_score(y_test, y_pred)
                    f1 = f1_score(y_test, y_pred)

                    model_metrics[model_name] = {
                        "accuracy": accuracy,
                        "precision": precision,
                        "recall": recall,
                        "f1_score": f1
                    }

                    # Log Metrics to MLflow
                    mlflow.log_metrics({
                        "accuracy": accuracy,
                        "precision": precision,
                        "recall": recall,
                        "f1_score": f1
                    })
                except Exception as e:
                    print(f"Error calculating metrics for {model_name}: {e}")
                    continue  # Skip metrics if computation fails

    # Debugging - Print model_metrics before plotting
    print("\nFinal Model Metrics:", model_metrics)

    # Ensure model_metrics is not empty before calling plot_radar_chart
    if model_metrics:
        plot_radar_chart(model_metrics, dataset_name)
    else:
        print("No models were successfully trained and evaluated.")