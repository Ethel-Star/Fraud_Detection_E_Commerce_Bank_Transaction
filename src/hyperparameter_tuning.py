import numpy as np
import pandas as pd
import shap
import lime
import lime.lime_tabular
import joblib
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt

from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, roc_curve

# -------------------------------
# 🔹 Function for Hyperparameter Tuning
# -------------------------------
def hyperparameter_tuning(X_train, y_train, X_test, y_test, model_name, dataset_name):
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 20],
        'bootstrap': [True, False]
    }
    
    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(
        estimator=rf, param_grid=param_grid, 
        cv=5, n_jobs=-1, verbose=2, scoring='roc_auc'
    )
    
    grid_search.fit(X_train, y_train)
    print(f"Best Parameters for {model_name} on {dataset_name}: ", grid_search.best_params_)
    
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)

    print(f"\n{model_name} Performance on {dataset_name} Test Data:")
    print(classification_report(y_test, y_pred))
    print(f"ROC-AUC Score: {roc_auc_score(y_test, y_pred)}")

    # ✅ Fix MLflow Warning: Convert integer columns to float
    input_example = X_test.copy().astype(np.float64).iloc[:5]  # Convert all columns to float
    signature = mlflow.models.infer_signature(X_test.astype(np.float64), best_model.predict(X_test))

    with mlflow.start_run():
        mlflow.sklearn.log_model(
            sk_model=best_model, 
            artifact_path=f"{model_name}_{dataset_name}_Model", 
            input_example=input_example,
            signature=signature
        )
        mlflow.log_params(grid_search.best_params_)
        mlflow.log_metric("roc_auc", roc_auc_score(y_test, y_pred))
    
    # ✅ Save Model
    model_filename = f"{dataset_name}_{model_name}_best_model.pkl"
    joblib.dump(best_model, model_filename)
    print(f"Model saved as {model_filename}")
    
    return best_model, grid_search

# -------------------------------
# 🔹 Function to Evaluate Models for Both Datasets
# -------------------------------
def evaluate_models(models, X_train, y_train, X_test, y_test, dataset_name):
    for model_name, model_function in models.items():
        print(f"\nTuning and Evaluating {model_name} for {dataset_name}")
        best_model, _ = model_function(X_train, y_train, X_test, y_test, model_name, dataset_name)

        # Calculate ROC Curve
        y_pred_prob = best_model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
        
        # Plot ROC Curve
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='blue', label=f'{model_name} (AUC = {roc_auc_score(y_test, y_pred_prob):.2f})')
        plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
        plt.title(f'ROC Curve for {model_name} on {dataset_name}')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend(loc='lower right')
        plt.show()
    
    return best_model


def explain_model_with_shap(best_model, X_train, dataset_name):
    """
    Generalized SHAP explanation function for different model types and datasets.
    Handles binary classification, multiclass, and regression.
    """
    # Ensure that the input data is a numpy array (SHAP works best with arrays)
    if not isinstance(X_train, np.ndarray):
        X_train = X_train.values  # Convert DataFrame to numpy array if needed

    print(f"Explaining model for {dataset_name} using SHAP...")
    
    # Try initializing the SHAP explainer
    try:
        if isinstance(best_model, shap.explainers._tree.TreeExplainer):
            explainer = shap.TreeExplainer(best_model)
        else:
            # For other models, use KernelExplainer
            explainer = shap.KernelExplainer(best_model.predict_proba, X_train)
    except Exception as e:
        print(f"Error initializing SHAP explainer: {e}")
        return

    # Try to get SHAP values and handle different output types (binary, multiclass, regression)
    try:
        shap_values = explainer.shap_values(X_train)
        
        # For binary classification, shap_values is a list with two entries (for class 0 and class 1)
        # For multiclass classification, shap_values will have one entry per class
        # For regression, shap_values is a single array
        if isinstance(shap_values, list):
            print("SHAP values shape (list):", [v.shape for v in shap_values])  # Print shape for each class
            shap_values_to_use = shap_values[1]  # Use the class of interest (1)
        else:
            print("SHAP values shape (single array):", shap_values.shape)
            shap_values_to_use = shap_values  # Use directly for regression or single-class classification
        
        # Visualize SHAP summary plot (Bar plot)
        shap.summary_plot(shap_values_to_use, X_train, plot_type="bar", show=True)
        # Visualize SHAP summary plot (default)
        shap.summary_plot(shap_values_to_use, X_train, show=True)

        print(f"SHAP explanations for {dataset_name} have been visualized.")
        
    except Exception as e:
        print(f"Error generating SHAP values: {e}")

    
def explain_model_with_lime(best_model, X_train, y_train, dataset_name):
    # Convert NumPy array back to DataFrame (fix StandardScaler issue)
    if isinstance(X_train, np.ndarray):
        X_train = pd.DataFrame(X_train)

    # LIME Explainer
    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=X_train.values, 
        feature_names=X_train.columns, 
        class_names=[str(i) for i in np.unique(y_train)], 
        discretize_continuous=True
    )

    # Explain a single instance
    i = 20
    explanation = explainer.explain_instance(X_train.iloc[i].values, best_model.predict_proba)

    # Visualize Explanation
    explanation.show_in_notebook(show_table=True)

    print(f"LIME explanations for {dataset_name} visualized successfully!")