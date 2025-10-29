# src/explainability.py
# SHAP-based explainability helpers (Phase 4)
import shap
import numpy as np
import pandas as pd

def shap_explain(model, X_train_sample, instance, feature_names):
    """
    Returns SHAP values and base_value for plotting in Streamlit.
    Note: ensure 'shap' is installed.
    """
    explainer = shap.Explainer(model, X_train_sample)
    shap_values = explainer(instance)
    # shap_values has .values and .base_values
    return shap_values
