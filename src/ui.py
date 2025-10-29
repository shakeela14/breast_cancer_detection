# src/ui.py
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

def set_page_theme():
    st.set_page_config(page_title="Breast Cancer Detection", page_icon="🎗️", layout="wide")

def css_style(theme="light"):
    if theme == "dark":
        background = "#0f1720"
        text = "white"
        card = "#111827"
    else:
        background = "#f8fafc"
        text = "#0f1720"
        card = "white"

    st.markdown(f"""
    <style>
    body {{ background: {background}; color: {text}; }}
    .card {{ background: {card}; padding: 1rem; border-radius: 12px; box-shadow: 0 6px 18px rgba(0,0,0,0.08); }}
    .pred {{ padding: 10px; border-radius: 8px; font-weight:600; }}
    </style>
    """, unsafe_allow_html=True)

def confidence_gauge(prob):
    # simple gauge using matplotlib (speedometer-like)
    fig, ax = plt.subplots(figsize=(4,2.2))
    ax.set_xlim(0, 100)
    ax.barh([0], [prob*100], height=0.6)
    ax.set_yticks([])
    ax.set_xlabel("Confidence (%)")
    ax.set_xlim(0, 100)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    return fig
