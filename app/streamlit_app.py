## 1. Import libraries
import streamlit as st
import pandas as pd
import joblib

## 2. Load Models
heart_model = joblib.load("app/heart_model.pkl")
stroke_model = joblib.load("app/stroke_model.pkl")
