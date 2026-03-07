import streamlit as st
import cv2
import numpy as np
from src.detectors.yolo import YOLODetector

st.set_page_config(page_title="Perception Benchmark", layout="wide")

st.title("🦅 Bird Perception Stack")
st.write("Current Status: Recognition Engine Online. Stereo Depth Engine Pending.")

# Simple test of your existing YOLO class
if st.button("Initialize YOLOv8n"):
    try:
        detector = YOLODetector()
        st.success("YOLOv8n Loaded Successfully from weights!")
    except Exception as e:
        st.error(f"Error loading weights: {e}")