import streamlit as st
import cv2
import numpy as np
import io

st.set_page_config(page_title="Data Lab", layout="wide")

st.title("🧪 Data Lab: Stereo Asset Loader")
st.write("Upload your stereo images, camera configuration, and ground truth depth map.")

# --- Session State Initialization ---
if 'pipeline_data' not in st.session_state:
    st.session_state['pipeline_data'] = {}

# --- 1. File Uploaders ---
st.subheader("Step 1: Upload Assets")
col1, col2 = st.columns(2)

with col1:
    up_l = st.file_uploader("Left Image (Reference)", type=['png', 'jpg', 'jpeg'])
    up_conf = st.file_uploader("Camera Config (.txt or .conf)", type=['txt', 'conf'])

with col2:
    up_r = st.file_uploader("Right Image (Stereo Match)", type=['png', 'jpg', 'jpeg'])
    up_gt = st.file_uploader("Ground Truth Depth (.npy)", type=['npy'])

# --- 2. Processing and Display ---
if up_l and up_r and up_conf and up_gt:
    # Read Images
    img_l = cv2.imdecode(np.frombuffer(up_l.read(), np.uint8), cv2.IMREAD_COLOR)
    img_r = cv2.imdecode(np.frombuffer(up_r.read(), np.uint8), cv2.IMREAD_COLOR)
    
    # Read Config (as text)
    conf_content = up_conf.read().decode("utf-8")
    
    # Read Ground Truth (.npy)
    # Note: Using BytesIO to allow numpy to load from the uploaded file buffer
    gt_depth = np.load(io.BytesIO(up_gt.read()))

    st.success("✅ All assets loaded successfully!")

    # --- 3. Asset Visualization ---
    st.divider()
    st.subheader("Step 2: Asset Visualization")

    # Display Stereo Pair
    st.write("### 📸 Stereo Pair")
    v_col1, v_col2 = st.columns(2)
    v_col1.image(cv2.cvtColor(img_l, cv2.COLOR_BGR2RGB), caption="Left View (Reference)", use_container_width=True)
    v_col2.image(cv2.cvtColor(img_r, cv2.COLOR_BGR2RGB), caption="Right View (Stereo)", use_container_width=True)

    # Display Ground Truth and Config Info
    st.write("### 📊 Depth & Metadata")
    v_col3, v_col4 = st.columns([2, 1])

    with v_col3:
        # Normalize depth map for visualization (colors: purple = near, yellow = far)
        # Handle potential NaNs or Inf in depth data
        gt_vis = np.nan_to_num(gt_depth, nan=0.0, posinf=np.nanmax(gt_depth[np.isfinite(gt_depth)]))
        st.image(gt_vis / np.max(gt_vis), caption="Ground Truth Depth Map (Normalized)", use_container_width=True)
        

    with v_col4:
        st.info("📄 Configuration File Content")
        st.text_area("Raw Config", conf_content, height=300)

    # --- 4. Store for Next Pages ---
    if st.button("🚀 Lock Data & Proceed to Benchmark"):
        st.session_state['pipeline_data'] = {
            "left": img_l,
            "right": img_r,
            "gt": gt_depth,
            "conf_raw": conf_content
        }
        st.success("Data stored in session! Move to the 'Recognition' or 'Tuning' page.")

else:
    st.info("Please upload all 4 files to proceed with the perception pipeline.")