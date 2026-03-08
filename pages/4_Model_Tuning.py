import streamlit as st
import cv2
import numpy as np
import time
import plotly.graph_objects as go
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.detectors.rce.features import REGISTRY

st.set_page_config(page_title="Model Tuning", layout="wide")
st.title("⚙️ Model Tuning: Train & Compare")

# ---------------------------------------------------------------------------
# Guard: require Data Lab & Feature Lab completion
# ---------------------------------------------------------------------------
if "pipeline_data" not in st.session_state or "crop" not in st.session_state.get("pipeline_data", {}):
    st.error("Please complete the **Data Lab** first (upload assets & define a crop).")
    st.stop()

assets = st.session_state["pipeline_data"]
crop = assets.get("crop_aug", assets["crop"])
crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
active_modules = st.session_state.get("active_modules", {k: True for k in REGISTRY})


# ---------------------------------------------------------------------------
# Cached model loaders
# ---------------------------------------------------------------------------
@st.cache_resource
def load_resnet():
    from src.detectors.resnet import ResNetDetector
    return ResNetDetector()

@st.cache_resource
def load_mobilenet():
    from src.detectors.mobilenet import MobileNetDetector
    return MobileNetDetector()

@st.cache_resource
def load_mobilevit():
    from src.detectors.mobilevit import MobileViTDetector
    return MobileViTDetector()

CNN_MODELS = {
    "ResNet-18":      {"loader": load_resnet,    "dim": 512},
    "MobileNetV3":    {"loader": load_mobilenet,  "dim": 576},
    "MobileViT-XXS":  {"loader": load_mobilevit,  "dim": 320},
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_training_images():
    """Scan the data folder for bird/background images."""
    from pathlib import Path
    from src.config import PROJECT_ROOT
    train_dir = PROJECT_ROOT / "data/artroom/bird/yolo/train/images"
    images, labels = [], []
    if not train_dir.exists():
        return images, labels
    for img_file in sorted(train_dir.glob("*.png")):
        img = cv2.imread(str(img_file))
        if img is None:
            continue
        fname = img_file.name.lower()
        if "bird" in fname:
            images.append(img)
            labels.append("bird")
        elif any(x in fname for x in ["room", "wall", "floor", "empty"]):
            images.append(img)
            labels.append("background")
    return images, labels


def build_rce_vector(img_bgr):
    """Build the RCE feature vector from active modules."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    vec = []
    for key, meta in REGISTRY.items():
        if active_modules.get(key, False):
            v, _ = meta["fn"](gray)
            vec.extend(v)
    return np.array(vec, dtype=np.float32)


# ===================================================================
# LAYOUT: LEFT = RCE  |  RIGHT = CNN
# ===================================================================
col_rce, col_cnn = st.columns(2)

# ---------------------------------------------------------------------------
# LEFT — RCE Training
# ---------------------------------------------------------------------------
with col_rce:
    st.header("🧬 RCE Training")

    # Show active modules
    active_names = [REGISTRY[k]["label"] for k in active_modules if active_modules[k]]
    if not active_names:
        st.error("No RCE modules selected. Go back to Feature Lab.")
        st.stop()

    st.write(f"**Active modules:** {', '.join(active_names)}")
    st.image(crop_rgb, caption="Training Crop", width=200)

    # Training config
    st.subheader("Training Parameters")
    rce_C = st.slider("Regularization (C)", 0.01, 10.0, 1.0, step=0.01,
                       help="Higher = less regularization, may overfit")
    rce_max_iter = st.slider("Max Iterations", 100, 5000, 1000, step=100)

    if st.button("🚀 Train RCE Head"):
        images, labels = load_training_images()
        if not images:
            st.error("No training images found in `data/artroom/bird/yolo/train/images/`")
        else:
            from sklearn.linear_model import LogisticRegression
            import joblib
            from src.config import MODEL_PATHS

            # Progress
            progress = st.progress(0, text="Extracting RCE features...")
            n = len(images)
            X = []
            for i, img in enumerate(images):
                X.append(build_rce_vector(img))
                progress.progress((i + 1) / n, text=f"Feature extraction: {i+1}/{n}")

            X = np.array(X)
            progress.progress(1.0, text="Fitting Logistic Regression...")

            t0 = time.perf_counter()
            head = LogisticRegression(max_iter=rce_max_iter, C=rce_C)
            head.fit(X, labels)
            train_time = time.perf_counter() - t0

            # Save
            head_path = str(MODEL_PATHS.get("rce_model", "models/rce_model.pkl"))
            joblib.dump(head, head_path)

            progress.progress(1.0, text="✅ Training complete!")

            # --- Results ---
            st.success(f"Trained in **{train_time:.2f}s** — saved to `{head_path}`")

            # Metrics
            from sklearn.metrics import accuracy_score
            preds = head.predict(X)
            train_acc = accuracy_score(labels, preds)

            m1, m2, m3 = st.columns(3)
            m1.metric("Train Accuracy", f"{train_acc:.1%}")
            m2.metric("Vector Size", f"{X.shape[1]} floats")
            m3.metric("Train Time", f"{train_time:.2f}s")

            # Confidence distribution chart
            probs = head.predict_proba(X)
            fig = go.Figure()
            for ci, cls in enumerate(head.classes_):
                fig.add_trace(go.Histogram(
                    x=probs[:, ci], name=cls, opacity=0.7, nbinsx=20
                ))
            fig.update_layout(title="Confidence Distribution",
                              barmode="overlay", template="plotly_dark", height=280,
                              xaxis_title="Confidence", yaxis_title="Count")
            st.plotly_chart(fig, use_container_width=True)

            # Store
            st.session_state["rce_head"] = head
            st.session_state["rce_train_acc"] = train_acc

    # Sanity-check predict
    if "rce_head" in st.session_state:
        st.divider()
        st.subheader("Quick Predict (Crop)")
        head = st.session_state["rce_head"]
        t0 = time.perf_counter()
        vec = build_rce_vector(crop)
        probs = head.predict_proba([vec])[0]
        dt = (time.perf_counter() - t0) * 1000
        idx = np.argmax(probs)
        st.write(f"**{head.classes_[idx]}** — {probs[idx]:.1%} confidence — {dt:.1f} ms")


# ---------------------------------------------------------------------------
# RIGHT — CNN Fine-Tuning
# ---------------------------------------------------------------------------
with col_cnn:
    st.header("🧠 CNN Fine-Tuning")

    selected = st.selectbox("Select Model", list(CNN_MODELS.keys()))
    meta = CNN_MODELS[selected]

    st.caption(f"Backbone embedding: **{meta['dim']}D** → Logistic Regression head")
    st.image(crop_rgb, caption="Training Crop", width=200)

    # Training config
    st.subheader("Training Parameters")
    cnn_C = st.slider("Regularization (C) ", 0.01, 10.0, 1.0, step=0.01,
                       key="cnn_c", help="Higher = less regularization")
    cnn_max_iter = st.slider("Max Iterations ", 100, 5000, 1000, step=100,
                              key="cnn_iter")

    if st.button(f"🚀 Train {selected} Head"):
        images, labels = load_training_images()
        if not images:
            st.error("No training images found in `data/artroom/bird/yolo/train/images/`")
        else:
            detector = meta["loader"]()

            progress = st.progress(0, text=f"Extracting {selected} features...")
            n = len(images)
            X = []
            for i, img in enumerate(images):
                X.append(detector._get_features(img))
                progress.progress((i + 1) / n, text=f"Feature extraction: {i+1}/{n}")

            X = np.array(X)
            progress.progress(1.0, text="Fitting Logistic Regression...")

            from sklearn.linear_model import LogisticRegression
            t0 = time.perf_counter()
            head = LogisticRegression(max_iter=cnn_max_iter, C=cnn_C)
            head.fit(X, labels)
            train_time = time.perf_counter() - t0

            # Assign head to detector and save
            detector.head = head
            import joblib
            if hasattr(detector, "head_path") and detector.head_path:
                joblib.dump(head, detector.head_path)

            progress.progress(1.0, text="✅ Training complete!")

            st.success(f"Trained in **{train_time:.2f}s**")

            from sklearn.metrics import accuracy_score
            preds = head.predict(X)
            train_acc = accuracy_score(labels, preds)

            m1, m2, m3 = st.columns(3)
            m1.metric("Train Accuracy", f"{train_acc:.1%}")
            m2.metric("Vector Size", f"{X.shape[1]}D")
            m3.metric("Train Time", f"{train_time:.2f}s")

            # Confidence distribution
            probs = head.predict_proba(X)
            fig = go.Figure()
            for ci, cls in enumerate(head.classes_):
                fig.add_trace(go.Histogram(
                    x=probs[:, ci], name=cls, opacity=0.7, nbinsx=20
                ))
            fig.update_layout(title="Confidence Distribution",
                              barmode="overlay", template="plotly_dark", height=280,
                              xaxis_title="Confidence", yaxis_title="Count")
            st.plotly_chart(fig, use_container_width=True)

            st.session_state[f"cnn_head_{selected}"] = head
            st.session_state[f"cnn_acc_{selected}"] = train_acc

    # Sanity-check predict
    if f"cnn_head_{selected}" in st.session_state:
        st.divider()
        st.subheader("Quick Predict (Crop)")
        detector = meta["loader"]()
        head = st.session_state[f"cnn_head_{selected}"]
        t0 = time.perf_counter()
        feats = detector._get_features(crop)
        probs = head.predict_proba([feats])[0]
        dt = (time.perf_counter() - t0) * 1000
        idx = np.argmax(probs)
        st.write(f"**{head.classes_[idx]}** — {probs[idx]:.1%} confidence — {dt:.1f} ms")


# ===========================================================================
# Bottom — Side-by-side comparison table
# ===========================================================================
st.divider()
st.subheader("📊 Training Comparison")

rce_acc = st.session_state.get("rce_train_acc")
rows = []
if rce_acc is not None:
    rows.append({"Model": "RCE", "Train Accuracy": f"{rce_acc:.1%}",
                 "Vector Size": str(sum(
                     10 for k in active_modules if active_modules[k]
                 ))})
for name in CNN_MODELS:
    acc = st.session_state.get(f"cnn_acc_{name}")
    if acc is not None:
        rows.append({"Model": name, "Train Accuracy": f"{acc:.1%}",
                     "Vector Size": f"{CNN_MODELS[name]['dim']}D"})

if rows:
    import pandas as pd
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
else:
    st.info("Train at least one model to see the comparison.")