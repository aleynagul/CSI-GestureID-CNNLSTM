import streamlit as st
import numpy as np
import pandas as pd
import time
import os
from tensorflow.keras.models import load_model
import pickle

# 🔥 SAYFA
st.set_page_config(layout="wide")
st.title("WiFi Gesture AI (Simulated Live)")

col1, col2 = st.columns([3,1])
chart = col1.empty()
result_box = col2.empty()

# 📁 BASE PATH
BASE_DIR = os.path.dirname(os.path.dirname(__file__))

# 🚀 MODEL LOAD (CACHE = EN KRİTİK)
@st.cache_resource
def load_all():

    activity_model = load_model(
        os.path.join(BASE_DIR, "notebooks", "activity_model.h5")
    )

    embedding_model = load_model(
        os.path.join(BASE_DIR, "models", "embedding_model.h5")
    )

    with open(os.path.join(BASE_DIR, "gallery.pkl"), "rb") as f:
        gallery = pickle.load(f)

    print("Gallery:", gallery.keys())
    print("Gallery inside loop:", gallery.keys())

    return activity_model, embedding_model, gallery


activity_model, embedding_model, gallery = load_all()

labels = ["LEFT", "RIGHT", "UP", "DOWN"]

# 📡 DATA LOAD
if "data" not in st.session_state:
    X_test = np.load(os.path.join(BASE_DIR, "notebooks", "X.npy"))
    st.session_state.data = X_test
    st.session_state.i = 0

# 🧠 HAREKET MODELİ
def predict_activity(x):

    x = np.expand_dims(x, axis=0)
    pred = activity_model.predict(x, verbose=0)

    pred_label = np.argmax(pred)

    return f"CLASS {pred_label}"

# 👤 USER MODEL
def predict_user(x):

    x = np.expand_dims(x, axis=0)
    emb = embedding_model.predict(x, verbose=0)[0]

    best_user = None
    best_score = float("inf")

    for user, stored_emb in gallery.items():

        dist = np.linalg.norm(emb - stored_emb)

        print(user, "distance:", dist)

        if dist < best_score:
            best_score = dist
            best_user = user

    print("BEST:", best_user, best_score)

    # 🔥 THRESHOLD DÜZELT
    if best_score > 1.62:
        return "UNKNOWN"

    return best_user


# 🔥 STEP
i = st.session_state.i
signal = st.session_state.data[i]

# 💥 CLEAN SIGNAL
clean_signal = signal[:, 0]
clean_signal = clean_signal[clean_signal != 0]

# 📊 GRAPH
chart.line_chart(pd.DataFrame(clean_signal, columns=["Signal"]))

# 🧠 TAHMİN
move = predict_activity(signal)
user = predict_user(signal)

# 💥 UI
result_box.markdown(f"""
<div style="font-size:90px; font-weight:800;">
{move}
</div>

<div style="font-size:30px; margin-top:20px;">
👤 {user}
</div>
""", unsafe_allow_html=True)

# 🔄 LOOP
st.session_state.i += 1
if st.session_state.i >= len(st.session_state.data):
    st.session_state.i = 0

time.sleep(0.5)
st.rerun()