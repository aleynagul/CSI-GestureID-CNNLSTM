import streamlit as st
import numpy as np
import pickle
import cv2
from tensorflow.keras.models import load_model

# ---------------------------
# Load Models
# ---------------------------
activity_model = load_model("models/activity_model.h5")
embedding_model = load_model("models/embedding_model.h5")

# ---------------------------
# Load Gallery
# ---------------------------
try:
    with open("gallery.pkl", "rb") as f:
        gallery = pickle.load(f)
except:
    gallery = {}

# ---------------------------
# UI
# ---------------------------
st.title("CSI Gesture & User Recognition System")

menu = st.sidebar.selectbox(
    "Menu",
    ["Live Recognition", "Register New User"]
)

# ---------------------------
# LIVE MODE
# ---------------------------
if menu == "Live Recognition":

    st.header("Live Recognition")

    if st.button("Run Recognition"):

        # Fake CSI input (temporary)
        dummy_input = np.random.rand(1, 300, 90)

        # Activity Prediction
        act_pred = activity_model.predict(dummy_input)
        gesture_id = np.argmax(act_pred)

        # Embedding
        emb = embedding_model.predict(dummy_input)[0]

        if len(gallery) > 0:
            sims = {u: np.dot(emb, gallery[u]) for u in gallery}
            best_user = max(sims, key=sims.get)
        else:
            best_user = "No Users"

        st.success(f"Gesture: {gesture_id}")
        st.info(f"User: {best_user}")

# ---------------------------
# REGISTER MODE
# ---------------------------
if menu == "Register New User":

    st.header("Register New User")

    username = st.text_input("Enter Username")

    if st.button("Register"):
        if username == "":
            st.warning("Please enter a username.")
        else:
            # Fake embedding average (we replace later)
            fake_emb = np.random.rand(128)
            gallery[username] = fake_emb

            with open("gallery.pkl", "wb") as f:
                pickle.dump(gallery, f)

            st.success(f"{username} registered successfully!")