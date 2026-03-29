import numpy as np
import time
from tensorflow.keras.models import load_model
import os

model = load_model("models/embedding_model.h5")

aley = np.load("Aley.npy")

last_time = 0

# eski veriyi sil
if os.path.exists("sequence.npy"):
    os.remove("sequence.npy")

while True:
    try:
        # dosya yoksa bekle
        if not os.path.exists("sequence.npy"):
            time.sleep(0.2)
            continue

        current_time = os.path.getmtime("sequence.npy")

        # yeni veri gelmiş mi kontrol
        if current_time != last_time:
            last_time = current_time

            data = np.load("sequence.npy")
            data = np.pad(data, ((0,0),(0,270),(0,27)))

            embedding = model.predict(data)[0]

            similarity = np.dot(embedding, aley) / (
                np.linalg.norm(embedding) * np.linalg.norm(aley)
            )

            print("Similarity:", round(similarity, 3))

            if similarity > 0.9995:
                print("USER: ALEY")
            else:
                print("UNKNOWN")

        time.sleep(0.2)

    except Exception as e:
        print("ERROR:", e)