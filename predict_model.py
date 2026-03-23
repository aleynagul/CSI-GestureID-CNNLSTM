import numpy as np
import time
from tensorflow.keras.models import load_model

model = load_model("models/embeddig_model.h5")

while True:
    try:
        data = np.load("sequence.npy")

        data = np.pad(data, ((0,0),(0,270),(0,27)))

        pred = model.predict(data)
        gesture = pred.argmax()

        print("GESTURE:", gesture)

        time.sleep(1)  

    except:
        pass