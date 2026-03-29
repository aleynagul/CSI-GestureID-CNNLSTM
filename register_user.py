import numpy as np
from tensorflow.keras.models import load_model

model = load_model("models/embedding_model.h5")

data = np.load("sequence.npy")
data = np.load("eren_sequence.npy")

#CSI modele uydur
data = np.pad(data,((0,0,),(0,270),(0,27)))

embedding = model.predict(data)[0]

np.save("Aley.npy",embedding)
np.save("Eren.npy", embedding)
print("User registered as Aley")