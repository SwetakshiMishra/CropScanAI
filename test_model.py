#testing model
import cv2
import numpy as np
from tensorflow.keras.models import load_model

model = load_model("pearl_millet_ergot_model.h5")

# Load test image
img = cv2.imread("test.jpg")
img = cv2.resize(img, (224, 224))
img = img.astype("float32") / 255.0
img = np.expand_dims(img, axis=0)

pred = model.predict(img)[0][0]

if pred > 0.5:
    print("Diseased: Ergot")
else:
    print("Healthy")
