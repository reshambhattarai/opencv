import cv2
from cvzone.HandTrackingModule import HandDetector
import tensorflow as tf
import numpy as np
import math

cap = cv2.VideoCapture(1)

detector = HandDetector(maxHands=1)

offset = 20
imgSize = 400

# Load the labels from the file
with open("ourlabels.txt", "r") as file:
    labels = [line.strip().split(" ")[1] for line in file]

# Load the model
model = tf.keras.models.load_model("ourmodel.h5")

while True:
    success, img = cap.read()
    imgOutput = img.copy()

    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

        # Check if imgCrop is empty
        if imgCrop.size == 0:
            continue

        imgCropShape = imgCrop.shape

        aspectRatio = h / w
        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap:wCal + wGap, :] = imgResize
        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hCal + hGap, :, :] = imgResize

        if imgWhite.size == 0:
            continue

        # Resize imgWhite to (400, 400)
        imgWhite = cv2.resize(imgWhite, (400, 400))

        # Preprocess the image
        imgWhite = imgWhite / 255.0
        imgWhite = np.expand_dims(imgWhite, axis=0)

        # Make prediction
        prediction = model.predict(imgWhite)
        index = np.argmax(prediction)

        label = labels[index]
        print(prediction, label)
        cv2.putText(imgOutput, label, (x, y - 20), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 0, 255), 2)
        cv2.rectangle(imgOutput, (x, y), (x + w, y + h), (255, 0, 255), 4)
        cv2.imshow("Image Crop", imgCrop)
        cv2.imshow("Image White", imgWhite[0])

    cv2.imshow("Image", imgOutput)
    cv2.waitKey(1)
