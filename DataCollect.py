import os
import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math

# Répertoire pour stocker les images collectées
DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Nombre de classes et taille du dataset
number_of_classes = 5
dataset_size = 150

# Initialisation de la capture vidéo et du détecteur de mains
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)

offset = 20
imgSize = 300

for j in range(number_of_classes):
    class_dir = os.path.join(DATA_DIR, str(j))
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)
    print(f'Collecting data for class {j}')
    
    # Attendre l'utilisateur pour commencer la collecte de données
    while True:
        ret, frame = cap.read()
        cv2.putText(frame, 'Ready? Press "Q"!', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
        cv2.imshow('frame', frame)
        if cv2.waitKey(25) == ord('q'):
            break

    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        original_frame = frame.copy()
        hands, frame = detector.findHands(frame)
        
        if hands:
            hand = hands[0]
            x, y, w, h = hand['bbox']

            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
            y1, y2 = max(0, y - offset), min(original_frame.shape[0], y + h + offset)
            x1, x2 = max(0, x - offset), min(original_frame.shape[1], x + w + offset)

            imgCrop = original_frame[y1:y2, x1:x2]

            if imgCrop.size > 0:
                imgCropShape = imgCrop.shape

                aspectRatio = h / w
                if aspectRatio > 1:
                    k = imgSize / h
                    wCal = math.ceil(k * w)
                    imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                    wGap = math.ceil((imgSize - wCal) / 2)
                    imgWhite[:, wGap:wCal + wGap] = imgResize
                else:
                    k = imgSize / w
                    hCal = math.ceil(k * h)
                    imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                    hGap = math.ceil((imgSize - hCal) / 2)
                    imgWhite[hGap:hCal + hGap, :] = imgResize

            cv2.imshow('imgCrop', imgCrop)
            cv2.imshow('imgWhite', imgWhite)
            cv2.imwrite(os.path.join(class_dir, f'{counter}.jpg'), imgWhite)
            counter += 1

        cv2.imshow('frame', frame)
        cv2.waitKey(25)

cap.release()
cv2.destroyAllWindows()
