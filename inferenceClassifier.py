import pickle
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Charger le modèle
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# Initialiser la capture vidéo
cap = cv2.VideoCapture(0)


# Initialiser Mediapipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Dictionnaire des étiquettes
labels_dict = {0: 'ك', 1: 'ل', 2: 'ي', 3: 'ا', 4: 'د'}

# Charger la police arabe
font_path = "fonts/Rubik-VariableFont_wght.ttf"  
font = ImageFont.truetype(font_path, 72)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    data_aux = []

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Calculer les coordonnées du rectangle englobant
            x_min = min(landmark.x for landmark in hand_landmarks.landmark)
            y_min = min(landmark.y for landmark in hand_landmarks.landmark)
            x_max = max(landmark.x for landmark in hand_landmarks.landmark)
            y_max = max(landmark.y for landmark in hand_landmarks.landmark)

            for landmark in hand_landmarks.landmark:
                data_aux.append(landmark.x)
                data_aux.append(landmark.y)

            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

            # Dessiner un rectangle autour de la main
            cv2.rectangle(frame, (int(x_min * frame.shape[1]), int(y_min * frame.shape[0])), 
                      (int(x_max * frame.shape[1]), int(y_max * frame.shape[0])), 
                      (255, 0, 255), 2)

        if len(data_aux) == 42: 
            prediction = model.predict([np.asarray(data_aux)])
            predicted_character = labels_dict[int(prediction[0])]
            
            # Ajouter le texte en arabe à l'image
            frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(frame_pil)
            draw.text((10, 10), predicted_character, font=font, fill=(0, 255, 0, 0))

            # Convertir PIL image en OpenCV image
            frame = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)

        else:
            print(f"Unexpected number of features: {len(data_aux)}")

    cv2.imshow('Hand Gesture Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
