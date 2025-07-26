import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelBinarizer
import pickle

# Load trained model
model = load_model("sign_language_model.h5")

# Setup label binarizer manually (since we trained on 0–23 excluding J)
labels = [i for i in range(24)]  # A-Y excluding J
lb = LabelBinarizer()
lb.fit(labels)

# Setup MediaPipe
mpHands = mp.solutions.hands
hands = mpHands.Hands(static_image_mode=False,
                      max_num_hands=1,
                      min_detection_confidence=0.75)
mpDraw = mp.solutions.drawing_utils

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    if not success:
        break

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    predicted_letter = ""

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            # Get bounding box
            h, w, c = img.shape
            x_list = []
            y_list = []
            for lm in handLms.landmark:
                x_list.append(int(lm.x * w))
                y_list.append(int(lm.y * h))
            x_min, x_max = min(x_list), max(x_list)
            y_min, y_max = min(y_list), max(y_list)

            # Slight padding
            x_min = max(x_min - 20, 0)
            y_min = max(y_min - 20, 0)
            x_max = min(x_max + 20, w)
            y_max = min(y_max + 20, h)

            # Extract hand image
            hand_img = img[y_min:y_max, x_min:x_max]
            try:
                hand_img = cv2.cvtColor(hand_img, cv2.COLOR_BGR2GRAY)
                hand_img = cv2.resize(hand_img, (28, 28))
                hand_img = hand_img.reshape(1, 28, 28, 1) / 255.0

                prediction = model.predict(hand_img)
                class_id = lb.classes_[np.argmax(prediction)]
                predicted_letter = chr(class_id + 65 if class_id < 9 else class_id + 66)  # skip J

            except:
                predicted_letter = "?"

            # Draw landmarks
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

            # Draw predicted letter
            cv2.putText(img, f'Predicted: {predicted_letter}', (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

    cv2.imshow("Sign Language Recognition", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
