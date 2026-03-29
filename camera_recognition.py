import cv2
import mediapipe as mp
import numpy as np  

sequence = []
SEQ_LEN = 30

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("KAMERA AÇILAMADI ❌")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for handLms in result.multi_hand_landmarks:

            mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

            landmarks = []
            for lm in handLms.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])

            sequence.append(landmarks)

            if len(sequence) == SEQ_LEN:
                input_data = np.array(sequence).reshape(1, SEQ_LEN, 63)

                np.save("sequence.npy", input_data)

                sequence = []
    else:
      sequence = []  #El yoksa sıfırla

    cv2.imshow("Hand Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()