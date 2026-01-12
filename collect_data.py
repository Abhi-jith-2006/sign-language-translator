import cv2
import mediapipe as mp
import csv
import os

# ===================== CONFIG =====================
LABEL = "WELCOME"     # CHANGE THIS FOR EACH GESTURE
SAMPLES = 200
# ==================================================

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

# Force macOS camera backend
cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)

if not cap.isOpened():
    print("ERROR: Camera not accessible")
    exit()

data = []
count = 0

print(f"Collecting data for label: {LABEL}")

while True:
    ret, frame = cap.read()
    if not ret:
        print("ERROR: Frame not read")
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        hand = results.multi_hand_landmarks[0]

        row = []
        for lm in hand.landmark:
            row.extend([lm.x, lm.y, lm.z])

        row.append(LABEL)
        data.append(row)
        count += 1

        mp_draw.draw_landmarks(
            frame,
            hand,
            mp_hands.HAND_CONNECTIONS
        )

    cv2.putText(
        frame,
        f"{LABEL} {count}/{SAMPLES}",
        (10, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow("Collecting Gesture Data", frame)

    if count >= SAMPLES or cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# ===================== SAVE DATA =====================
os.makedirs("data", exist_ok=True)
file_path = "data/gestures.csv"
file_exists = os.path.isfile(file_path)

with open(file_path, "a", newline="") as f:
    writer = csv.writer(f)

    if not file_exists:
        header = [f"f{i}" for i in range(63)] + ["label"]
        writer.writerow(header)

    writer.writerows(data)

print(f"Saved {count} samples for label '{LABEL}'")

