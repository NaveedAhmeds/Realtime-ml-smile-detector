# Face and Smile Detector
import cv2
import os
from datetime import datetime

# 1. Capture video frames from a webcam.

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
output_folder = "all_the_smiles"
os.makedirs(output_folder, exist_ok=True)

cap = cv2.VideoCapture(1)
if not cap.isOpened():
    print("Error: Cannot open webcam")
    exit()

print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Cannot read frame")
        break

    # 2. Convert frames to grayscale for detection.
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 3. Detect faces using haarcascade_frontalface_default.xml.
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        # 4. Crop each detected face region.
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        # 5. Detect smiles within the cropped face region using haarcascade_smile.xml.
        smiles = smile_cascade.detectMultiScale(roi_gray, scaleFactor=1.9, minNeighbors=20)

        if len(smiles) > 0:
            label = "Smiling :)"
            color = (0, 255, 0)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S%f")
            face_filename = os.path.join(output_folder, f"smile_{timestamp}.jpg")
            cv2.imwrite(face_filename, roi_color)
        else:
            label = "No smile"
            color = (0, 0, 255)

        # 6. Draw rectangles around detected faces and smiles.
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, label, (x, y - 10),
                    cv2.FONT_HERSHEY_DUPLEX, 0.8, color, 2, cv2.LINE_AA)

        for (sx, sy, sw, sh) in smiles:
            cv2.rectangle(roi_color, (sx, sy), (sx+sw, sy+sh), color, 2)

    # 7. Display results in real time and save cropped faces with smiles to disk.
    cv2.imshow('Face & Smile Detector', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
