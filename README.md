# Realtime ML Smile Detector

Real-time smile detection pipeline using Python, OpenCV, and Haar cascades. The system ingests a live webcam stream, performs face localization, constrains smile detection to a face ROI, and logs positive samples to disk for downstream ML experiments and dataset curation.

![smile](https://github.com/user-attachments/assets/cb9727da-2d8b-4ad4-a55a-6e3a1ab6b1da)


---

## System overview

At a high level, the detector implements a low-latency CV pipeline:

```text
Webcam → BGR frame → Grayscale
       → Face detection (Haar)
       → Face ROI crop
       → Smile detection (Haar on ROI)
       → Label + rectangles overlay
       → Optional: save smiling face to disk
```

### Key design points:

- Cascade-based detection for faces and smiles, using  
  `haarcascade_frontalface_default.xml` and `haarcascade_smile.xml`.

- ROI-constrained smile search so classification runs only inside the  
  detected face region (less noise, fewer false positives).

- Timestamped sample logging into `all_the_smiles/` whenever a smile  
  is detected, creating a small but useful labeled dataset of positives.

---

## Technical details

### Input

- Webcam frames from `cv2.VideoCapture(0)`
- BGR → grayscale conversion for cascade detectors

---

### Face detection

```python
face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
```

Returns `(x, y, w, h)` bounding boxes for each face per frame.

---

### Smile detection (inside ROI)

For each face, extract:

```python
roi_gray = gray[y:y+h, x:x+w]
roi_color = frame[y:y+h, x:x+w]
```

Run:

```python
smiles = smile_cascade.detectMultiScale(
    roi_gray,
    scaleFactor=1.9,
    minNeighbors=20
)
```

---

### Labeling & visualization

If `len(smiles) > 0` → label **"Smiling :)"** in green  
Else → **"Neutral"** in red  

Draw:

- Blue rectangle around the face
- Optional green/red rectangles around smile boxes in `roi_color`

---

### Data logging

On a smile:

1. Generate timestamp: `YYYYMMDD_HHMMSSffffff`
2. Save `roi_color` as:

```bash
all_the_smiles/smile_<timestamp>.jpg
```

This gives you:

- A live UX layer (overlays on the camera feed), and  
- A data layer (curated positive samples) you can later use to train a  
  learned classifier or compare against the cascade baseline.

---

## Run locally

```bash
pip install -r requirements.txt
python smile_detector.py
# press 'q' to quit
```
