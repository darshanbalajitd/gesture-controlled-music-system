# Gesture-Controlled Music Player

A gesture-based project that detects static hand gestures from images and uses them to control music playback. It processes images, extracts hand landmarks with MediaPipe, trains a lightweight MLP classifier, and runs a realtime detector that maps detected gestures to audio playback actions.

## Stack
- Python (3.x)
- OpenCV (`cv2`)
- MediaPipe Hands
- pandas
- tqdm
- PyTorch
- scikit-learn
- joblib
- VLC (via `python-vlc`) for audio playback

## Setup (Virtual Environment)
- Create and activate a virtual environment on Windows:
  ```
  python -m venv venv
  .\venv\Scripts\activate
  python -m pip install --upgrade pip
  ```
- Install required packages:
  ```
  pip install opencv-python mediapipe==0.10.14 protobuf==3.20.* pandas tqdm scikit-learn joblib python-vlc torch
  ```
  - If PyTorch fails to install, use the official wheels for your platform (CPU-only example):
    ```
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    ```

## Folder Structure
```
<project-root>/
  data/
    <label>/
      sample.jpg
      ...
  data-preprocessed/
    <label>/
      sample.jpg
      ...
  music/
    hindi/
      song1.mp3
      ...
    indie/
      ...
    k-pop/
      ...
    phonk/
      ...
    rap/
      ...
  sounds/
    playing_audio.mp3
    pausing_audio.mp3
    stop_audio.mp3
    next_audio.mp3
    previous_audio.mp3
    increase_audio.mp3
    decrease_audio.mp3
```

## Workflow
1) Preprocess images to a uniform size
   - Run:
     ```
     python pre-processor.py
     ```
   - Input: `./data/<label>/*.jpg|*.png` (recursively)
   - Output: `./data-preprocessed/<label>/...` resized to `224x224`

2) Extract MediaPipe hand landmarks to CSV
   - Run:
     ```
     python feature-extracrter.py
     ```
   - Input: `./data-preprocessed/<label>/...`
   - Output: `./landmarks.csv` with columns:
     - `label`, `image_path`, then `x0,y0,z0,...,x20,y20,z20` for 21 landmarks

3) Train the Lightweight MLP model
   - Run:
     ```
     python train_mlp.py
     ```
   - Artifacts written to `./models`:
     - `mlp_state.pth` (model weights + architecture config)
     - `scaler.pkl` (feature standardizer)
     - `label_map.json` (class names)
     - `metrics_mlp.csv` (epoch metrics)

4) Run the realtime detector
   - Run:
     ```
     python realtime_mlp.py
     ```
   - Opens the webcam, extracts landmarks, scales features, infers gesture class, and plays corresponding audio prompts and music.

## Gestures
- Project uses 9 static gestures:
  - Genre selection: `genre-1`, `genre-2`, `genre-3`, `genre-4`, `genre-5`
  - Control: `next`, `previous`, `volume-up`, `volume-down`
- Each gesture category should have a representative sample image during setup.

## Data Requirements
- Create custom gesture datasets for your labels under `./data/<label>/...`.
- Target at least ~200 images per gesture, covering diverse conditions:
  - Multiple hands and users
  - Varying backgrounds and lighting
  - Slight pose variations
- After collection, run preprocessing and landmark extraction before training.

## Customization
- Music: Replace or add songs by copying files into the appropriate `./music/<genre-folder>/`.
- Sounds: Generate or replace prompt audio (`playing_audio`, `pausing_audio`, `stop_audio`, `next_audio`, `previous_audio`, `increase_audio`, `decrease_audio`) using tools like ElevenLabs or any text-to-speech model.

## How It Works
- Image Preprocessing: Normalizes all images to a fixed resolution to simplify downstream processing.
- Landmark Extraction: MediaPipe Hands produces 21 normalized landmarks per detected hand; each landmark includes `(x, y, z)`.
- Model Training (MLP): A small neural network learns to map the 63-dim feature vector (`21 * 3`) to gesture classes; features are standardized with a scaler.
- Realtime Detection: The webcam frames are processed on-the-fly. For each frame, the detector:
  - Extracts landmarks
  - Scales features
  - Predicts the gesture class via the trained MLP
  - Applies stability and thresholding, then plays prompt audio and triggers music actions.
  - Stability: gestures are detected once per 8-frame window and a 5-second timeout prevents repeated triggers.

## Notes
- Ensure required Python packages are installed (OpenCV, MediaPipe, PyTorch, scikit-learn, pandas, tqdm, joblib, python-vlc).
- If MediaPipe import errors occur due to protobuf version mismatch, install compatible versions:
  - `pip install mediapipe==0.10.14 protobuf==3.20.*`
