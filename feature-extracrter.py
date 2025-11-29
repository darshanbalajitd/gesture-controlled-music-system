import sys
import argparse
from pathlib import Path

import cv2
import pandas as pd
from tqdm import tqdm

try:
    import mediapipe as mp
except Exception as e:
    print("mediapipe is required. Install with: pip install mediapipe")
    raise


IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG'}


def iter_label_images(root: Path):
    if not root.exists():
        return
    for label_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
        label = label_dir.name
        files = [f for f in label_dir.rglob('*') if f.suffix in IMAGE_EXTS]
        for f in files:
            yield label, f


def build_header():
    cols = ['label', 'image_path']
    for i in range(21):
        cols.extend([f'x{i}', f'y{i}', f'z{i}'])
    return cols


def extract_one(image_bgr, hands):
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    result = hands.process(image_rgb)
    if not result.multi_hand_landmarks:
        return None
    lm = result.multi_hand_landmarks[0]
    row = []
    for i in range(21):
        p = lm.landmark[i]
        row.extend([p.x, p.y, p.z])
    return row


def run(input_dir: Path, output_csv: Path, min_conf: float):
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=min_conf,
    )

    rows = []
    total = 0
    missed = 0
    for label, path in tqdm(list(iter_label_images(input_dir)), desc='Images'):
        total += 1
        img = cv2.imread(str(path))
        if img is None:
            missed += 1
            continue
        values = extract_one(img, hands)
        if values is None:
            missed += 1
            continue
        rows.append([label, str(path)] + values)

    hands.close()

    if rows:
        df = pd.DataFrame(rows, columns=build_header())
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_csv, index=False)
        print(f"Saved {len(rows)} rows to {output_csv}")
    else:
        print("No landmarks extracted. CSV not created.")

    print(f"Processed: {total}, missed: {missed}, saved: {len(rows)}")


def main(argv=None):
    parser = argparse.ArgumentParser(description='Extract MediaPipe Hands landmarks to CSV')
    parser.add_argument('--input-dir', type=Path, default=Path.cwd() / 'data-preprocessed')
    parser.add_argument('--output-csv', type=Path, default=Path.cwd() / 'landmarks.csv')
    parser.add_argument('--min-conf', type=float, default=0.5)
    args = parser.parse_args(argv)

    print(f"Input dir : {args.input_dir}")
    print(f"Output csv: {args.output_csv}")
    print(f"Min conf  : {args.min_conf}")
    run(args.input_dir, args.output_csv, args.min_conf)


if __name__ == '__main__':
    main()

