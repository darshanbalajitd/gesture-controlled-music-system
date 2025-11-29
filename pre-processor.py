import os
import cv2
import shutil
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# ================== CONFIG ==================
DATASET_ROOT = Path.cwd() / "data"
OUTPUT_ROOT  = Path.cwd() / "data-preprocessed"

IMAGE_SIZE = (224, 224)
IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG'}
MAX_WORKERS = 16  # Adjust for your CPU cores

# ============================================

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def _on_rm_error(func, path, exc_info):
    try:
        os.chmod(path, 0o777)
        func(path)
    except Exception:
        pass

def resize_and_save_image(src: Path, dst: Path, size=IMAGE_SIZE):
    img = cv2.imread(str(src))
    if img is None:
        return False
    resized = cv2.resize(img, size)
    cv2.imwrite(str(dst), resized)
    return True

# (videos removed)

# ================== IMAGE PROCESSING ==================
def process_images(dataset_root: Path, output_root: Path):
    images_root = dataset_root
    if not images_root.exists():
        print(f"No image folder found at {images_root}. Skipping images.")
        return 0

    gesture_dirs = [p for p in images_root.iterdir() if p.is_dir()]
    all_tasks = []

    for gesture_dir in gesture_dirs:
        gesture_name = gesture_dir.name
        out_gesture_dir = output_root / gesture_name
        ensure_dir(out_gesture_dir)
        image_files = [f for f in gesture_dir.rglob('*') if f.suffix.lower() in IMAGE_EXTS]
        for img_path in image_files:
            dst_path = out_gesture_dir / img_path.name
            all_tasks.append((img_path, dst_path))

    print(f"\n🖼️ Found {len(all_tasks)} image files across {len(gesture_dirs)} gesture classes.")

    total_success = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(resize_and_save_image, src, dst) for src, dst in all_tasks]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Images"):
            if future.result():
                total_success += 1

    print(f"✅ Finished processing images. Total resized: {total_success}")
    return total_success

# ================== VIDEO PROCESSING ==================
# (videos removed)

# ================== MAIN ==================
def main():
    print(f"📂 Dataset root: {DATASET_ROOT}")
    print(f"📁 Output root : {OUTPUT_ROOT}")
    if OUTPUT_ROOT.exists():
        print("Clearing old processed data...")
        shutil.rmtree(OUTPUT_ROOT, onexc=_on_rm_error)
    ensure_dir(OUTPUT_ROOT)

    img_count = process_images(DATASET_ROOT, OUTPUT_ROOT)

    print("\n===== SUMMARY =====")
    print(f"Total images processed : {img_count}")
    print("===================")
    print(f"Processed data saved to: {OUTPUT_ROOT}")

if __name__ == "__main__":
    main()
