import sys
import os
import cv2
from pathlib import Path

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.detectors.mobilevit import MobileViTDetector
from src.config import PROJECT_ROOT

def main():
    print("🚀 Starting MobileViT Training Pipeline...")
    
    # 1. Load Data
    images, labels = [], []
    train_dir = PROJECT_ROOT / "data/artroom/bird/yolo/train/images"
    
    print(f"📂 Scanning {train_dir}...")
    for img_file in train_dir.glob("*.png"):
        img = cv2.imread(str(img_file))
        if img is None: continue
        
        fname = img_file.name.lower()
        if "bird" in fname:
            images.append(img)
            labels.append("bird")
        elif any(x in fname for x in ["room", "wall", "floor", "empty"]):
            images.append(img)
            labels.append("background")

    print(f"📊 Data Summary: {labels.count('bird')} Birds, {labels.count('background')} Backgrounds")
    
    if not images:
        print("❌ Error: No images found.")
        return

    # 2. Initialize & Train
    detector = MobileViTDetector()
    detector.train_head(images, labels)
    
    # 3. Sanity Check
    print("\n🔎 Sanity Check (Image 0):")
    lbl, conf, ms = detector.predict(images[0])
    print(f"   Result: {lbl} | Conf: {conf:.2%} | Time: {ms:.2f}ms")

if __name__ == "__main__":
    main()