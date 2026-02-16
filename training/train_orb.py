import sys
import os
import cv2
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.detectors.orb import ORBDetector
from src.config import PROJECT_ROOT

def main():
    print("🚀 Starting ORB 'Training' (Reference Extraction)...")
    
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
        # We don't strictly need background images for ORB training, 
        # but we load them to match the pattern.
        elif any(x in fname for x in ["room", "wall", "floor", "empty"]):
            images.append(img)
            labels.append("background")

    bird_count = labels.count('bird')
    print(f"📊 Found {bird_count} Bird images to scan for best reference.")
    
    if bird_count == 0:
        print("❌ Error: No bird images found! Cannot create reference.")
        return

    # 2. Initialize & Train
    # This will automatically find the best image and save 'orb_reference.pkl'
    detector = ORBDetector()
    detector.train(images, labels)
    
    # 3. Sanity Check
    print("\n🔎 Sanity Check (Testing on the first bird image):")
    # Find first bird image
    first_bird = next(img for img, lbl in zip(images, labels) if lbl == 'bird')
    
    lbl, conf, ms = detector.predict(first_bird)
    print(f"   Result: {lbl} | Conf: {conf:.0%} | Time: {ms:.3f}ms")
    print("   (Note: ORB should be extremely fast, < 1ms)")

if __name__ == "__main__":
    main()