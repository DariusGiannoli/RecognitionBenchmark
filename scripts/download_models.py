import torch
import sys
from pathlib import Path
import timm
from ultralytics import YOLO
from torchvision import models

FILE_PATH = Path(__file__).resolve()
PROJECT_ROOT = FILE_PATH.parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.config import MODEL_DIR, MODEL_PATHS

print(f"⬇️ Downloading models to: {MODEL_DIR}\n")

print("1️⃣ Downloading YOLOv8 Nano...")
model = YOLO('yolov8n.pt')
src_path = Path('yolov8n.pt')
if src_path.exists():
    src_path.rename(MODEL_PATHS['yolo'])
    print(f"✅ Saved to {MODEL_PATHS['yolo']}")

print("\n2️⃣ Downloading MobileNetV3...")
mobilenet = models.mobilenet_v3_small(weights='DEFAULT')
torch.save(mobilenet.state_dict(), MODEL_PATHS['mobilenet'])
print(f"✅ Saved to {MODEL_PATHS['mobilenet']}")

print("\n3️⃣ Downloading ResNet-18...")
resnet = models.resnet18(weights='DEFAULT')
torch.save(resnet.state_dict(), MODEL_PATHS['resnet'])
print(f"✅ Saved to {MODEL_PATHS['resnet']}")

print("\n4️⃣ Downloading MobileViT-XXS...")
mobilevit = timm.create_model('mobilevit_xxs.cvnets_in1k', pretrained=True)
torch.save(mobilevit.state_dict(), MODEL_DIR / "mobilevit_xxs.pth")
print(f"✅ Saved to {MODEL_DIR / 'mobilevit_xxs.pth'}")

print("\n🎉 All models downloaded successfully.")