from ultralytics import YOLO
import torch

if __name__ == "__main__":
    print(torch.cuda.is_available())      # Should be True
    print(torch.cuda.get_device_name(0))  # Should show your GPU name
    # Load model
    model = YOLO("./HockeyAI_model_weight.pt")
    # Train model on custom dataset
    results = model.train(
        data="./labeled/data.yaml",
        epochs=100,
        device=0,
        imgsz=640,
        batch=16,
        exist_ok=True,
        project="hockey_seg",   # saves to hockey_seg/train/ instead of runs/
        name="v1",
        patience=20,            # stops early if no improvement for 20 epochs
        save_period=10,         # checkpoint every 10 epochs
        workers=8,              # parallel data loading
    )
