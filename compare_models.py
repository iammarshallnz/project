from ultralytics import YOLO

# Load a model

# model = YOLO("./runs/detect/hockey_seg/v1/weights/best.pt")
model = YOLO("HockeyAI_model_weight.pt")  # load a custom model

if __name__ == "__main__":
    # Validate the model
    metrics = model.val(data="labeled/data.yaml", split="val" )  
