import cv2
import numpy as np
from ultralytics import YOLO

model = YOLO("./runs/detect/hockey_seg/v1/weights/best.pt")

PLAYER_CLASS_ID     = 1
GOALKEEPER_CLASS_ID = 0
REFEREE_CLASS_ID    = 3
PUCK_CLASS_ID       = 2

PLAYER_CLASSES = {PLAYER_CLASS_ID, GOALKEEPER_CLASS_ID}

CLASS_COLOURS = {
    PLAYER_CLASS_ID:     (255, 100,  50),
    GOALKEEPER_CLASS_ID: (50,  200, 255),
    REFEREE_CLASS_ID:    (0,   220, 220),
}

CLASS_NAMES = {
    PLAYER_CLASS_ID:     "Player",
    GOALKEEPER_CLASS_ID: "Goalkeeper",
    REFEREE_CLASS_ID:    "Referee",
    PUCK_CLASS_ID:       "Puck",
}

cap = cv2.VideoCapture("./highlight.mp4")
fps = cap.get(cv2.CAP_PROP_FPS  * 2)  # Fixed: removed * 2 (was doubling fps incorrectly)

out = cv2.VideoWriter(
    "./output_seg.mp4",
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
     int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
)

puck_last_known = None

def get_ice_region(frame):
    """Detect the white ice surface and return its top boundary."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # White/ice colour range in HSV
    lower_white = np.array([0,   0,  180])
    upper_white = np.array([180, 40, 255])
    
    mask = cv2.inRange(hsv, lower_white, upper_white)
    
    # Remove noise
    kernel = np.ones((15, 15), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # Find the largest white region (the ice)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
    
    largest = max(contours, key=cv2.contourArea)
    
    # Filter out small detections (jerseys, boards etc)
    if cv2.contourArea(largest) < (frame.shape[0] * frame.shape[1] * 0.1):
        return None
    
    x, y, w, h = cv2.boundingRect(largest)
    return x, y, w, h


# Smoothing to prevent jitter between frames
ice_region_smooth = None
SMOOTH = 0.85  # Higher = more stable but slower to update

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    fh, fw = frame.shape[:2]

    # --- Detect ice region ---
    ice_rect = get_ice_region(frame)

    if ice_rect is not None:
        x, y, w, h = ice_rect
        if ice_region_smooth is None:
            ice_region_smooth = ice_rect
        else:
            # Exponential smoothing so the crop doesn't jump around
            ice_region_smooth = (
                int(ice_region_smooth[0] * SMOOTH + x * (1 - SMOOTH)),
                int(ice_region_smooth[1] * SMOOTH + y * (1 - SMOOTH)),
                int(ice_region_smooth[2] * SMOOTH + w * (1 - SMOOTH)),
                int(ice_region_smooth[3] * SMOOTH + h * (1 - SMOOTH)),
            )

    # Fallback to bottom half if ice not detected
    if ice_region_smooth is None:
        crop_x, crop_y, crop_w, crop_h = 0, fh // 2, fw, fh // 2
    else:
        crop_x, crop_y, crop_w, crop_h = ice_region_smooth
        # Add a small padding
        pad = 80
        crop_x = max(0,  crop_x - pad)
        crop_y = max(0,  crop_y - pad)
        crop_w = min(fw, crop_w + pad * 2)
        crop_h = min(fh, crop_h + pad * 2)

    # Crop to ice region
    roi = frame[crop_y:crop_y + crop_h, crop_x:crop_x + crop_w]
    results = model.track(roi, tracker="bytetrack.yaml", persist=True, verbose=False)

    puck_found = False
    masks_data = results[0].masks
    boxes_data = results[0].boxes

    if boxes_data is not None and boxes_data.id is not None:
        classes   = boxes_data.cls.cpu().numpy().astype(int)
        boxes     = boxes_data.xyxy.cpu().numpy()
        track_ids = boxes_data.id.cpu().numpy().astype(int)
        masks_np  = masks_data.data.cpu().numpy() if masks_data is not None else None

        for i, (cls_id, box, track_id) in enumerate(zip(classes, boxes, track_ids)):

            if cls_id not in {PLAYER_CLASS_ID, GOALKEEPER_CLASS_ID,
                              REFEREE_CLASS_ID, PUCK_CLASS_ID}:
                continue

            x1, y1, x2, y2 = box.astype(int)

            # Offset back to full frame coordinates
            x1 += crop_x;  x2 += crop_x
            y1 += crop_y;  y2 += crop_y

            # --- Puck ---
            if cls_id == PUCK_CLASS_ID:
                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                r = max(4, (x2 - x1) // 4)
                cv2.circle(frame, (cx, cy), r, (0, 0, 255), -1)
                cv2.putText(frame, "Puck", (cx - 15, cy - r - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                puck_last_known = {"pos": (cx, cy), "radius": r, "frames_since_seen": 0}
                puck_found = True
                continue

            colour = CLASS_COLOURS.get(cls_id, (180, 180, 180))

            if masks_np is not None and i < len(masks_np):
                mask = masks_np[i]
                mask_resized = cv2.resize(mask, (crop_w, crop_h), interpolation=cv2.INTER_NEAREST)
                full_mask = np.zeros((fh, fw), dtype=np.float32)
                full_mask[crop_y:crop_y + crop_h, crop_x:crop_x + crop_w] = mask_resized
                colored_mask = np.zeros_like(frame)
                colored_mask[full_mask > 0.5] = colour
                frame = cv2.addWeighted(frame, 1.0, colored_mask, 0.35, 0)

            feet_x = int((x1 + x2) / 2)
            radius = int((x2 - x1) / 4)
            cv2.ellipse(frame, (feet_x, y2), (radius, radius // 2), 0, 0, 360, colour, 2)

            label = f"{CLASS_NAMES.get(cls_id, 'Unknown')} #{track_id}"
            label_y = max(y1 - 8, crop_y + 15)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            cv2.rectangle(frame, (x1, label_y - th - 4), (x1 + tw, label_y), colour, -1)
            cv2.putText(frame, label, (x1, label_y - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

    # Debug: draw the ice boundary
    if ice_region_smooth:
        cv2.rectangle(frame,
                      (crop_x, crop_y),
                      (crop_x + crop_w, crop_y + crop_h),
                      (0, 255, 0), 2)

    # Ghost puck
    if not puck_found and puck_last_known is not None:
        puck_last_known["frames_since_seen"] += 1
        if puck_last_known["frames_since_seen"] < 30:
            cx, cy = puck_last_known["pos"]
            r = puck_last_known["radius"]
            cv2.circle(frame, (cx, cy), r, (0, 0, 180), 1)
            cv2.putText(frame, "Puck?", (cx - 18, cy - r - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 180), 1)

    out.write(frame)
    cv2.imshow("hockey", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()