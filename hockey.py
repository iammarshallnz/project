import cv2
import numpy as np
from ultralytics import YOLO
from filterpy.kalman import KalmanFilter, UnscentedKalmanFilter, MerweScaledSigmaPoints



model = YOLO("./runs/detect/hockey_seg/v1/weights/best.pt")
# model = YOLO("./HockeyAI_model_weight.pt")
# Only the classes we care about — update IDs to match your data.yaml order
PLAYER_CLASS_ID = 1
GOALKEEPER_CLASS_ID = 0
REFEREE_CLASS_ID = 3
PUCK_CLASS_ID = 2

PLAYER_CLASSES = {PLAYER_CLASS_ID, GOALKEEPER_CLASS_ID}


# Colours per class (BGR)
CLASS_COLOURS = {
    PLAYER_CLASS_ID:     (255, 100,  50),
    GOALKEEPER_CLASS_ID: (50,  200, 255),
    REFEREE_CLASS_ID:    (0,   220, 220),
}

def draw_mask(frame, mask_bool, colour, alpha=0.4):
    # Resize mask to match frame dimensions
    frame_h, frame_w = frame.shape[:2]
    mask_resized = cv2.resize(
        mask_bool.astype(np.uint8),
        (frame_w, frame_h),
        interpolation=cv2.INTER_NEAREST
    ).astype(bool)

    overlay = frame.copy()
    overlay[mask_resized] = colour
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    mask_uint8 = (mask_resized * 255).astype(np.uint8)
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(frame, contours, -1, colour, 1)

cap = cv2.VideoCapture("./highlight.mp4")
fps = cap.get(cv2.CAP_PROP_FPS * 2)

out = cv2.VideoWriter(
    "./output_seg.mp4",
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
     int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
)


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model.track(frame, persist=True )

    # Construct the Kalman Filter and initialize the variables.
    def fx(x, dt):
        # State transition function.
        F = np.array([[1,0,dt,0],
                      [0,1,0,dt],
                      [0,0,1,0],
                      [0,0,0,1]], np.float32)
        return np.dot(F, x)

    def hx(x):
        # Extract the measurement from the state.
        return np.array(x[:2])
    
    points = MerweScaledSigmaPoints(4, 0.1, 2.0, 1)

    kalman = UnscentedKalmanFilter(dim_x=4, dim_z=2, dt=fps, fx=fx, hx=hx, points=points)
    kalman.x = np.array([1,1,1,1])  # Initial State
    kalman.P = np.array([[2,0,0,0],
                         [0,2,0,0],
                         [0,0,2,0],
                         [0,0,0,2]], np.float32)  # Covariance Matrix
    kalman.R = np.array([[1,0],
                         [0,1]], np.float32)  # Measurement Noise
    kalman.Q = np.array([[1,0,0,0],
                         [0,1,0,0],
                         [0,0,100,0],
                         [0,0,0,100]], np.float32)  # Process Noise

    puck_found = False
    masks_data = results[0].masks
    boxes_data = results[0].boxes

    if boxes_data is not None and boxes_data.id is not None:
        classes  = boxes_data.cls.cpu().numpy().astype(int)
        boxes    = boxes_data.xyxy.cpu().numpy()
        masks_np = masks_data.data.cpu().numpy() if masks_data is not None else None
        for i, (cls_id, box) in enumerate(zip(classes, boxes)):

            # Skip everything except our three targets
            if cls_id not in {PLAYER_CLASS_ID, GOALKEEPER_CLASS_ID,
                               REFEREE_CLASS_ID, PUCK_CLASS_ID}:
                continue

            x1, y1, x2, y2 = box.astype(int)

            # --- Puck ---
            if cls_id == PUCK_CLASS_ID:
                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                r = max(4, (x2 - x1) // 4)
                cv2.circle(frame, (cx, cy), r, (0, 0, 255), 2)
                puck_last_known = {"pos": (cx, cy), "radius": r, "frames_since_seen": 0}
                puck_found = True
                center = (cx, cy)
                print('\nMeasurement:\t', center)

                kalman.predict()  # Predict the ball's position.

                # Draw an ellipse showing the uncertainty of the predicted position.
                center_ = (int(kalman.x[0]), int(kalman.x[1]))
                axis_lengths = (int(kalman.P_prior[0, 0]), int(kalman.P_prior[1, 1]))
                cv2.ellipse(frame, center_, axis_lengths, 0, 0, 360, color=(255, 0, 0))

                if center is not None and radius is not None:
                    cv2.circle(frame, tuple(center), radius, (0,255,0), 2)  # Draw circle around the ball.
                    cv2.circle(frame, tuple(center), 1, (0,255,0), 2)  # Draw the center (not centroid!) of the ball.

                    # The Kalman filter expects the x,y coordinates in a 2D array.
                    measured = np.array([center[0], center[1]], dtype="float32")
                    # Update the Kalman filter with the current ball location if we have it.
                    kalman.update(measured)
                    print('Estimate:\t', np.int32(kalman.x))
                    print('Variance:\t', np.diag(kalman.P))
                continue

            # --- Players / goalkeeper / referee ---
            colour = CLASS_COLOURS.get(cls_id, (180, 180, 180))

            if masks_np is not None and i < len(masks_np):
                mask_bool = masks_np[i] > 0.5
                draw_mask(frame, mask_bool, colour)
            else:
                # Fallback to ellipse if no mask
                feet_x = int((x1 + x2) / 2)
                radius = int((x2 - x1) / 4)
                cv2.ellipse(frame, (feet_x, y2), (radius, radius // 2),
                            0, 0, 360, colour, 2)

    
    out.write(frame)
    cv2.imshow("hockey", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()