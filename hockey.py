import cv2
import numpy as np
from ultralytics import YOLO
from filterpy.kalman import KalmanFilter

import torch



model = YOLO("./runs/detect/hockey_seg/v1/weights/best.pt")

device = "cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


# HockeyAi has classes we do not want 
PLAYER_CLASS_ID     = 4
GOALKEEPER_CLASS_ID = 3
REFEREE_CLASS_ID    = 6
PUCK_CLASS_ID       = 5


CLASS_COLOURS = {
    PLAYER_CLASS_ID:     (255, 100,  50),
    GOALKEEPER_CLASS_ID: (50,  200, 255),
    REFEREE_CLASS_ID:    (0,   220, 220),
}

CLASS_NAMES = {
    PLAYER_CLASS_ID:     "player",
    GOALKEEPER_CLASS_ID: "goaltender",
    REFEREE_CLASS_ID:    "referee",
    PUCK_CLASS_ID:       "puck",
}

video = "./highlight.mp4"

process_noise = 0.5

similarity_thresh = 0.7


VALID_CLASSES = {PLAYER_CLASS_ID, GOALKEEPER_CLASS_ID, 
                 REFEREE_CLASS_ID, PUCK_CLASS_ID}

cap = cv2.VideoCapture(video)
fps = cap.get(cv2.CAP_PROP_FPS)
TARGET_FPS = min(60, fps)  
FRAME_DELAY = int(1000 / TARGET_FPS)
out = cv2.VideoWriter(
    "./output_seg.mp4",
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
     int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
)


def make_kalman_filter(x1, y1, x2, y2):

    kf = KalmanFilter(dim_x=8, dim_z=4)
    # [cx, cy, w, h, vx, vy, vw, vh]
    # State transition matrix - constant velocity model
    kf.F = np.array([
        [1,0,0,0, 1,0,0,0],  # cx += vx
        [0,1,0,0, 0,1,0,0],  # cy += vy
        [0,0,1,0, 0,0,1,0],  # w  += vw
        [0,0,0,1, 0,0,0,1],  # h  += vh
        [0,0,0,0, 1,0,0,0],
        [0,0,0,0, 0,1,0,0],
        [0,0,0,0, 0,0,1,0],
        [0,0,0,0, 0,0,0,1],
    ], dtype=float)

    # Measurement matrix - we only observe cx, cy, w, h
    kf.H = np.array([
        [1,0,0,0, 0,0,0,0],
        [0,1,0,0, 0,0,0,0],
        [0,0,1,0, 0,0,0,0],
        [0,0,0,1, 0,0,0,0],
    ], dtype=float)

    kf.R *= 10    # Measurement noise bigger = trust less
    kf.P *= 100   # Covariance Matrix
    kf.Q *= process_noise   # process noise

    # center x and y with height and width 
    cx = (x1 + x2) / 2 
    cy = (y1 + y2) / 2
    w  = x2 - x1
    h  = y2 - y1
    kf.x = np.array([[cx], [cy], [w], [h],
                      [0],  [0],  [0], [0]], dtype=float)
    return kf


def kf_to_box(kf):
    """Convert Kalman filter state back to x1,y1,x2,y2."""
    cx, cy, w, h = kf.x[0,0], kf.x[1,0], kf.x[2,0], kf.x[3,0]
    x1 = int(cx - w / 2)
    y1 = int(cy - h / 2)
    x2 = int(cx + w / 2)
    y2 = int(cy + h / 2)
    return x1, y1, x2, y2


def clamp_box(x1, y1, x2, y2, frame_shape):
    h, w = frame_shape[:2]
    return (max(0, min(x1, w)), max(0, min(y1, h)),
            max(0, min(x2, w)), max(0, min(y2, h)))

def draw_dashed_rectangle(frame, pt1, pt2, colour, gap=8, thickness=2):
    x1, y1 = pt1
    x2, y2 = pt2
    for x in range(x1, x2, gap * 2):
        cv2.line(frame, (x, y1), (min(x + gap, x2), y1), colour, thickness)
        cv2.line(frame, (x, y2), (min(x + gap, x2), y2), colour, thickness)
    for y in range(y1, y2, gap * 2):
        cv2.line(frame, (x1, y), (x1, min(y + gap, y2)), colour, thickness)
        cv2.line(frame, (x2, y), (x2, min(y + gap, y2)), colour, thickness)


def search_around_prediction(frame, predicted_box, reference_hist,
                              search_radius=60, similarity_thresh=0.7):
    """
    Sample candidate boxes around the Kalman predicted position.
    Returns the best matching box if similarity is above threshold,
    otherwise returns None (stick with pure Kalman prediction).
    
    This handles the case where the tracker lost the player but they
    only moved slightly — we find them by appearance near the prediction.
    """
    if reference_hist is None:
        return None

    px1, py1, px2, py2 = predicted_box
    pw = px2 - px1
    ph = py2 - py1
    pcx = (px1 + px2) / 2
    pcy = (py1 + py2) / 2

    best_box   = None
    best_score = similarity_thresh  # must beat this to count

    # Search a grid of offsets around predicted centre
    step = search_radius // 3
    for dx in range(-search_radius, search_radius + 1, step):
        for dy in range(-search_radius, search_radius + 1, step):
            cx = pcx + dx
            cy = pcy + dy

            # Candidate box — same size as last known box
            cx1 = int(cx - pw / 2)
            cy1 = int(cy - ph / 2)
            cx2 = int(cx + pw / 2)
            cy2 = int(cy + ph / 2)

            # Skip if out of frame
            if cx1 < 0 or cy1 < 0 or cx2 > frame.shape[1] or cy2 > frame.shape[0]:
                continue

            candidate_hist = get_colour_histogram(frame, [cx1, cy1, cx2, cy2])
            score = histogram_similarity(reference_hist, candidate_hist)

            if score > best_score:
                best_score = score
                best_box   = (cx1, cy1, cx2, cy2)

    return best_box



def get_colour_histogram(frame, box, bins=16):
    """
    Extract HSV colour histogram from player crop.
    """
    x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
    
    # Crop the player
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0 or crop.shape[0] < 10 or crop.shape[1] < 10:
        return None
    
    # Ignore sides, go from 1/4 -> 3/4 e.g. 1/2
    h = crop.shape[0]
    torso = crop[h//4 : 3*h//4, :]
    
    # HSV histogram
    hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(
        hsv,
        np.array([0,   0,  0]),   # lower
        np.array([255, 200, 180])   # upper
    ) # extract out all pixels except for ones with high saturation and brightness
    
    # If too few pixels survive the mask, return None
    # possibly redundent 
    if cv2.countNonZero(mask) < 20:
        return None

    hist = cv2.calcHist([hsv], [0, 1], mask, [bins, bins],
                        [0, 180, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()

def histogram_similarity(hist1, hist2):
    if hist1 is None or hist2 is None:
        return 0
    return cv2.compareHist(
        hist1.reshape(16, 16).astype(np.float32),
        hist2.reshape(16, 16).astype(np.float32),
        cv2.HISTCMP_CORREL  # 1.0 = identical, -1.0 = opposite
    ) # 16 bins 
    
def update_track_histogram(track_histograms, track_id, new_hist, alpha=0.50):
    """
    Rolling average histogram per track so appearance slowly adapts
    as the player moves without drifting too fast.
    alpha=0.5 means 50% old, 50% new each frame.
    """
    if new_hist is None:
        return
    if track_id not in track_histograms:
        track_histograms[track_id] = new_hist
    else:
        track_histograms[track_id] = (
            alpha * track_histograms[track_id] + (1 - alpha) * new_hist
        )




def manual_kalman():
    """
    Runs the model with a kalman filter and histogram similarity matching 
    """
    # Store last known position per track for Kalman prediction display
    track_kalman  = {}   # track_id -> KalmanFilter
    track_cls     = {}   # track_id -> class_id
    track_lost    = {}   # track_id -> frames since last seen
    track_histograms = {}   # track_id -> rolling average histogram
    frame_id      = 1
    MAX_LOST_FRAMES = 30   # how long to keep predicting after losing a track

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model.track(
            frame,
            persist=True,
            verbose=False,
            conf=0.15,
            )

        current_ids = set()

        if results[0].boxes.id is not None:
            boxes     = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.cpu().numpy().astype(int)
            classes   = results[0].boxes.cls.cpu().numpy().astype(int)
            
            for box, track_id, cls_id in zip(boxes, track_ids, classes):
                x1, y1, x2, y2 = map(int, box)
                current_ids.add(track_id)

                # Update Kalman with real detection
                if track_id not in track_kalman:
                    track_kalman[track_id] = make_kalman_filter(x1, y1, x2, y2)
                else:
                    cx = (x1 + x2) / 2
                    cy = (y1 + y2) / 2
                    track_kalman[track_id].update(
                        np.array([[cx], [cy],
                                [float(x2-x1)], [float(y2-y1)]], dtype=float)
                    )

                track_kalman[track_id].predict()
                track_cls[track_id]  = cls_id
                track_lost[track_id] = 0

                # Update appearance histogram (skip puck - too small)
                if cls_id != PUCK_CLASS_ID:
                    hist = get_colour_histogram(frame, [x1, y1, x2, y2])
                    update_track_histogram(track_histograms, track_id, hist)

                colour = CLASS_COLOURS.get(cls_id, (255, 255, 255))
                label  = f"{CLASS_NAMES.get(cls_id, '?')} #{track_id}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
                cv2.putText(frame, label, (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 2)

        # ---- Lost track prediction + colour correction ---
        lost_ids = set(track_kalman.keys()) - current_ids

        for track_id in list(lost_ids):
            track_lost[track_id] = track_lost.get(track_id, 0) + 1

            if track_lost[track_id] > MAX_LOST_FRAMES:
                track_kalman.pop(track_id, None)
                track_cls.pop(track_id, None)
                track_lost.pop(track_id, None)
                track_histograms.pop(track_id, None)
                continue

            # Advance Kalman without measurement
            track_kalman[track_id].predict()
            pred_box = kf_to_box(track_kalman[track_id])
            pred_box = clamp_box(*pred_box, frame.shape) # keep within screen

            cls_id = track_cls.get(track_id, 0)
            colour = CLASS_COLOURS.get(cls_id, (255, 255, 255))

            reference_hist = track_histograms.get(track_id)
            colour_match   = None

            if cls_id != PUCK_CLASS_ID and reference_hist is not None:
                colour_match = search_around_prediction(
                    frame,
                    pred_box,
                    reference_hist,
                    search_radius=60,
                    similarity_thresh=similarity_thresh
                )

            if colour_match is not None:
                # if we found a similar histogram
                mx1, my1, mx2, my2 = colour_match
                cx = (mx1 + mx2) / 2
                cy = (my1 + my2) / 2
                w  = float(mx2 - mx1)
                h  = float(my2 - my1)

                # Update Kalman with colour match position
                track_kalman[track_id].update(
                    np.array([[cx], [cy], [w], [h]], dtype=float)
                )

                draw_dashed_rectangle(frame, (mx1, my1), (mx2, my2), (0, 165, 255))
                label = f"{CLASS_NAMES.get(cls_id, '?')} #{track_id} [colour pos]"
                cv2.putText(frame, label, (mx1, my1 - 8),
                            cv2.FONT_HERSHEY_PLAIN, 0.4, (0, 165, 255), 1)
            else:
                # No colour match — use predicted position as a soft update
                px1, py1, px2, py2 = pred_box
                cx = (px1 + px2) / 2
                cy = (py1 + py2) / 2
                w  = float(px2 - px1)
                h  = float(py2 - py1)

                # Higher measurement noise = less trust in this self-update
                # vs a real detection or colour match
                original_R = track_kalman[track_id].R.copy()
                track_kalman[track_id].R *= 5  # increase by 5x 

                track_kalman[track_id].update(
                    np.array([[cx], [cy], [w], [h]], dtype=float)
                )

                # Restore normal measurement noise for next real detection
                track_kalman[track_id].R = original_R

                frames_lost = track_lost[track_id]
                draw_dashed_rectangle(frame, (px1, py1), (px2, py2), colour)
                label = f"{CLASS_NAMES.get(cls_id, '?')} #{track_id} [kalman pos {frames_lost}f]"
                cv2.putText(frame, label, (px1, py1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, colour, 1)

        cv2.imshow("Tracking", frame)
        key = cv2.waitKey(FRAME_DELAY) & 0xFF

        if key == ord("q"):
            break
        elif key == ord(" "):
            cv2.waitKey(0)

        frame_id += 1

    cap.release()

    out.release()
    cv2.destroyAllWindows()

manual_kalman()