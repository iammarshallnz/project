import cv2
import os

cap = cv2.VideoCapture("./pwhl_one_hour.mp4")
fps = cap.get(cv2.CAP_PROP_FPS)
output_dir = "./new_frames"
os.makedirs(output_dir, exist_ok=True)

# Extract 1 frame every 2 seconds — avoids near-duplicate frames
interval = int(fps * 30)
frame_count = 0
saved = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    if frame_count % interval == 0:
        cv2.imwrite(f"{output_dir}/frame_{saved:05d}.jpg", frame)
        saved += 1
    frame_count += 1

cap.release()
print(f"Saved {saved} frames")