import cv2
import torch
import pandas as pd
from ultralytics import YOLO

# Load the pre-trained YOLO model
model = YOLO("./runs/detect/train5/weights/best.pt") 

# Define the video path and output paths
video_path = "/Users/jenny/OneDrive/Desktop/chromis/dataset/C0058.mp4"
output_video_path = "/Users/jenny/OneDrive/Desktop/chromis/AnnotatedVideos/C0058_ATry.mp4"
output_csv_path = '/Users/jenny/OneDrive/Desktop/chromis/dataset/C0058_DataTry.csv'

# Start video capture
cap = cv2.VideoCapture(video_path)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Prepare video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

# Prepare to save data
detections = []

frame_idx = 0
# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        results = model.track(frame, persist=True)
        annotated_frame = results[0].plot()
        out.write(annotated_frame)

        for det in results[0].boxes:
            xyxyn = det.xyxyn.tolist()[0]  # [x1, y1, x2, y2]
            x1, y1, x2, y2 = xyxyn[0], xyxyn[1], xyxyn[2], xyxyn[3]
            # Calculate the center of the bounding box
            x = (x1 + x2) / 2
            y = (y1 + y2) / 2
            track_id = det.id if det.id is not None else -1  # Track ID

            detections.append({
                'frame': frame_idx,
                'x': x,
                'y': y,
                'track_id': int(track_id.item())
            })

        frame_idx += 1  # Increment frame index

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        if frame_idx >= 30:
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()
df = pd.DataFrame(detections)
df.to_csv(output_csv_path, index=False)