import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
from breakPointTracker import FishTracker 


model = YOLO("./runs/detect/train5/weights/best.pt")


video_path = "/Users/jenny/OneDrive/Desktop/chromis/dataset/C0084.mp4"
output_video_path = "/Users/jenny/OneDrive/Desktop/chromis/AnnotatedVideos/C0084_A_OPT2.mp4"
output_csv_path = '/Users/jenny/OneDrive/Desktop/chromis/dataset/C0084_DataOPT2.csv'

# Initialize the FishTracker for Optimization
tracker = FishTracker(threshold=5.0, lambda1=0.1, lambda2=0.1)


cap = cv2.VideoCapture(video_path)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Initialize video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))


detections = []

frame_idx = 0
previous_fishes = []
fish_positions = {}  # store fish positions across frames
latest_id = 0

# def calculate_speed(fish_id, current_position, previous_position, fps):
#     distance = ((current_position[0] - previous_position[0]) ** 2 + (current_position[1] - previous_position[1]) ** 2) ** 0.5
#     speed = distance * fps
#     return speed
def calculate_speed(current_position, previous_position, time_elapsed):
    if time_elapsed == 0:
        return [0, 0]
    speed_vector = (np.array(current_position) - np.array(previous_position)) / time_elapsed
    return speed_vector.tolist()

# Loop through the video frames
while cap.isOpened():
    # Read frame
    success, frame = cap.read()

    if success:
        results = model.track(frame, persist=True, tracker="botsort.yaml")

        current_fishes = []
        new_fishes = []
        disappeared_fishes = []
        id_mapping = {}

        for det in results[0].boxes:
            xyxyn = det.xyxyn.tolist()[0]  # [x1, y1, x2, y2]
            x1, y1, x2, y2 = xyxyn[0], xyxyn[1], xyxyn[2], xyxyn[3]
            x = (x1 + x2) / 2
            y = (y1 + y2) / 2
            track_id = det.id if det.id is not None else -1  # Track ID

            fish = {
                'id': int(track_id.item()) if track_id != -1 else None,
                'position': [x, y],
                'size': (x2 - x1) * (y2 - y1),
                'new': False,
                'disappeared': False,
                'confidence': det.conf,
                'class_id': det.cls,
                'bounding_box': [x1, y1, x2, y2],
                'speed': [0, 0]
            }

            current_fishes.append(fish)

        # find disappeared fishes
        current_ids = set(fish['id'] for fish in current_fishes if fish['id'] is not None)
        for previous_fish in previous_fishes:
            if previous_fish['id'] not in current_ids:
                previous_fish['disappeared'] = True
                disappeared_fishes.append(previous_fish)

        # find new fishes
        previous_ids = set(fish['id'] for fish in previous_fishes if fish['id'] is not None)
        for fish in current_fishes:
            if fish['id'] is None or fish['id'] not in previous_ids:
                fish['new'] = True
                new_fishes.append(fish)
        
        if latest_id == 0:
            latest_id = len(current_fishes)

        # update the FishTracker with disappeared fishes
        tracker.update_disappeared_fishes(current_frame=frame_idx, fishes=disappeared_fishes)
        # connect breakpoints
        tracker.track_fish(current_frame=frame_idx, new_fishes=new_fishes, basic = False)

        # assign new IDs to unmatched new fishes
        for fish in new_fishes:
            if fish['id'] is None:
                fish['id'] = latest_id  
                latest_id += 1  

        # Ensure latest_id is up-to-date
        for fish in current_fishes:
            if fish['id'] is not None:
                latest_id = max(latest_id, fish['id'] + 1)
        # for fish in new_fishes:
        #     if fish['id'] is None:
        #         fish['id'] = len(detections)  # Assign a new ID if not matched
            # id_mapping[(fish['bounding_box'][0], fish['bounding_box'][1], fish['bounding_box'][2], fish['bounding_box'][3])] = fish['id']

        for fish in current_fishes:
            fish_id = fish['id']
            current_position = fish['position']
            speed = [0, 0]
            id_mapping[(fish['bounding_box'][0], fish['bounding_box'][1], fish['bounding_box'][2], fish['bounding_box'][3])] = fish['id']

            if fish_id in fish_positions:
                # previous_position = fish_positions[fish_id]
                # speed = calculate_speed(fish_id, current_position, previous_position, fps)
                previous_position, previous_speed = fish_positions[fish_id]
                time_elapsed = 1 / fps
                speed = calculate_speed(current_position, previous_position, time_elapsed)

            # fish_positions[fish_id] = current_position
            fish_positions[fish_id] = (current_position, speed)

            detections.append({
                'frame': frame_idx,
                'x': fish['position'][0],
                'y': fish['position'][1],
                'track_id': int(fish['id']),
                'x1': fish['bounding_box'][0],
                'y1': fish['bounding_box'][1],
                'x2': fish['bounding_box'][2],
                'y2': fish['bounding_box'][3],
                'confidence': fish['confidence'],
                'class_id': fish['class_id'],
                'speed_x': speed[0],
                'speed_y': speed[1]
            })

        # Visualize the results on the frame
        annotated_frame = frame.copy()
        for det in results[0].boxes:
            xyxyn = det.xyxyn.tolist()[0]
            x1, y1, x2, y2 = xyxyn[0], xyxyn[1], xyxyn[2], xyxyn[3]
            if (xyxyn[0], xyxyn[1], xyxyn[2], xyxyn[3]) in id_mapping:
                track_id = id_mapping[(xyxyn[0], xyxyn[1], xyxyn[2], xyxyn[3])]
            else:
                track_id = -1

            x1 = int(x1 * width)
            x2 = int(x2 * width)
            y1 = int(y1 * height)
            y2 = int(y2 * height)
            # Draw bounding box and ID on the frame
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(annotated_frame, f'ID: {track_id}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 2)

        # writ to output video
        out.write(annotated_frame)

        previous_fishes = current_fishes

        frame_idx += 1  

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        # if frame_idx >= 30:  # For testing, break after 10 frames
        #     break
    else:
        # end of video
        break

# Release the video capture object and close the display window
cap.release()
out.release()
cv2.destroyAllWindows()
df = pd.DataFrame(detections)
df.to_csv(output_csv_path, index=False)
