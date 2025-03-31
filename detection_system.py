from transformers import pipeline
import cv2
import torch

# Load object detection model with MPS acceleration
object_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
device = 'mps' if torch.backends.mps.is_available() else 'cpu'
object_model.to(device)

# Initialize emotion detection pipeline
emotion_classifier = pipeline(
    "text-classification",
    model="joeddav/distilbert-base-uncased-go-emotions-student",
    return_all_scores=True
)

def detect_emotions(frame):
    # This is a placeholder implementation
    # In a real implementation, you would use a vision-based emotion detection model
    # For demonstration, we'll return dummy data
    emotions = [
        ("happy", 0.9, (100, 100, 200, 200)),
        ("sad", 0.8, (300, 100, 400, 200))
    ]
    return emotions

def detect_objects(frame):
    results = object_model(frame)
    objects = []
    for det in results.pred[0]:
        x1, y1, x2, y2, conf, cls = det[:6]
        obj_class = results.names[int(cls)]
        objects.append((obj_class, conf.item(), (int(x1), int(y1), int(x2), int(y2))))
    return objects

def main():
    # Get list of connected cameras
    camera_indexes = []
    for i in range(10):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            camera_indexes.append(i)
            cap.release()
    
    if not camera_indexes:
        print("No cameras found. Please connect a camera and try again.")
        return
    
    # Select first available camera
    cap = cv2.VideoCapture(camera_indexes[0])
    if not cap.isOpened():
        print(f"Failed to open camera {camera_indexes[0]}")
        return
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame from camera")
            break
        
        # Detect emotions
        emotions = detect_emotions(frame)
        
        # Detect objects
        objects = detect_objects(frame)
        
        # Draw emotion bounding boxes and labels
        for emotion, confidence, box in emotions:
            x1, y1, x2, y2 = box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'{emotion} {confidence:.2f}', 
                       (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        # Draw object bounding boxes and labels
        for obj_class, confidence, box in objects:
            x1, y1, x2, y2 = box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, f'{obj_class} {confidence:.2f}', 
                       (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        
        cv2.imshow('Detection System', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()