import cv2
import torch

def detect_robots_live(model):
    import cv2

    # Initialize the webcam (use 0 for the default camera)
    cap = cv2.VideoCapture(0)

    # Check if the webcam opened successfully
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Retrieve class names if available
    class_names = model.names if hasattr(model, 'names') else {}

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        # Run inference on the frame
        results = model(frame)

        # Convert detections to a pandas DataFrame for easy processing
        detections = results.pandas().xyxy[0]  # Results in a DataFrame

        # Iterate through detections
        for _, row in detections.iterrows():
            # Extract bounding box coordinates and class info
            x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
            confidence = row['confidence']
            class_id = int(row['class'])

            # Get class label or default to 'Unknown'
            label = f"{class_names.get(class_id, 'Unknown')}: {confidence:.2f}"

            # Draw bounding box and label on the frame
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Display the annotated frame
        cv2.imshow('Robot Detection - Press Q to Exit', frame)

        # Exit loop on 'Q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close windows
    cap.release()
    cv2.destroyAllWindows()



# Main function to run the live detection
def main():
    # Load the YOLOv5 model
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=r'C:\Users\godof\Documents\GitHub\2025 Game Piece Detection\yolov5\runs\train\good model\weights\best.pt')


    # Run live detection
    detect_robots_live(model)

if __name__ == "__main__":
    main()
