from ultralytics import YOLO
import cv2

# Load trained model
model = YOLO("best.pt")

# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Camera open nahi ho raha!")
    exit()

print("✅ Camera started!")
print("Press Q to quit.")

while True:
    ret, frame = cap.read()

    if not ret:
        print("❌ Frame read nahi ho raha!")
        break

    # YOLO prediction
    results = model(frame, conf=0.25, verbose=False)

    # Draw detections
    for result in results:
        if result.boxes is None:
            continue

        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])

            class_name = model.names[class_id]

            # Mask = Green
            if class_name == "Mask":
                color = (0, 255, 0)

            # No_Mask = Red
            elif class_name == "No_Mask":
                color = (0, 0, 255)

            # Other classes = Blue
            else:
                color = (255, 0, 0)

            # Draw box
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                color,
                2
            )

            # Label
            label = f"{class_name} {confidence:.2f}"

            cv2.putText(
                frame,
                label,
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )

    # Display
    cv2.imshow("Multi Object Detection", frame)

    # Press Q to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release camera
cap.release()
cv2.destroyAllWindows()