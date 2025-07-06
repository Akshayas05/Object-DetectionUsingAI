from ultralytics import YOLO
import cv2
import pyttsx3

model = YOLO("yolov5s.pt")

engine = pyttsx3.init()
engine.setProperty('rate', 150)  

cap = cv2.VideoCapture(0)
last_objects = set()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    detections = results[0]

    current_objects = set()

    for box in detections.boxes:
        confidence = float(box.conf[0])
        if confidence > 0.5:
            class_id = int(box.cls[0])
            label = model.names[class_id]
            current_objects.add(label)
            print(f"Detected: {label} ({confidence * 100:.2f}%)")

    new_objects = current_objects - last_objects
    if new_objects:
        spoken_text = ", ".join(new_objects)
        print("Speaking:", spoken_text)
        engine.say(f"I see {spoken_text}")
        engine.runAndWait()
        last_objects = current_objects

    annotated_frame = results[0].plot()
    cv2.imshow("YOLOv5 Object Detection with Voice", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
