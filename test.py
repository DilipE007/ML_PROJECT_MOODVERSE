import cv2
import torch
import torch.nn as nn
from torchvision import transforms, models
import mediapipe as mp
import numpy as np
from PIL import Image

# Model path
model_path = 'C:\\Users\\lenovo\\Desktop\\ML_PROJECT_MOODVERSE\\model\\trainedmodel\\emotion_model.pth'

# Emotion classes (mapped to model output indices)
class_names = ['Angry', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Load the trained model
num_classes = 6
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = models.resnet18(weights=None)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, num_classes)
try:
    model.load_state_dict(torch.load(model_path, map_location=device))
except Exception as e:
    print(f"Error loading model: {str(e)}")
    exit()
model = model.to(device)
model.eval()

# Transform for image preprocessing (matches training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Mediapipe Face Detector
mp_face = mp.solutions.face_detection
face_detector = mp_face.FaceDetection(model_selection=1, min_detection_confidence=0.3)

def detect_emotion_live():
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    print("Starting live emotion detection. Press 'q' to exit.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame from webcam.")
            break
        
        # Convert to RGB for Mediapipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        results = face_detector.process(rgb_frame)
        emotion = "No Face Detected"
        confidence = 0.0
        
        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                h, w, _ = frame.shape
                x1 = max(0, int(bboxC.xmin * w))
                y1 = max(0, int(bboxC.ymin * h))
                x2 = min(w, x1 + int(bboxC.width * w))
                y2 = min(h, y1 + int(bboxC.height * h))
                
                # Extract ROI (face)
                face_img = frame[y1:y2, x1:x2]
                if face_img.size > 0 and face_img.shape[0] >= 10 and face_img.shape[1] >= 10:
                    face_img_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                    face_img_resized = cv2.resize(face_img_rgb, (224, 224))
                    face_pil = Image.fromarray(face_img_resized)
                    face_tensor = transform(face_pil).unsqueeze(0).to(device)
                    
                    # Predict emotion
                    with torch.no_grad():
                        outputs = model(face_tensor)
                        probabilities = torch.softmax(outputs, dim=1)
                        _, pred = torch.max(outputs, 1)
                        emotion = class_names[pred.item()]
                        confidence = probabilities[0][pred.item()].item() * 100
                    
                    # Draw bounding box and emotion label
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"{emotion} ({confidence:.1f}%)", (x1, y1 - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    break  # Process only the first face
        else:
            cv2.putText(frame, "No Face Detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Display result in OpenCV window
        cv2.imshow('Live Emotion Detection', frame)
        print(f"Detected Emotion: {emotion} ({confidence:.1f}% confidence)")
        
        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    print("Live emotion detection stopped.")

# Run live webcam testing
if __name__ == "__main__":
    detect_emotion_live()