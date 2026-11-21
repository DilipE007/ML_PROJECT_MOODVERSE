import streamlit as st
import cv2
import torch
import torch.nn as nn
from torchvision import transforms, models
import mediapipe as mp
import numpy as np
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import time
import os
from PIL import Image
import pandas as pd
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Spotify setup
SPOTIFY_CLIENT_ID = '89b03e82959a4333835a921241959177'
SPOTIFY_CLIENT_SECRET = 'a4292d368e9b4dff9454bc741a8d5f6f'
client_credentials_manager = SpotifyClientCredentials(client_id=SPOTIFY_CLIENT_ID, client_secret=SPOTIFY_CLIENT_SECRET)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

# Load your trained model
model_path = os.path.join('model', 'trainedmodel', 'emotion_model.pth')
num_classes = 6
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = models.resnet18(weights=None)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, num_classes)
model_loaded = False
if os.path.exists(model_path):
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model = model.to(device)
        model.eval()
        model_loaded = True
    except Exception as e:
        # If loading fails, present the error in the app but don't crash
        st.error(f"Failed to load model: {e}")
        model = None
else:
    st.error(f"Model file not found at '{model_path}'.\nPlease place your trained model at this path or update `model_path`.")
    model = None

# Class names (match training)
class_names = ['Angry', 'Fear', 'Happy', 'Neutral', 'Sad', 'Suprise']

# Emotion to Spotify query mapping
EMOTION_QUERIES = {
    'Angry': 'angry rock metal playlist',
    'Fear': 'calm relaxing anxiety relief playlist',
    'Happy': 'happy upbeat pop playlist',
    'Neutral': 'chill ambient playlist',
    'Sad': 'sad emotional ballad playlist',
    'Suprise': 'exciting adventure soundtrack playlist'
}

# Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Mediapipe Face Detector
mp_face = mp.solutions.face_detection
face_detector = mp_face.FaceDetection(model_selection=1, min_detection_confidence=0.3)  # Lowered for better detection

# Session state
if 'detection_history' not in st.session_state:
    st.session_state.detection_history = pd.DataFrame(columns=['Timestamp', 'Emotion', 'Confidence'])
if 'last_emotion' not in st.session_state:
    st.session_state.last_emotion = None
if 'last_update_time' not in st.session_state:
    st.session_state.last_update_time = 0
if 'suggestions' not in st.session_state:
    st.session_state.suggestions = []

# Streamlit App
st.set_page_config(page_title="Emotion Detection", layout="wide")
st.title("🎭 Emotion Detection with Spotify Suggestions")
st.markdown("Upload a photo or use the camera to detect emotions and get personalized music recommendations!")

# Sidebar
st.sidebar.header("Controls")
run_detection = st.sidebar.checkbox("Enable Detection", value=True)
confidence_threshold = st.sidebar.slider("Min Confidence (%)", 0, 100, 30)
clear_history = st.sidebar.button("Clear History")

if clear_history:
    st.session_state.detection_history = pd.DataFrame(columns=['Timestamp', 'Emotion', 'Confidence'])
    st.session_state.suggestions = []
    st.success("History and suggestions cleared!")

# Main layout
col1, col2 = st.columns([2, 1])

with col1:
    st.header("📸 Detection Feed")
    if run_detection:
        # Try camera input first; if not available or camera blocked, provide an uploader fallback
        camera_input = None
        try:
            camera_input = st.camera_input("Click to take a photo with your camera")
        except Exception:
            camera_input = None

        if not camera_input:
            st.info("Camera not available or blocked. You can upload a photo instead.")
            upload_input = st.file_uploader("Or upload an image", type=['png', 'jpg', 'jpeg'])
            image_source = upload_input
        else:
            image_source = camera_input

        if image_source:
            try:
                # Process image
                image = Image.open(image_source)
                frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Detect faces
                results = face_detector.process(rgb_frame)
                emotion = "No Face Detected"
                confidence = 0
                
                if results.detections:
                    for detection in results.detections:
                        bboxC = detection.location_data.relative_bounding_box
                        h, w, _ = frame.shape
                        x1 = max(0, int(bboxC.xmin * w))
                        y1 = max(0, int(bboxC.ymin * h))
                        x2 = min(w, x1 + int(bboxC.width * w))
                        y2 = min(h, y1 + int(bboxC.height * h))
                        
                        # Crop face
                        face_img = frame[y1:y2, x1:x2]
                        if face_img.size > 0 and face_img.shape[0] >= 10 and face_img.shape[1] >= 10:
                            face_img_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                            face_img_resized = cv2.resize(face_img_rgb, (224, 224))
                            face_pil = Image.fromarray(face_img_resized)
                            face_tensor = transform(face_pil).unsqueeze(0).to(device)

                            # Predict only if model loaded
                            if model is not None and model_loaded:
                                with torch.no_grad():
                                    outputs = model(face_tensor)
                                    probabilities = torch.softmax(outputs, dim=1)
                                    _, pred = torch.max(outputs, 1)
                                    emotion = class_names[pred.item()]
                                    confidence = probabilities[0][pred.item()].item() * 100
                            else:
                                emotion = "Model not loaded"
                                confidence = 0
                            
                            # Draw
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(frame, f"{emotion} ({confidence:.1f}%)", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                            break  # First face
                else:
                    cv2.putText(frame, "No Face Detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # Display
                st.image(frame, channels="BGR", caption="Processed Image", use_column_width=True)
                
                # Store in history if high confidence
                if confidence > confidence_threshold:
                    new_row = pd.DataFrame({
                        'Timestamp': [time.strftime("%Y-%m-%d %H:%M:%S")],
                        'Emotion': [emotion],
                        'Confidence': [confidence]
                    })
                    st.session_state.detection_history = pd.concat([st.session_state.detection_history, new_row], ignore_index=True)
                
                # Current detection message
                if emotion != "No Face Detected":
                    st.success(f"**Detected Emotion**: {emotion} ({confidence:.1f}% confidence)")
                else:
                    st.warning("No face detected—try a clearer photo!")
                
                # Update suggestions (inside scope)
                if confidence > confidence_threshold and emotion != st.session_state.last_emotion and time.time() - st.session_state.last_update_time > 5:
                    st.session_state.last_emotion = emotion
                    st.session_state.last_update_time = time.time()
                    
                    try:
                        query = EMOTION_QUERIES.get(emotion, 'mood playlist')
                        results_sp = sp.search(q=query, type='playlist', limit=5)
                        st.session_state.suggestions = []
                        
                        for playlist in results_sp['playlists']['items']:
                            playlist_url = playlist['external_urls']['spotify']
                            st.write(f"**Playlist**: [{playlist['name']}]({playlist_url})")
                            
                            # Top 3 tracks
                            tracks = sp.playlist_tracks(playlist['id'])['items'][:3]
                            for track in tracks:
                                if track['track']:
                                    track_name = track['track']['name']
                                    artist = track['track']['artists'][0]['name']
                                    st.session_state.suggestions.append(f"{track_name} by {artist}")
                                    st.write(f"  - {track_name} by {artist}")
                    except Exception as e:
                        st.error(f"Error fetching suggestions: {str(e)}")
                        
            except Exception as e:
                st.error(f"Error processing image: {str(e)}")
                st.info("Try a different photo or check lighting.")
        else:
            st.info("Click the camera button to take a photo or upload one.")

        # Troubleshooting guidance for camera issues
        with st.expander("Camera troubleshooting"):
            st.write("If your camera isn't working: ")
            st.write("- Make sure your browser has permission to use the camera (check site settings).")
            st.write("- Try a different browser (Chrome/Edge/Firefox).")
            st.write("- Close other apps that may be using the camera.")
            st.write("- If using localhost, ensure your browser supports getUserMedia on non-HTTPS for localhost.")
            st.write("- As a fallback, upload an image using the uploader above.")
    else:
        st.info("Enable detection in the sidebar to begin.")

with col2:
    st.header("📊 Detection History")
    if not st.session_state.detection_history.empty:
        st.dataframe(st.session_state.detection_history.tail(10), use_container_width=True)
        # Download
        csv = st.session_state.detection_history.to_csv(index=False)
        st.download_button("📥 Download History (CSV)", csv, "emotion_history.csv", "text/csv")
    else:
        st.info("No detections yet—start detection!")

    st.header("🎵 Music Suggestions")
    if st.session_state.suggestions:
        for i, song in enumerate(st.session_state.suggestions, 1):
            st.write(f"{i}. {song}")
    else:
        st.info("Detection with high confidence needed for suggestions.")

# Footer
st.markdown("---")
st.caption("Built with Streamlit, PyTorch, Mediapipe, and Spotify API.")