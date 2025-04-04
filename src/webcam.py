import cv2
import numpy as np
import joblib
import os
from collections import deque
from scipy.stats import mode  # To get the most frequent prediction

# Load trained model
model_path = "../models/hand_gesture_mlp.pkl"
model = joblib.load(model_path)

# Directory to save preprocessed frames
save_dir = "../data/preprocessed_frames/"
os.makedirs(save_dir, exist_ok=True)

frame_count = 0
max_frames_to_save = 5  

gesture_emojis = {
    0: "OK",  1: "One",  2: "Two",  3: "Spock",  4: "Four",
    5: "Five",  6: "Three",  7: "Estranho",  8: "Rockk",  9: "Supimpa",
    10: "Faz o LLLLLLLLLLLL", 11: "Libras", 12: "Call", 13: "Juradinho", 14: "Fist",
    15: "Punch", 16: "Back", 17: "Hang-loose", 18: "Pinky", 19: "Thumbs Up", -1: "No hand"
}

# Store the last N predictions for stability
N_FRAMES = 20 
prediction_buffer = deque(maxlen=N_FRAMES)  

def preprocess_frame(frame, size=(64, 64)):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel = np.ones((5, 5), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(binary.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        cropped = binary[y:y+h, x:x+w]
    else:
        cropped = binary

    # Pad to square
    h, w = cropped.shape
    diff = abs(h - w)
    if h > w:
        pad_left = diff // 2
        pad_right = diff - pad_left
        padded = cv2.copyMakeBorder(cropped, 0, 0, pad_left, pad_right, cv2.BORDER_CONSTANT, value=0)
    else:
        pad_top = diff // 2
        pad_bottom = diff - pad_top
        padded = cv2.copyMakeBorder(cropped, pad_top, pad_bottom, 0, 0, cv2.BORDER_CONSTANT, value=0)

    # Resize to target size
    resized = cv2.resize(padded, size)
    normalized = resized / 255.0

    return normalized.flatten().reshape(1, -1), resized


cap = cv2.VideoCapture(0)
predicted_class = -1  

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # Mirror the frame

    roi_x, roi_y, roi_size = frame.shape[1] - 220, 20, 200  
    roi = frame[roi_y:roi_y + roi_size, roi_x:roi_x + roi_size]  

    processed_frame, binary_img = preprocess_frame(roi)

    # Get the current prediction and store it in the buffer
    current_prediction = model.predict(processed_frame)[0]
    prediction_buffer.append(current_prediction)

    # Compute the mode (most frequent prediction)
    if len(prediction_buffer) == N_FRAMES:
        predicted_class = mode(prediction_buffer).mode.item() # Extract the mode value

    emoji = gesture_emojis.get(predicted_class, "Unknown")

    # Draw ROI rectangle and prediction
    cv2.rectangle(frame, (roi_x, roi_y), (roi_x + roi_size, roi_y + roi_size), (255, 0, 0), 2)
    cv2.putText(frame, "ROI", (roi_x + 5, roi_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    prediction_text = f"Predicted: {predicted_class} {emoji}"
    cv2.putText(frame, prediction_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow("Hand Gesture Recognition", frame)
    cv2.imshow("ROI - Binary Hand", binary_img)  

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
