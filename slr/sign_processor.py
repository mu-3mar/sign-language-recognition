import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
from collections import deque
import time
from spellchecker import SpellChecker
import os

# CNN model for sign language recognition
class CNN1D(nn.Module):
    def __init__(self, input_size=63, num_classes=28):
        super(CNN1D, self).__init__()
        self.conv1 = nn.Conv1d(1, 64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(64, 32, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool1d(2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear((input_size // 4) * 32, 128)
        self.dropout = nn.Dropout(0.4)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool1(x)
        x = torch.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Main sign language processor class
class SignProcessor:
    def __init__(self, model_path=None, labels_path=None):
        # Repo root (parent of `slr/`)
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if model_path is None:
            model_path = os.path.join(base_dir, "models", "cnn_asl_model_full.pth")
        if labels_path is None:
            labels_path = os.path.join(base_dir, "models", "label_encoder_classes.npy")

        # Initialize tracking variables
        self.landmark_history = deque(maxlen=5)  # Increased history size for better stability
        self.prediction_buffer = deque(maxlen=4)
        self.word = []
        self.last_letter = None
        self.last_letter_time = time.time()
        self.last_delete_stable_time = 0
        self.action_feedback = None
        self.action_feedback_time = 0
        self.spell = SpellChecker(language='en')

        # Configuration parameters
        self.PREDICTION_INTERVAL = 2
        self.LETTER_DELAY = 0.7  # Reduced delay for better response
        self.CONFIDENCE_THRESHOLD = 0.75
        self.REPEAT_DELAY = 0.8
        self.DELETE_DELAY = 0.2
        self.DELETE_STABILITY_DURATION = 0.2
        self.STABILITY_THRESHOLD = 0.1  # Stricter stability criteria
        self.MIN_PREDICTION_COUNT = 2
        self.MIN_DELETE_PREDICTION_COUNT = 3

        # Performance tracking
        self.frame_count = 0
        self.processing_time = deque(maxlen=10)
        self.is_stable = False

        # Setup MediaPipe
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7
        )
        self.mp_drawing = mp.solutions.drawing_utils

        # Load model
        try:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model = CNN1D()
            checkpoint = torch.load(model_path, map_location=self.device)
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            self.model.to(self.device)
            self.model.eval()
            print("[INFO] Model loaded successfully")
        except Exception as e:
            print(f"[ERROR] Failed to load model: {e}")
            raise

        # Load labels
        try:
            self.labels = np.load(labels_path, allow_pickle=True)
            print("[INFO] Labels loaded successfully")
        except Exception as e:
            print(f"[ERROR] Failed to load labels: {e}")
            raise

    def normalize_landmarks(self, landmarks):
        """Normalize hand landmarks relative to wrist position"""
        landmarks = landmarks.reshape(-1, 3)
        wrist = landmarks[0]
        normalized = landmarks - wrist
        scale = np.std(normalized) + 1e-6
        normalized = normalized / scale
        return normalized.flatten()

    def check_stability(self, current, history):
        """Check if hand position is stable across frames"""
        if len(history) < history.maxlen:
            return False
        history_array = np.array(history)
        diffs = np.mean(np.abs(history_array - current), axis=1)
        return np.all(diffs < self.STABILITY_THRESHOLD)

    def process_prediction(self, predicted_letter, confidence):
        """Process model predictions and update text"""
        if confidence < self.CONFIDENCE_THRESHOLD:
            return

        current_time = time.time()

        # Count letter occurrences in buffer
        letter_counts = {}
        for letter in self.prediction_buffer:
            letter_counts[letter] = letter_counts.get(letter, 0) + 1

        # Handle delete action
        if predicted_letter == 'del' and letter_counts.get('del', 0) >= self.MIN_DELETE_PREDICTION_COUNT:
            if current_time - self.last_letter_time > self.DELETE_DELAY and self.word:
                if self.last_delete_stable_time == 0:
                    self.word.pop()
                    self.last_letter_time = current_time
                    self.last_delete_stable_time = current_time
                    self.action_feedback = "Deleted"
                    self.action_feedback_time = current_time
                    self.prediction_buffer.clear()
                elif self.is_stable and (current_time - self.last_delete_stable_time) >= self.DELETE_STABILITY_DURATION:
                    self.word.pop()
                    self.last_letter_time = current_time
                    self.last_delete_stable_time = current_time
                    self.action_feedback = "Deleted"
                    self.action_feedback_time = current_time
                    self.prediction_buffer.clear()
            return

        # Handle letter/space input
        if predicted_letter != 'del' and letter_counts.get(predicted_letter, 0) >= self.MIN_PREDICTION_COUNT:
            if current_time - self.last_letter_time > self.LETTER_DELAY:
                if predicted_letter != self.last_letter or current_time - self.last_letter_time > self.REPEAT_DELAY:
                    if predicted_letter == 'space':
                        current_word = ''.join(self.word).strip().split(' ')
                        if current_word:
                            last_word = current_word[-1]
                            corrected = self.spell.correction(last_word)
                            if corrected and corrected.lower() != last_word.lower():
                                for _ in range(len(last_word)):
                                    self.word.pop()
                                for i, c in enumerate(corrected):
                                    if not current_word[:-1] and i == 0:
                                        self.word.append(c.upper())
                                    else:
                                        self.word.append(c.lower())
                        self.word.append(' ')
                    else:
                        new_char = predicted_letter.lower()
                        if not self.word:
                            new_char = new_char.upper()
                        self.word.append(new_char)

                    self.last_letter = predicted_letter
                    self.last_letter_time = current_time
                    self.last_delete_stable_time = 0
                    self.action_feedback = f"Added: {predicted_letter}"
                    self.action_feedback_time = current_time
                    self.prediction_buffer.clear()

    def process_frame(self, frame):
        """Process video frame and return results"""
        start_time = time.time()
        frame = cv2.flip(frame, 1)
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(image_rgb)

        predicted_letter = None
        confidence = 0.0

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
            normalized = self.normalize_landmarks(landmarks)

            self.frame_count += 1
            self.landmark_history.append(normalized)
            self.is_stable = bool(self.check_stability(normalized, self.landmark_history))

            if not self.is_stable:
                self.last_delete_stable_time = 0

            if self.is_stable and self.frame_count % self.PREDICTION_INTERVAL == 0:
                input_tensor = torch.tensor(normalized, dtype=torch.float32).reshape(1, 1, -1).to(self.device)
                with torch.no_grad():
                    output = self.model(input_tensor)
                    probs = torch.softmax(output, dim=1)
                    confidence = float(torch.max(probs).item())
                    predicted_index = int(torch.argmax(probs, dim=1).item())
                predicted_letter = str(self.labels[predicted_index])
                self.prediction_buffer.append(predicted_letter)
                if len(self.prediction_buffer) == self.prediction_buffer.maxlen:
                    try:
                        self.process_prediction(predicted_letter, confidence)
                    except Exception as e:
                        print(f"[ERROR] Prediction processing error: {e}")
                        self.prediction_buffer.clear()

        # Calculate FPS
        self.processing_time.append(time.time() - start_time)
        fps = float(1.0 / (sum(self.processing_time) / len(self.processing_time)))

        # Return results
        return {
            "status": "success",
            "is_stable": self.is_stable,
            "current_text": ''.join(self.word),
            "action_feedback": self.action_feedback if self.action_feedback and time.time() - self.action_feedback_time < 2.0 else None,
            "fps": fps,
            "predicted_letter": predicted_letter,
            "confidence": confidence
        }

    def reset(self):
        """Reset processor state"""
        self.word = []
        self.last_letter = None
        self.last_letter_time = time.time()
        self.last_delete_stable_time = 0
        self.action_feedback = None
        self.action_feedback_time = 0
        self.prediction_buffer.clear()
        self.landmark_history.clear()
        self.frame_count = 0
        self.is_stable = False
        return {"status": "success", "message": "Processor reset"}

    def close(self):
        """Close MediaPipe detector"""
        self.hands.close()