# pytest test_sign_language_recognition.py -v
import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import torch
from fastapi.testclient import TestClient
import sys
import os

# Add the parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from slr.api import app
from slr.sign_processor import SignProcessor, CNN1D
import cv2
import base64
from collections import deque

class TestSignLanguageRecognition(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures."""
        self.client = TestClient(app)
        # Get the path to the models directory (one level up from test file)
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.model_path = os.path.join(base_dir, "models", "cnn_asl_model_full.pth")
        self.labels_path = os.path.join(base_dir, "models", "label_encoder_classes.npy")
        
        # Mock dependencies for SignProcessor
        self.mock_model = MagicMock(spec=CNN1D)
        self.mock_labels = np.array(['A', 'B', 'del', 'space'])
        self.mock_hands = MagicMock()
        self.mock_mp_hands = MagicMock()
        self.mock_mp_hands.Hands.return_value = self.mock_hands
        self.mock_mp_drawing = MagicMock()

        # Mock state_dict for CNN1D
        self.mock_state_dict = {
            'conv1.weight': torch.randn(64, 1, 3),
            'conv1.bias': torch.randn(64),
            'conv2.weight': torch.randn(32, 64, 3),
            'conv2.bias': torch.randn(32),
            'fc1.weight': torch.randn(128, (63 // 4) * 32),
            'fc1.bias': torch.randn(128),
            'fc2.weight': torch.randn(28, 128),
            'fc2.bias': torch.randn(28),
        }

    @patch('torch.load')
    @patch('numpy.load')
    @patch('mediapipe.python.solutions.hands.Hands')
    def test_sign_processor_initialization(self, mock_hands, mock_np_load, mock_torch_load):
        """Test SignProcessor initialization."""
        mock_torch_load.return_value = {'model_state_dict': self.mock_state_dict}
        mock_np_load.return_value = self.mock_labels
        mock_hands.return_value = self.mock_hands

        processor = SignProcessor(model_path=self.model_path, labels_path=self.labels_path)
        
        # Normalize device comparison to handle 'cuda' vs 'cuda:0'
        model_device = next(processor.model.parameters()).device
        expected_device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.assertEqual(model_device, expected_device)
        self.assertEqual(processor.labels.tolist(), self.mock_labels.tolist())
        mock_torch_load.assert_called_with(self.model_path, map_location=processor.device)
        mock_np_load.assert_called_with(self.labels_path, allow_pickle=True)

    def test_normalize_landmarks(self):
        """Test landmark normalization."""
        processor = SignProcessor()
        landmarks = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])
        normalized = processor.normalize_landmarks(landmarks)
        
        self.assertEqual(normalized.shape, (9,))
        self.assertAlmostEqual(np.std(normalized), 1.0, places=5)
        self.assertTrue(np.allclose(normalized[:3], 0))  # Wrist should be origin

    def test_check_stability(self):
        """Test stability checking."""
        processor = SignProcessor()
        processor.landmark_history = deque(maxlen=3)  # Create new deque with maxlen=3
        current = np.array([0.1, 0.1, 0.1])
        history = [np.array([0.11, 0.11, 0.11]), np.array([0.09, 0.09, 0.09]), np.array([0.1, 0.1, 0.1])]
        processor.landmark_history.extend(history)
        
        is_stable = processor.check_stability(current, processor.landmark_history)
        self.assertTrue(is_stable)
        
        unstable_history = [np.array([0.3, 0.3, 0.3]), np.array([0.09, 0.09, 0.09]), np.array([0.1, 0.1, 0.1])]
        processor.landmark_history = deque(unstable_history, maxlen=3)
        is_stable = processor.check_stability(current, processor.landmark_history)
        self.assertFalse(is_stable)

    @patch('time.time')
    def test_process_prediction_letter(self, mock_time):
        """Test process_prediction for letter input."""
        processor = SignProcessor()
        processor.labels = self.mock_labels
        processor.MIN_PREDICTION_COUNT = 2
        processor.prediction_buffer = deque(['A', 'A'], maxlen=3)  # Initialize with maxlen=3
        mock_time.return_value = 10.0
        processor.last_letter_time = 9.0
        
        processor.process_prediction('A', 0.8)
        self.assertEqual(processor.word, ['A'])
        self.assertEqual(processor.action_feedback, 'Added: A')
        self.assertEqual(len(processor.prediction_buffer), 0)

    @patch('time.time')
    def test_process_prediction_delete(self, mock_time):
        """Test process_prediction for delete action."""
        processor = SignProcessor()
        processor.labels = self.mock_labels
        processor.MIN_DELETE_PREDICTION_COUNT = 2
        processor.prediction_buffer = deque(['del', 'del'], maxlen=3)  # Initialize with maxlen=3
        processor.word = ['H', 'E', 'L', 'L', 'O']
        mock_time.return_value = 10.0
        processor.last_letter_time = 9.0
        processor.is_stable = True
        
        processor.process_prediction('del', 0.8)
        self.assertEqual(processor.word, ['H', 'E', 'L', 'L'])
        self.assertEqual(processor.action_feedback, 'Deleted')
        self.assertEqual(len(processor.prediction_buffer), 0)

    @patch('cv2.flip')
    @patch('cv2.cvtColor')
    @patch('torch.tensor')
    @patch('mediapipe.python.solutions.hands.Hands')
    def test_process_frame(self, mock_hands, mock_tensor, mock_cvtColor, mock_flip):
        """Test frame processing."""
        processor = SignProcessor()
        processor.labels = self.mock_labels
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Mock MediaPipe results
        mock_landmarks = [MagicMock(x=0.5, y=0.5, z=0.0) for _ in range(21)]
        mock_hand_landmarks = MagicMock(landmark=mock_landmarks)
        mock_results = MagicMock(multi_hand_landmarks=[mock_hand_landmarks])
        processor.hands.process.return_value = mock_results
        
        # Mock model inference
        mock_output = torch.tensor([[0.1, 0.8, 0.05, 0.05]])
        processor.model.return_value = mock_output
        mock_tensor.return_value = torch.tensor(np.zeros((1, 1, 63)))
        
        result = processor.process_frame(frame)
        
        self.assertEqual(result['status'], 'success')
        self.assertFalse(result['is_stable'])
        self.assertIsNone(result['predicted_letter'])
        self.assertEqual(result['current_text'], '')

    def test_reset(self):
        """Test processor reset."""
        processor = SignProcessor()
        processor.word = ['H', 'E', 'L', 'L', 'O']
        processor.prediction_buffer.append('A')
        processor.landmark_history.append(np.array([0.1, 0.1, 0.1]))
        processor.frame_count = 100
        processor.is_stable = True
        
        result = processor.reset()
        
        self.assertEqual(result['status'], 'success')
        self.assertEqual(processor.word, [])
        self.assertEqual(len(processor.prediction_buffer), 0)
        self.assertEqual(len(processor.landmark_history), 0)
        self.assertEqual(processor.frame_count, 0)
        self.assertFalse(processor.is_stable)

    @patch('slr.api.processor', new=MagicMock())  # Mock SignProcessor to reduce initialization
    def test_root_endpoint(self):
        """Test GET / endpoint."""
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"message": "Sign Language Recognition API is running"})

    @patch('slr.api.processor', new=MagicMock())  # Mock SignProcessor
    def test_process_image_endpoint_invalid_data(self):
        """Test POST /ws with invalid base64 image data."""
        # Use an invalid base64 string to trigger binascii.Error
        invalid_image_data = "invalid_base64_string"  # Missing proper padding and characters
        data = {"image": invalid_image_data}
        
        response = self.client.post("/ws", json=data)
        self.assertEqual(response.status_code, 400)
        self.assertIn("Invalid base64 image data", response.json()['detail'])

    @patch('cv2.imdecode')
    @patch('numpy.frombuffer')
    @patch('slr.api.processor')
    def test_process_image_endpoint(self, mock_processor, mock_frombuffer, mock_imdecode):
        """Test POST /ws with valid image data."""
        # Create a dummy image and encode it
        dummy_image = np.zeros((100, 100, 3), dtype=np.uint8)
        _, buffer = cv2.imencode('.png', dummy_image)
        b64_image = base64.b64encode(buffer).decode('utf-8')
        
        mock_frombuffer.return_value = buffer
        mock_imdecode.return_value = dummy_image
        mock_processor.process_frame.return_value = {
            "status": "success",
            "is_stable": False,
            "current_text": "",
            "action_feedback": None,
            "fps": 30.0,
            "predicted_letter": None,
            "confidence": 0.0
        }
        
        data = {"image": b64_image}
        response = self.client.post("/ws", json=data)
        
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()['status'], 'success')

    @patch('slr.api.processor')
    def test_reset_endpoint(self, mock_processor):
        """Test POST /reset endpoint."""
        mock_processor.reset.return_value = {"status": "success", "message": "Processor reset"}
        response = self.client.post("/reset")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()['status'], 'success')

if __name__ == '__main__':
    unittest.main()
