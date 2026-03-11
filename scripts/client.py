import cv2
import requests
import base64
import numpy as np

# API endpoints
API_URL = "http://localhost:5000/ws"
RESET_URL = "http://localhost:5000/reset"

def frame_to_base64(frame):
    """Convert frame to base64 string"""
    _, buffer = cv2.imencode('.jpg', frame)
    base64_image = base64.b64encode(buffer).decode('utf-8')
    return base64_image

def reset_processor():
    """Reset the processor state"""
    try:
        response = requests.post(RESET_URL)
        if response.status_code == 200:
            print("Processor reset successfully")
        else:
            print(f"Reset error: {response.status_code}, {response.text}")
    except Exception as e:
        print(f"Error resetting processor: {e}")

def process_frame(frame):
    """Send frame to API and get response"""
    try:
        base64_image = frame_to_base64(frame)
        payload = {"image": base64_image}
        response = requests.post(API_URL, json=payload)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Error: {response.status_code}, {response.text}")
            return None
    except Exception as e:
        print(f"Error processing frame: {e}")
        return None

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    print("Starting video stream. Press 'q' to quit, 'r' to reset.")
    reset_processor()  # Reset processor at start

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        # Create a flipped copy for display
        display_frame = cv2.flip(frame, 1)
        
        # Send original frame for processing
        result = process_frame(frame)
        if result:
            current_text = result.get("current_text", "")
            action_feedback = result.get("action_feedback", "")
            fps = result.get("fps", 0)
            is_stable = result.get("is_stable", False)
            predicted_letter = result.get("predicted_letter", None)
            confidence = result.get("confidence", 0)

            # Display results on flipped frame
            cv2.putText(display_frame, f"Text: {current_text}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(display_frame, f"Action: {action_feedback}", (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            cv2.putText(display_frame, f"FPS: {fps:.2f}", (10, 90), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(display_frame, f"Stable: {is_stable}", (10, 120), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(display_frame, f"Letter: {predicted_letter} ({confidence:.2f})", (10, 150), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.imshow("Sign Language Recognition", display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            reset_processor()

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()