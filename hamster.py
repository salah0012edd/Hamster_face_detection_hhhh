import cv2  # OpenCV library for computer vision tasks (image processing, video capture, display)
import mediapipe as mp  # MediaPipe library for face and hand landmark detection
import numpy as np  # NumPy for numerical operations (arrays, math)
import os  # Operating system interface (for checking file existence)

# Initialize MediaPipe solutions
mp_face_mesh = mp.solutions.face_mesh  # Face mesh solution for detailed facial landmarks
mp_hands = mp.solutions.hands  # Hands solution for hand landmark detection
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)  # Process up to 1 face with refined (iris) landmarks
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)  # Detect up to 2 hands with 70% minimum confidence

# Define size for all meme images (consistent overlay size)
MEME_SIZE = (300, 300)

# Load and resize the three hamster meme images from the 'memes' folder
memes = {
    'normal': cv2.resize(cv2.imread('memes/normal.jpg'), MEME_SIZE),  # Calm hamster
    'peace': cv2.resize(cv2.imread('memes/peace.jpg'), MEME_SIZE),    # Peace sign hamster
    'scary': cv2.resize(cv2.imread('memes/scary.jpg'), MEME_SIZE),    # Screaming hamster
}

# Mouth landmarks: indices 13 (upper inner lip) and 14 (lower inner lip) for measuring mouth opening
MOUTH_OPEN_LANDMARKS = [13, 14]

def is_mouth_open(landmarks, img_shape, threshold=0.1):
    """
    Detect if the mouth is significantly open.
    Uses distance between upper and lower inner lip, normalized by actual face height (nose to chin).
    Higher threshold requires wider mouth opening to trigger 'scary' state.
    """
    if not landmarks:
        return False
    # Get face height from nose tip (landmark 1) to chin (landmark 152) for better normalization
    nose = landmarks[1]
    chin = landmarks[152]
    face_height = abs(chin.y - nose.y)
    if face_height == 0:  # Prevent division by zero
        return False
    top = landmarks[13]    # Upper inner lip
    bottom = landmarks[14] # Lower inner lip
    dist = abs(top.y - bottom.y)  # Vertical distance between lips
    return dist > face_height * threshold  # True only if mouth is wide open

def is_peace_sign(hand_landmarks, img_shape):
    """
    Detect a proper peace sign (✌️).
    Checks that index and middle fingers are extended, while ring and pinky are folded.
    This prevents triggering just by showing any two fingers.
    """
    if not hand_landmarks:
        return False
    # Wrist as reference
    wrist = hand_landmarks.landmark[0]
    
    # Index finger: MCP (5) and tip (8)
    index_mcp = hand_landmarks.landmark[5]
    index_tip = hand_landmarks.landmark[8]
    
    # Middle finger: MCP (9) and tip (12)
    middle_mcp = hand_landmarks.landmark[9]
    middle_tip = hand_landmarks.landmark[12]
    
    # Ring finger: MCP (13) and tip (16)
    ring_mcp = hand_landmarks.landmark[13]
    ring_tip = hand_landmarks.landmark[16]
    
    # Pinky finger: MCP (17) and tip (20)
    pinky_mcp = hand_landmarks.landmark[17]
    pinky_tip = hand_landmarks.landmark[20]
    
    # Extended: fingertip is higher than MCP joint (y smaller = higher on screen)
    index_extended = index_tip.y < index_mcp.y
    middle_extended = middle_tip.y < middle_mcp.y
    
    # Folded: fingertip is lower than MCP joint
    ring_folded = ring_tip.y > ring_mcp.y
    pinky_folded = pinky_tip.y > pinky_mcp.y
    
    # All conditions must be true for a clear peace sign
    return index_extended and middle_extended and ring_folded and pinky_folded

def overlay_meme(frame, meme_img, position=(50, 50)):
    """
    Overlay the selected hamster meme on the frame with transparency.
    Also adds a text label above the meme.
    """
    h, w = meme_img.shape[:2]  # Get meme dimensions
    x, y = position            # Top-left corner for overlay
    # Blend meme onto frame (30% original frame + 70% meme)
    frame[y:y+h, x:x+w] = cv2.addWeighted(frame[y:y+h, x:x+w], 0.3, meme_img, 0.7, 0)
    # Add detection text above the meme
    cv2.putText(frame, "HAMSTER MEME DETECTED!", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

def detect_hamster_meme(frame):
    """
    Main detection logic: determines current state (normal/peace/scary) based on face and hand.
    Priority: peace sign > mouth open > normal
    """
    # Convert frame to RGB for MediaPipe processing
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_results = face_mesh.process(rgb_frame)  # Detect facial landmarks
    hand_results = hands.process(rgb_frame)      # Detect hand landmarks

    mouth_open = False
    peace_sign = False

    # Check for wide mouth opening
    if face_results.multi_face_landmarks:
        for face_landmarks in face_results.multi_face_landmarks:
            mouth_open = is_mouth_open(face_landmarks.landmark, frame.shape, threshold=0.1)

    # Check for proper peace sign
    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            if is_peace_sign(hand_landmarks, frame.shape):
                peace_sign = True

    # Decision priority
    if peace_sign:
        return 'peace'
    elif mouth_open:
        return 'scary'
    else:
        return 'normal'

# === MAIN FUNCTION: Webcam loop and controls ===
def main():
    cap = cv2.VideoCapture(0)  # Open default webcam
    # Try to set higher resolution for larger window
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    uploaded_image_path = 'test_image.jpg'  # Path for testing static images

    # Print controls to console
    print("Press 'c' to capture and classify current frame")
    print("Press 'u' to upload and test an image")
    print("Press 'q' to quit")

    while True:
        ret, frame = cap.read()  # Read frame from camera
        if not ret:              # Break if camera fails
            break

        frame = cv2.flip(frame, 1)  # Mirror the frame horizontally (selfie view)
        
        # Force larger display size (in case camera doesn't support high res)
        frame = cv2.resize(frame, (1280, 720))

        # Detect current meme state
        current_meme = detect_hamster_meme(frame)
        meme_img = memes[current_meme]  # Select corresponding meme

        # Overlay meme in top-right corner
        overlay_meme(frame, meme_img, position=(frame.shape[1]-320, 20))
        
        # Display current state text on screen
        cv2.putText(frame, f"State: {current_meme.upper()}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

        # Show the final frame
        cv2.imshow('Hamster Meme Detector', frame)

        # Keyboard controls
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):  # Quit the program
            break
        elif key == ord('c'):  # Capture current frame to file
            cv2.imwrite('captured.jpg', frame)
            print(f"Captured! Detected: {current_meme}")
        elif key == ord('u'):  # Test with uploaded image
            if os.path.exists(uploaded_image_path):
                test_img = cv2.imread(uploaded_image_path)
                test_img = cv2.resize(test_img, (1280, 720))  # Consistent size
                result = detect_hamster_meme(test_img)
                meme_result = memes[result]
                overlay_meme(test_img, meme_result, (50, 50))
                cv2.putText(test_img, f"UPLOADED: {result.upper()}", (50, 340),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)
                cv2.imshow('Uploaded Image Result', test_img)
                cv2.waitKey(0)  # Wait for any key to close result window
                cv2.destroyWindow('Uploaded Image Result')
            else:
                print("No uploaded image found! Place 'test_image.jpg' in folder.")

    # Clean up: release camera and close windows
    cap.release()
    cv2.destroyAllWindows()

# Entry point
if __name__ == "__main__":
    main()