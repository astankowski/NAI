"""
Prototype of "Red Light, Green Light" game using Computer Vision and Mediapipe Pose detection.
The player must remain still when the light is red and can move when the light is green.
If the player moves during the red light, the game ends.
Win condition: Reach a score of 500 by moving during green lights without being caught.
You score points for each frame you move during green and yellow light.
Moving during green light gives 1 point per frame, moving during yellow light gives 3 points per frame.

Authors:
    - Daniel Bieliński (s27292)
    - Aleksander Stankowski (s27549)
Date:
    - 2026-01-07

Environment setup:
    - Python 3.12
    - python3.12 -m venv venv
    - source venv/bin/activate  (Linux/Mac)
    - venv\Scripts\activate     (Windows)
    - pip install opencv-python mediapipe==0.10.14 <- specific version for compatibility
    
Usage:
    1. Ensure your camera is connected and working.
    2. Run the script in an environment that supports OpenCV GUI functions. (python3.12 RedLightGreenLight.py)
    3. Follow on-screen instructions to play the game:
        - Press 'Q' to quit the game.
        - Press 'R' to restart the game after a game over.

Gameplay Example:
    - https://drive.google.com/file/d/1zOp3vDrN8ZxlkuntHXztfHAAKw3uWwgH/view?usp=drive_link

"""

import cv2 as cv
import mediapipe as mp
import math 
from collections import deque
import copy 
import time
import random
    
def normalized_size_to_pixels(mp_x, mp_y, img_width, img_height):
    """
    Converts MediaPipe normalized coordinates (0.0 - 1.0) to pixel coordinates

    Args:
        mp_x (float): Normalized x coordinate from MediaPipe
        mp_y (float): Normalized y coordinate from MediaPipe
        img_width (int): Width of the image/frame in pixels
        img_height (int): Height of the image/frame in pixels
    Returns:
        (int, int): Tuple of (x, y) pixel coordinates
    """
    
    x_pixel = int(mp_x * img_width)
    y_pixel = int(mp_y * img_height)
    return x_pixel, y_pixel

def draw_crosshair(img, x, y, color):
    """
    Draws crosshair at given (x,y) coordinates

    Args:
        img (np.ndarray): Image/Frame to draw crosshair on (OpenCV format)
        x (int): x pixel coordinate
        y (int): y pixel coordinate
        color (tuple): BGR color tuple

    Returns:
        np.ndarray: Image/Frame with crosshair drawn on it
    """
    
    cv.circle(img, (x, y), 15, color, 1)
    cv.line(img, (x - 20, y), (x + 20, y), color, 1)
    cv.line(img, (x, y - 20), (x, y + 20), color, 1)
    return img

def calculate_body_movement(current_landmarks, previous_landmarks, visibility_threshold = 0.5) -> float:
    """
    Calculates the average movement distance of body landmarks between two frames.
    
    Args:
        current_landmarks (List[landmark_pb2.NormalizedLandmark]): List of MediaPipe NormalizedLandmark objects for the current frame
        previous_landmarks (List[landmark_pb2.NormalizedLandmark]): List of MediaPipe NormalizedLandmark objects for the previous frame
        visibility_threshold (float): Minimum visibility score to consider a landmark for movement calculation
    Returns:
        float: Average movement distance of landmarks that are visible in both frames
    """
    
    if not current_landmarks or not previous_landmarks:
        return 0.0
    
    total_distance = 0.0
    visible_points_count = 0

    for i in range(len(current_landmarks)):
        curr_lm = current_landmarks[i]
        prev_lm = previous_landmarks[i]
        
        if curr_lm.visibility > visibility_threshold and prev_lm.visibility > visibility_threshold:
            delta_x = curr_lm.x - prev_lm.x
            delta_y = curr_lm.y - prev_lm.y

            distance = math.sqrt(delta_x ** 2 + delta_y ** 2)
            total_distance += distance
            visible_points_count += 1
            
    if visible_points_count == 0:
        return 0.0
    
    average_distance = total_distance / visible_points_count
    return average_distance

def load_video_capture(video_device: int = 0, output_width: int = 640, output_height:int = 480) -> cv.VideoCapture:
    """
    Loads video capture device based on the specified number

    Args:
        video_device (int, optional): Index of camera device to open. Defaults to 0
        output_width (int, optional): The desired width of the video frame. Defaults to 640
        output_height (int, optional): The desired height of the video frame. Defaults to 480

    Returns:
        cv.VideoCapture: VideoCapture object ready for rendering frames
    """
    capture = cv.VideoCapture(video_device)

    if not capture.isOpened():
        print("Error: Could not open camera.")
        exit()
        
    capture.set(cv.CAP_PROP_FRAME_WIDTH, output_width)
    capture.set(cv.CAP_PROP_FRAME_HEIGHT, output_height)

    return capture

def run_game():
    # Mediapipe and OpenCV setup
    #mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
    capture = load_video_capture()
    
    # Setting up a buffer to store previous movement positions
    buffer_size = 5
    movement_score_buffer = deque(maxlen=buffer_size)
    prev_landmarks = None
    
    # Movement threshold for detecting significant movement for game over condition
    MOVEMENT_THRESHOLD = 0.002

    # Basic game state variables
    light_status = "GREEN"
    last_switch_time = time.time()
    time_limit = random.uniform(2.0, 6.0)  # Random time limit between 2 to 6 seconds
    is_alive = True
    score = 0
    score_win_condition = 500
    game_won = False
    
    # Start video capture and pose detection loop
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while capture.isOpened():
            isTrue, frame = capture.read()
            if not isTrue: break
            
            # Mediapipe uses rgb images instead of OpenCV's bgr format thus the conversion
            image = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)
            image.flags.writeable = True
            image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
            
            # Managing light status based on time limit
            if is_alive:
                elapsed = time.time() - last_switch_time
                
                if elapsed > time_limit:
                    if light_status == "GREEN":
                        light_status = "YELLOW"
                        time_limit = 1.5
                        
                    elif light_status == "YELLOW":
                        light_status = "RED"
                        time_limit = random.uniform(2.0, 4.0)
                        
                    elif light_status == "RED":
                        light_status = "GREEN"
                        time_limit = random.uniform(2.0, 5.0)
                    
                    last_switch_time = time.time()
            
            # Drawing pose landmarks on the image
            #mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            
            movement_status = "Waiting"
            avg_movement = 0.0
            
            # Checking whether landmarks are detected
            if results.pose_landmarks:                
                current_landmarks = results.pose_landmarks.landmark
                
                if prev_landmarks:
                    raw_movement = calculate_body_movement(current_landmarks, prev_landmarks)
                    movement_score_buffer.append(raw_movement)
                    
                    # Calculating movement distance
                    if len(movement_score_buffer) > 0:

                        avg_movement = sum(movement_score_buffer) / len(movement_score_buffer)

                        nose = current_landmarks[mp_pose.PoseLandmark.NOSE]
                        nose_x, nose_y = normalized_size_to_pixels(nose.x, nose.y-0.05, frame.shape[1], frame.shape[0])

                        if avg_movement > MOVEMENT_THRESHOLD:
                            movement_status = "Moving"
                            draw_crosshair(image, nose_x, nose_y, (0,0,255))

                        else:
                            movement_status = "Still"
                            draw_crosshair(image, nose_x, nose_y, (0,255,0))

                       
                        # Game over condition
                        if is_alive and light_status == "RED" and avg_movement > MOVEMENT_THRESHOLD:
                            is_alive = False
                            movement_status = "GAME OVER"
                            print("GAME OVER! You moved during RED light.")
                        elif is_alive and avg_movement > MOVEMENT_THRESHOLD:
                            if light_status == "GREEN":
                                score += 1
                            elif light_status == "YELLOW":
                                score += 3
                            if score >= score_win_condition:
                                is_alive = False
                                game_won = True
                                print("CONGRATULATIONS! You reached the winning score.")
                            
                # DeepCopy the landmarks to save them for the next frame.
                # Without deepcopy, 'prev' points to 'current', and they change together, 
                # resulting in 0 movement always. 
                prev_landmarks = copy.deepcopy(current_landmarks)


            h, w, c = image.shape                
            
            if light_status == "GREEN":
                bar_color = (0, 255, 0)
                status_text = "GO!"
            elif light_status == "YELLOW":
                bar_color = (0, 255, 255)
                status_text = "PREPARE TO STOP!"
            else:
                bar_color = (0, 0, 255)
                status_text = "STOP!" 
            
            cv.rectangle(image, (0,0), (w, 50), bar_color, -1)
            
            if not is_alive:
                if game_won:
                    bar_color = (0, 255, 0)
                    cv.rectangle(image, (0,0), (w, h), (0, 0, 0), -1)
                    cv.putText(image, "CONGRATULATIONS! YOU WON!", (int(w/10), int(h/2)), cv.FONT_HERSHEY_SIMPLEX, 1.2, (0,200,0), 3)
                    cv.putText(image, "Press 'R' to restart.", (int(w/6), int(h/2) + 40), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                else:
                    bar_color = (0, 0, 255)
                    cv.rectangle(image, (0,0), (w, h), (0, 0, 0), -1)
                    cv.putText(image, "GAME OVER!", (int(w/6), int(h/2)), cv.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), 3)
                    cv.putText(image, "Press 'R' to restart.", (int(w/6), int(h/2) + 40), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            else:
                display_text = f"Light: {light_status} | Movement: {movement_status} | Avg Movement: {avg_movement:.4f}"
                cv.putText(image, display_text, (10,30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

            cv.putText(image, f"Score: {score}", (10, h - 20), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            cv.putText(image, "Press 'Q' to quit.", (w - 150, h - 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
            cv.imshow("Red light, Green light", image)
            
            # Key controls for quitting and restarting the game 
            # (waitKey sets delay to 5ms between frames. Actual FPS = 1000ms / (5ms + processing time))
            key = cv.waitKey(5) & 0xFF
            if key == ord('q'):
                break
            if key == ord('r') and not is_alive:
                is_alive = True
                light_status = "GREEN"
                last_switch_time = time.time()
                movement_score_buffer.clear()
                prev_landmarks = None
                score = 0
                game_won = False

    # Release resources
    capture.release()
    cv.destroyAllWindows()
    
if __name__ == "__main__":
    run_game()