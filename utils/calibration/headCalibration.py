import time
import os
import cv2
import dlib
from imutils import face_utils
from imutils.video import VideoStream
import imutils
from gtts import gTTS
from playsound import playsound
import cv2
import numpy as np
import mediapipe as mp


class HeadDirectionDetector:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.mp_drawing = mp.solutions.drawing_utils
        self.drawing_spec = self.mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

    def detect_direction(self, image):
        img_h, img_w, img_c = image.shape
        face_3d = []
        face_2d = []

        results = self.face_mesh.process(image)
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                for idx, lm in enumerate(face_landmarks.landmark):
                    if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
                        if idx == 1:
                            nose_2d = (lm.x * img_w, lm.y * img_h)
                            nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 3000)

                        x, y = int(lm.x * img_w), int(lm.y * img_h)
                        face_2d.append([x, y])
                        face_3d.append([x, y, lm.z])       

                face_2d = np.array(face_2d, dtype=np.float64)
                face_3d = np.array(face_3d, dtype=np.float64)

                focal_length = 1 * img_w
                cam_matrix = np.array([[focal_length, 0, img_h / 2], [0, focal_length, img_w / 2], [0, 0, 1]])
                dist_matrix = np.zeros((4, 1), dtype=np.float64)
                _success, rot_vec, _trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)
                rmat, jac = cv2.Rodrigues(rot_vec)
                angles, _mtxR, _mtxQ, _Qx, _Qy, _Qz = cv2.RQDecomp3x3(rmat)
                x = angles[0] * 360
                y = angles[1] * 360
        
                return x, y
            
        return None, None


def play_sound(file_path):
    playsound(file_path)


def create_sounds():
    dir_path = os.path.dirname(os.path.abspath(__file__))
    tts = gTTS(text=f"Please look to the left", lang='en')
    tts.save(os.path.join(dir_path, "head_left.mp3"))
    tts = gTTS(text=f"Please look to the right", lang='en')
    tts.save(os.path.join(dir_path, "head_right.mp3"))
    tts = gTTS(text=f"Please look down", lang='en')
    tts.save(os.path.join(dir_path, "head_down.mp3"))
    tts = gTTS(text=f"Please look up", lang='en')
    tts.save(os.path.join(dir_path, "head_up.mp3"))
    tts = gTTS(text=f"Please look forward", lang='en')
    tts.save(os.path.join(dir_path, "forward.mp3"))
    tts = gTTS(text="Head calibration complete", lang='en')
    tts.save(os.path.join(dir_path, "head_calibration_complete.mp3"))


def calculate_head_direction_threshold():
    create_sounds()
    head_detector = HeadDirectionDetector() 
    time_per_posture = 5
    calibration_running = True
    left_values = []
    right_values = []
    up_values = []
    down_values = []
    forward_x = []
    forward_y = []

    dir_path = os.path.dirname(os.path.abspath(__file__))

    vs = VideoStream(src=1).start()

    start_time = time.time()
    started_left = False
    started_up = False
    started_down = False
    started_right = False
    started_farward = False

    while calibration_running:
        frame = vs.read()
        frame = cv2.flip(frame, 1)
        cv2.imshow("Frame", frame)
        frame = imutils.resize(frame, width=800)
        start = time.time()
        x, y = head_detector.detect_direction(frame)
        
        if x and y:

            time_from_start = time.time() - start_time

            if time_from_start < time_per_posture:
                if not started_up:
                    play_sound(os.path.join(dir_path, "head_up.mp3"))
                    started_up = True
                up_values.append(x)

            elif time_per_posture < time_from_start < 2 * time_per_posture:
                if not started_down:
                    play_sound(os.path.join(dir_path, "head_down.mp3"))
                    started_down = True
                down_values.append(x)

            elif 2 * time_per_posture < time_from_start < 3 * time_per_posture:
                if not started_right:
                    play_sound(os.path.join(dir_path, "head_right.mp3"))
                    started_right = True
                right_values.append(y)

            elif 3 * time_per_posture < time_from_start < 4 * time_per_posture:
                if not started_left:
                    play_sound(os.path.join(dir_path, "head_left.mp3"))
                    started_left = True
                left_values.append(y)

            elif 4 * time_per_posture < time_from_start < 5 * time_per_posture:
                if not started_farward:
                    play_sound(os.path.join(dir_path, "forward.mp3"))
                    started_farward = True
                forward_x.append(x)
                forward_y.append(y)
            
            else:
                calibration_running = False
                break

            end = time.time()
            totalTime = end - start
            fps = 1 / totalTime
            cv2.putText(frame, f'FPS: {int(fps)}', (20,450), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)
            cv2.putText(frame, "X: {}".format(x), (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, "Y: {}".format(y), (300, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):  # Press 'q' to quit
                break

    
    vs.stop()
    cv2.destroyAllWindows()
    play_sound(os.path.join(dir_path, "head_calibration_complete.mp3"))
    left_average = sum(left_values) / len(left_values)
    right_average = sum(right_values) / len(right_values)
    up_average = sum(up_values) / len(up_values)
    down_average = sum(down_values) / len(down_values)
    forward_x_average = sum(forward_x) / len(forward_x)
    forward_y_average = sum(forward_y) / len(forward_y)

    print('average left:', left_average)
    print('average right:', right_average)
    print('average up:', up_average)
    print('average down:', down_average)
    print("Calibration complete.")

    result = {
        "left": 1/2 * (left_average + forward_x_average),
        "right": 1/2 * (right_average + forward_x_average), 
        "up": 1/2 * (up_average + forward_y_average), 
        "down": 1/2 * (down_average + forward_y_average)
        }
    
    print("result: ")
    print(result)
    
    return result


if __name__ == "__main__":
    calculate_head_direction_threshold()
