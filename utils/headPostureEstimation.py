import cv2
import numpy as np
import mediapipe as mp
from utils.consecutive_checker import ConsecutiveChecker


class HeadDirectionDetector:
    def __init__(self, head_directions, consecutive_frames=100):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.mp_drawing = mp.solutions.drawing_utils
        self.drawing_spec = self.mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
        self.consecutive_frames = ConsecutiveChecker(consecutive_frames)
        self.left_th = head_directions.get("left")
        self.right_th = head_directions.get("right")
        self.up_th = head_directions.get("up")
        self.down_th = head_directions.get("down")

    def detect_direction(self, image):
        img_h, img_w, _img_c = image.shape
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
                # z = angles[2] * 360

                if y < self.left_th:
                    direction = "Looking Left"
                    self.consecutive_frames.add_value(True)
                elif y > self.right_th:
                    direction = "Looking Right"
                    self.consecutive_frames.add_value(True)
                elif x < self.down_th:
                    direction = "Looking Down"
                    self.consecutive_frames.add_value(True)
                elif x > self.up_th:
                    direction = "Looking Up"
                    self.consecutive_frames.add_value(True)
                else:
                    direction = "Forward"
                    self.consecutive_frames.add_value(False)
                    
                # p1 = (int(nose_2d[0]), int(nose_2d[1]))
                # p2 = (int(nose_2d[0] + y * 10) , int(nose_2d[1] - x * 10))
                # cv2.line(image, p1, p2, (255, 0, 0), 3)

                # cv2.putText(image, "x: {}".format(x), (20, 500), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                # cv2.putText(image, "y: {}".format(y), (20, 530), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                return direction