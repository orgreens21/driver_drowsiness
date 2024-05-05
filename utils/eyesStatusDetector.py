import cv2
import dlib
from imutils import face_utils
from utils.threshold_calc import ThresholdCalculator
from utils.consecutive_checker import ConsecutiveChecker
import os


class EyeStatusDetector:
    def __init__(self, start_th=0.225, consecutive_frames=200):
        self.detector = dlib.get_frontal_face_detector()
        dir_path = os.path.dirname(os.path.abspath(__file__))
        shape_predictor_path = os.path.join(dir_path, 'calibration', 'shape_predictor_68_face_landmarks.dat')
        self.predictor = dlib.shape_predictor(shape_predictor_path)
        # self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        self.lStart, self.lEnd = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        self.rStart, self.rEnd = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
        self.eye_threshold = ThresholdCalculator(start_th)
        self.consecutive_frames = ConsecutiveChecker(consecutive_frames)

    def eye_aspect_ratio(self, eye):
        A = cv2.norm(eye[1] - eye[5])
        B = cv2.norm(eye[2] - eye[4])
        C = cv2.norm(eye[0] - eye[3])
        ear = (A + B) / (2.0 * C)
        return ear

    def detect_eye_status(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        rects = self.detector(gray, 0)
        eye_status = 'Unknown'
        for rect in rects:
            shape = self.predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)
            # for (x, y) in shape:
            #     cv2.circle(image, (x, y), 2, (0, 0, 255), -1)
            eye_status = 'OPEN'
            shape = self.predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)
            leftEye = shape[self.lStart:self.lEnd]
            rightEye = shape[self.rStart:self.rEnd]
            leftEAR = self.eye_aspect_ratio(leftEye)
            rightEAR = self.eye_aspect_ratio(rightEye)
            ear = (leftEAR + rightEAR) / 2
            self.eye_threshold.add_value(ear)

            if ear < self.eye_threshold.get_threshold():
                self.consecutive_frames.add_value(True)
                if self.consecutive_frames.check_consecutive():
                    eye_status = 'CLOSED'

            # cv2.putText(image, "Eye: {}".format(eye_status), (10, 30),
            #         cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            # cv2.putText(image, "EAR: {:.2f}".format(ear), (300, 30),
            #         cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        return eye_status