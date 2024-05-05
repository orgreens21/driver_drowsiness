import time
import os
import cv2
import dlib
from imutils import face_utils
from imutils.video import VideoStream
import imutils
from gtts import gTTS
from playsound import playsound

def play_sound(file_path):
    playsound(file_path)


class EyeStatusDetector:
    def __init__(self):
        self.detector = dlib.get_frontal_face_detector()
        dir_path = os.path.dirname(os.path.abspath(__file__))
        shape_predictor_path = os.path.join(dir_path, 'shape_predictor_68_face_landmarks.dat')
        self.predictor = dlib.shape_predictor(shape_predictor_path)
        # self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        self.lStart, self.lEnd = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        self.rStart, self.rEnd = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

    def eye_aspect_ratio(self, eye):
        A = cv2.norm(eye[1] - eye[5])
        B = cv2.norm(eye[2] - eye[4])
        C = cv2.norm(eye[0] - eye[3])
        ear = (A + B) / (2.0 * C)
        return ear

    def detect_eye_status(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        rects = self.detector(gray, 0)
        for rect in rects:
            shape = self.predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)
            for (x, y) in shape:
                cv2.circle(image, (x, y), 1, (0, 0, 255), -1)
            shape = self.predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)
            leftEye = shape[self.lStart:self.lEnd]
            rightEye = shape[self.rStart:self.rEnd]
            leftEAR = self.eye_aspect_ratio(leftEye)
            rightEAR = self.eye_aspect_ratio(rightEye)
            ear = (leftEAR + rightEAR) / 2

            return ear

        return None


def create_sounds():
    dir_path = os.path.dirname(os.path.abspath(__file__))
    tts = gTTS(text=f"Please open your eyes", lang='en')
    tts.save(os.path.join(dir_path, "open_eyes.mp3"))
    tts = gTTS(text=f"Please close your eyes", lang='en')
    tts.save(os.path.join(dir_path, "close_eyes.mp3"))
    tts = gTTS(text="Eyes calibration complete", lang='en')
    tts.save(os.path.join(dir_path, "eyes_calibration_complete.mp3"))


def calculate_eye_threshold():
    calibration_time_for_each_state = 5
    create_sounds()
    eye_status_detector = EyeStatusDetector()

    calibration_running = True
    open_ear_values = []
    closed_ear_values = []
    dir_path = os.path.dirname(os.path.abspath(__file__))

    vs = VideoStream(src=1).start()
    time.sleep(1.0)
    open_eyes_test = False
    close_eyes_test = False
    start_time = time.time()

    while calibration_running:
        frame = vs.read()
        frame = cv2.flip(frame, 1)
        frame = imutils.resize(frame, width=800)
        frame_ear = eye_status_detector.detect_eye_status(frame)
        cv2.imshow("Frame", frame)
        if frame_ear:
            start = time.time()
            time_from_start = time.time() - start_time
            print("time_from_start: ", time_from_start)
            if time_from_start < calibration_time_for_each_state:
                print(1)
                if not open_eyes_test:
                    play_sound(os.path.join(dir_path, "open_eyes.mp3"))
                    open_eyes_test = True
                open_ear_values.append(frame_ear)

            elif time_from_start < 2 * calibration_time_for_each_state:
                print(2)
                if not close_eyes_test:
                    play_sound(os.path.join(dir_path, "close_eyes.mp3"))
                    close_eyes_test = True
                closed_ear_values.append(frame_ear)

            else:
                calibration_running = False
                break
            
            totalTime = time.time() - start
            if totalTime != 0:
                fps = 1 / totalTime
                cv2.putText(frame, f'FPS: {int(fps)}', (20,450), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)
                cv2.putText(frame, "frame_ear: {}".format(frame_ear), (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):  # Press 'q' to quit
            break

    vs.stop()
    cv2.destroyAllWindows()
    play_sound(os.path.join(dir_path, "eyes_calibration_complete.mp3"))
    average_open_ear = sum(open_ear_values) / len(open_ear_values)
    average_closed_ear = sum(closed_ear_values) / len(closed_ear_values)
    threshold = (average_closed_ear + average_open_ear) / 2
    print('average_open_ear:', average_open_ear)
    print('average_closed_ear:', average_closed_ear)
    print("Calibration complete. Start threshold set to:", threshold)
    return threshold


if __name__ == "__main__":
    calculate_eye_threshold()
