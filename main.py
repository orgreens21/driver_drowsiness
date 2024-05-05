from imutils.video import VideoStream
import imutils
import time
import cv2
import winsound
from utils.headPostureEstimation import HeadDirectionDetector
from utils.eyesStatusDetector import EyeStatusDetector
import os
from utils.calibration.eyesCalibration import calculate_eye_threshold
from utils.calibration.headCalibration import calculate_head_direction_threshold
def empty(x):
    pass

def main():
    try:

        try:
            # raise RuntimeError("Test")
            eye_th = calculate_eye_threshold()
        
        except Exception as e:
            eye_th = 0.255

        try:
            # raise RuntimeError("Test")
            head_directions = calculate_head_direction_threshold()    
        
        except Exception as e:
            head_directions = {
                "left": -10,
                "right": 10,
                "up": 10, 
                "down": -10
            }

        time.sleep(1)

        vs = VideoStream(src=0).start()
        time.sleep(1.0)
        # cv2.namedWindow("parameters")
        # cv2.resizeWindow("parameters", 640, 240)
        
        eyes_detector = EyeStatusDetector(eye_th)
        head_detector = HeadDirectionDetector(head_directions)

        while True:
            try:
                start = time.time()
                frame = vs.read()
                frame = cv2.flip(frame, 1)
                frame = imutils.resize(frame, width=800)

                # Detect eye status and head direction
                eye_status = eyes_detector.detect_eye_status(frame)
                head_direction = head_detector.detect_direction(frame)    

                if eye_status == 'CLOSED' or head_direction != 'Forward':
                    # Comment out for alert sound 
                    # winsound.Beep(1000, 200)
                    cv2.putText(frame, "ALERT!!!", (380, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                cv2.putText(frame, "Eyes status: {}".format(eye_status), (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(frame, "Head direction: {}".format(head_direction), (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                end = time.time()
                totalTime = end - start
                fps = 1 / totalTime
                cv2.putText(frame, f'FPS: {int(fps)}', (20,450), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)

                cv2.imshow("Frame", frame)

            except Exception as e:
                print(e)


            # Check for 'q' key press to exit
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            # Pause or resume the video stream
            elif key == ord("p"):  
                while True:
                    key = cv2.waitKey(0) & 0xFF
                    if key == ord("p"):
                        break

        # Cleanup
        cv2.destroyAllWindows()
        vs.stop()

    except Exception as e:
        print('Error in main', e)


if __name__ == "__main__":
    main()