import cv2


class ConsecutiveChecker:
    def __init__(self, consecutive_results_th=5):
        self.threshold = consecutive_results_th
        self.consecutive_count = 0
    
    def add_value(self, value):
        if value:
            self.consecutive_count += 1
        else:
            self.consecutive_count = 0
    
    def check_consecutive(self):
        return self.consecutive_count >= self.threshold