class ThresholdCalculator:
    def __init__(self, init_th, buffer_size=2000):
        self.buffer_size = buffer_size
        self.closed_values = [init_th + 1 for i in range(self.buffer_size)]
        self.open_values = [init_th - 1 for i in range(self.buffer_size)]

    
    def add_value(self, value):
        # Determine whether the value belongs to the closed or open area
        if value >= self.get_threshold():
            if len(self.closed_values) > self.buffer_size:
                self.closed_values.pop()
            self.closed_values.append(value)
        else:
            if len(self.open_values) > self.buffer_size:
                self.open_values.pop()
            self.open_values.append(value)
    
    def get_threshold(self):
        # Calculate the average of the higher and lower values
        closed_avg = sum(self.closed_values) / len(self.closed_values) if self.closed_values else 0
        open_avg = sum(self.open_values) / len(self.open_values) if self.open_values else 0
        
        # Return the mean of the two averages
        return (closed_avg + open_avg) / 2