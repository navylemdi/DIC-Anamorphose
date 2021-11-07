import numpy as np

class Camera:

    def __init__(self, deck):
        self.focal_length = deck.focal_length
        self.sensor_height = deck.sensor_height
        self.fov = 2*np.arctan(self.sensor_height/(2*self.focal_length))
