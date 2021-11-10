import numpy as np

class Camera:

    def __init__(self, deck):
        self.focal_length = deck.Focal_length
        self.sensor_height = deck.Sensor_height
        self.fov = 2*np.arctan(self.sensor_height/(2*self.focal_length))
        print('Camera field of view : ' + str(round(self.fov*180/np.pi, 2)) +'° or '+str(round(self.fov,2))+' rad' )
