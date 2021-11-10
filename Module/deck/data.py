import yaml
import os.path, sys
import numpy as np
import cv2 as cv2

class Deck():
    def __init__(self, inputhpath):
        if not os.path.exists(inputhpath):
            print("File " + inputhpath + " not found")
            sys.exit(1)
        else:
            with open(inputhpath, 'r') as f:
                self.doc = yaml.load(f, Loader=yaml.BaseLoader)

                self.Camera = self.doc['Camera']
                self.Focal_length = float(self.Camera['Focal_length'])
                self.Sensor_height = float(self.Camera['Sensor_height'])

                self.Input_Speckle = self.doc['Input_speckle']
                self.Step = int(self.Input_Speckle['Step'])
                self.Begining = int(self.Input_Speckle['Begining'])
                self.Height = float(self.Input_Speckle['Height'])
                self.Width = float(self.Input_Speckle['Width'])
                self.Path = self.Input_Speckle['Path']
                self.Generic_name = self.Input_Speckle['Generic_name']
                self.NbImage = int(self.Input_Speckle['NbImage'])
                self.Position_centre = np.array(self.Input_Speckle['Position_centre'], dtype = float)

                self.Surface = self.doc['Surface']
                self.a = float(self.Surface['a'])
                self.b = float(self.Surface['b'])
                self.c = float(self.Surface['c'])
                self.Position = np.array(self.Surface['Position'], dtype = float)
                self.Radius = float(self.Surface['Radius'])
                self.Surface_type = self.Surface['Surface_type']
                self.Wingframe = np.array(self.Surface['Wingframe'], dtype = float)

                self.Output_Speckle = self.doc['Output_speckle']
                self.Height_printable = float(self.Output_Speckle['Height_printable'])
                self.Width_printable = float(self.Output_Speckle['Width_printable'])
                self.Print_path = self.Output_Speckle['Print_path']
    def Images(self):
        list=[]
        i=0
        while len(list) < self.NbImage:
            if type((cv2.imread(self.Path + '/' + self.Generic_name + str(i) + '.png'))) != type(None):
                print(self.Generic_name + str(i) + '.png loaded')
                list.append(cv2.imread(self.Path + '/' + self.Generic_name + str(i) + '.png'))
            elif i == 100:
                print('No .png file found until '+ self.Generic_name + str(100)+'. Research abandoned.\nExit program')
                sys.exit()
            else:
                print(self.Generic_name + str(i) + '.png not found')
            i+=1
        return list