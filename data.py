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

                self.Input_Speckle = self.doc['Input_Speckle']
                self.step = int(self.Input_Speckle['step'])
                self.begining = int(self.Input_Speckle['begining'])
                self.height = float(self.Input_Speckle['height'])
                self.width = float(self.Input_Speckle['width'])
                self.path = self.Input_Speckle['Path']
                self.generic_name = self.Input_Speckle['Generic_Name']
                self.NbImage = int(self.Input_Speckle['NbImage'])
                self.PositionCentre = np.array(self.Input_Speckle['PositionCentre'], dtype = float)

                self.Surface = self.doc['Surface']
                self.a = float(self.Surface['a'])
                self.b = float(self.Surface['b'])
                self.c = float(self.Surface['c'])
                self.Position = np.array(self.Surface['Position'], dtype = float)
                self.radius = float(self.Surface['Radius'])
                self.SurfaceType = self.Surface['Surface_Type']

                self.Output_Speckle = self.doc['Output_Speckle']
                self.heightPrintable = float(self.Output_Speckle['heightPrintable'])
                self.widthPrintable = float(self.Output_Speckle['widthPrintable'])
                self.PrintPath = self.Output_Speckle['PrintPath']
    def Images(self):
        list=[]
        i=0
        while len(list) < self.NbImage:
            if type((cv2.imread(self.path + self.generic_name + str(i) + '.png'))) != type(None):
                print(self.generic_name + str(i) + '.png loaded')
                list.append(cv2.imread(self.path + self.generic_name + str(i) + '.png'))
            else:
                print(self.generic_name + str(i) + '.png not found')
            i+=1
        return list
#else (print(self.path + self.generic_name + str(i) + '.png not found'))