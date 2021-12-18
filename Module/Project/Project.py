from numpy.core.shape_base import block
import yaml
import subprocess
import os
import sys
import numpy as np
from Module.deck.data import Deck
from Module.Speckle.Speckle import Speckle
from Module.Camera.Camera import Camera
from Module.Plot.Plot import Plot
import matplotlib.pyplot as plt

class Project:
    def __init__(self):
        pass

    def save(self, Project_name, path, deck, List_Unfolded_0, WingFrameUnfolded, yf, zf, Liste_projection):# yf, zf, camera, Liste_Projection):
        self.Project_name = Project_name
        self.path = path
        os.chdir(path)
        bashCommand = "touch " + Project_name + ".yaml"
        process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
        output, error = process.communicate()
        data = dict(Deck = {'Camera': {'Focal_length': deck.Focal_length, 'Sensor_height': deck.Sensor_height},

                    'Input_speckle': {'Step': deck.Step,
                    'Begining': deck.Begining,
                    'Height': deck.Height,
                    'Width': deck.Width,
                    'Path': deck.Path,
                    'Generic_name': deck.Generic_name,
                    'NbImage': deck.NbImage,
                    'Position_centre': deck.Position_centre.tolist()},

                    'Surface': {'a': deck.a,
                    'b': deck.b,
                    'c': deck.c,
                    'Radius': deck.Radius,
                    'Position': deck.Position.tolist(),
                    'Surface_type': deck.Surface_type,
                    'Wingframe': deck.Wingframe.tolist()},

                    'Output_speckle': {'Height_printable': deck.Height_printable,
                    'Width_printable': deck.Width_printable,
                    'Print_path': deck.Print_path}},
                    Unfolded = {'List_Unfolded' : [[List_Unfolded_0[0][i].tolist() if type(List_Unfolded_0[0][i]) is np.ndarray else List_Unfolded_0[0][i] for i in range(len(List_Unfolded_0[0][:]))]], 'WingFrameUnfolded' : WingFrameUnfolded.tolist()},
                    yf = yf.tolist(),
                    zf = zf.tolist(),
                    Liste_Projection = [[Liste_projection[0][i].tolist() if type(Liste_projection[0][i]) is np.ndarray else Liste_projection[0][i] for i in range(len(Liste_projection[0][:]))]])

        with open(self.path+'/'+self.Project_name+'.yaml', 'w') as yaml_file:
            yaml.dump(data, yaml_file, default_flow_style=False)

    def open(self, Project_path):
        Project_Name = Project_path[Project_path.rfind('/')+1:]
        print('Open ' + Project_Name)
        self.Project_path = Project_path
        if not os.path.exists(self.Project_path):
            print("File " + self.Project_path + " not found")
            sys.exit(1)
        else:
            self.deck = Deck(Project_path)
            self.List_Sheets = Speckle(self.deck).List_Sheets
            self.camera = Camera(self.deck)
            with open(self.Project_path, 'r') as f:
                self.doc = yaml.load(f, Loader=yaml.UnsafeLoader)
                self.Unfolded = self.doc['Unfolded']
                self.List_Unfolded_0 = np.array(self.Unfolded['List_Unfolded'], dtype=object)
                for i in range(len(self.List_Unfolded_0[0])):
                    self.List_Unfolded_0[0][i] = np.array(self.List_Unfolded_0[0][i])
                self.WingFrameUnfolded = np.array(self.Unfolded['WingFrameUnfolded'])
                self.yf = np.array(self.doc['yf'])
                self.zf = np.array(self.doc['zf'])
                self.projection = np.array(self.doc['Liste_Projection'], dtype=object)
                for i in range(len(self.projection[0])):
                    self.projection[0][i] = np.array(self.projection[0][i])

    def PlotReference(self):
        p = Plot()
        p.PlotReference(self.deck, self.List_Sheets)

    def Plot3D(self):
        p = Plot()
        p.Plot3D(self.deck, self.List_Sheets, self.projection, self.camera)

    def PlotUnfolded(self):
        p = Plot()
        p.PlotUnfolded(self.deck, self.List_Sheets, self.List_Unfolded_0, self.WingFrameUnfolded, self.yf, self.zf)
    
    def Show_plots(self):
        plt.show()