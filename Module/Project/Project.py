import yaml
import os
import sys
import numpy as np
from Module.deck.data import Deck
from Module.Speckle.Speckle import Speckle
from Module.Camera.Camera import Camera
from Module.Plot.Plot import Plot
import matplotlib.pyplot as plt

class Project:
    """A class to save and open anamorphose projects
    
    Methods
    -------
    save
        Save all project's data
    open
        Load saved data
    Plot3D
        Plots in a 3D plot the camera position, speckle sheets position, anamorphic speckle sheets position
    PlotReference
        Plots all the input speckle sheets
    PlotUnfolded
        Plots the unfolded anamorphic speckles
    Show_plots
        Shows the plots
    """
    def __init__(self):
        pass

    def save(self, Project_name, path, inputdeck, List_Unfolded_0, WingFrameUnfolded, yf, zf, InputListe_projection):
        """
        Parameters
        ----------
        Project_name : str
            Name of the yaml project file
        path : str
            Path to the yaml project file
        inputdeck : Module.deck.data.Deck
            Deck of the current project
        List_Unfolded_0 : list
            List of position of unfolded speckle
        WingFrameUnfolded : numpy.ndarray
            Position of frame of the wing
        yf : numpy.ndarray
            Array of a y coordinate of mesh inside the wing frame
        zf : numpy.ndarray
            Array of a x coordinate of mesh inside the wing frame
        InputListe_projection : list
            List of position of projected speckle
        """

        self.Project_name = Project_name
        self.path = path
        os.chdir(path)
        bashCommand = "touch " + Project_name + ".yaml"
        print("save project " + Project_name)
        Liste_Projection = InputListe_projection.copy()
        for i in range(inputdeck.NbImage):
            for j in range(len(InputListe_projection[i][:])):
                if type(InputListe_projection[i][j]) is np.ndarray:
                    Liste_Projection[i][j] = InputListe_projection[i][j].tolist()
                else:
                    Liste_Projection[i][j] = InputListe_projection[i][j]
        List_Unfolded = List_Unfolded_0.copy()
        for i in range(inputdeck.NbImage):
            for j in range(len(List_Unfolded_0[i][:])):
                if type(List_Unfolded_0[i][j]) is np.ndarray:
                    List_Unfolded[i][j] = List_Unfolded_0[i][j].tolist()
                else:
                    List_Unfolded[i][j] = List_Unfolded_0[i][j]

        data = dict(Deck = {'Camera': {'Focal_length': inputdeck.Focal_length, 'Sensor_height': inputdeck.Sensor_height},

                    'Input_speckle': {'Step': inputdeck.Step,
                    'Begining':  inputdeck.Begining,
                    'Height': inputdeck.Height,
                    'Width': inputdeck.Width,
                    'Path': inputdeck.Path,
                    'Generic_name': inputdeck.Generic_name,
                    'NbImage': inputdeck.NbImage,
                    'Position_centre': inputdeck.Position_centre.tolist()},

                    'Surface': {'a': inputdeck.a,
                    'b': inputdeck.b,
                    'c': inputdeck.c,
                    'Radius': inputdeck.Radius,
                    'Position': inputdeck.Position.tolist(),
                    'Surface_type': inputdeck.Surface_type,
                    'Wingframe': inputdeck.Wingframe.tolist()},

                    'Output_speckle': {'Height_printable': inputdeck.Height_printable,
                    'Width_printable': inputdeck.Width_printable,
                    'Print_path': inputdeck.Print_path}},
                    Unfolded = {'List_Unfolded' : List_Unfolded, 'WingFrameUnfolded' : WingFrameUnfolded.tolist()},
                    yf = yf.tolist(),
                    zf = zf.tolist(),
                    Liste_Projection = Liste_Projection)


        with open(self.path+'/'+self.Project_name+'.yaml', 'w') as yaml_file:
            yaml.dump(data, yaml_file, default_flow_style=False)

    def open(self, Project_path):
        """
        Parameters
        ----------
        Project_path : str
            Path to the project data
        """
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
                for j in range(self.deck.NbImage):
                    for i in range(len(self.List_Unfolded_0[j])):
                        self.List_Unfolded_0[j][i] = np.array(self.List_Unfolded_0[j][i])
                self.WingFrameUnfolded = np.array(self.Unfolded['WingFrameUnfolded'])
                self.yf = np.array(self.doc['yf'])
                self.zf = np.array(self.doc['zf'])
                self.projection = np.array(self.doc['Liste_Projection'], dtype=object)
                for j in range(self.deck.NbImage):
                    for i in range(len(self.projection[j])):
                        self.projection[j][i] = np.array(self.projection[j][i])

    def PlotReference(self):
        """"""
        p = Plot()
        p.PlotReference(self.deck, self.List_Sheets)

    def Plot3D(self):
        """"""
        p = Plot()
        p.Plot3D(self.deck, self.List_Sheets, self.projection, self.camera)

    def PlotUnfolded(self):
        """"""
        p = Plot()
        p.PlotUnfolded(self.deck, self.List_Sheets, self.List_Unfolded_0, self.WingFrameUnfolded, self.yf, self.zf)
    
    def Show_plots(self):
        """"""
        plt.show()