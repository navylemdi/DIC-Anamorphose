import matplotlib.pyplot as plt
import numpy as np

class Plot:
    """A class to plot the results of anamorphose
    
    Methods
    -------
    set_aspect_equal_3d
        Fix equal aspect for 3D plots
    Plot3D
        Plots in a 3D plot the camera position, speckle sheets position, anamorphic speckle sheets position
    PlotReference
        Plots all the input speckle sheets
    PlotUnfolded
        Plots the unfolded anamorphic speckles
    Show_plots
        Shows the plots
    """
    def __init__(self) -> None:
        print("Display of graphics..")
        pass

    def set_aspect_equal_3d(self, ax):
        """
        Parameters
        ----------
        ax : matplotlib.pyplot.axis
            Axis of the 3D plot
        """

        xlim = ax.get_xlim3d()
        ylim = ax.get_ylim3d()
        zlim = ax.get_zlim3d()

        from numpy import mean
        xmean = mean(xlim)
        ymean = mean(ylim)
        zmean = mean(zlim)

        plot_radius = max([abs(lim - mean_)
                        for lims, mean_ in ((xlim, xmean),
                                            (ylim, ymean),
                                            (zlim, zmean))
                        for lim in lims])

        ax.set_xlim3d([xmean - plot_radius, xmean + plot_radius])
        ax.set_ylim3d([ymean - plot_radius, ymean + plot_radius])
        ax.set_zlim3d([zmean - plot_radius, zmean + plot_radius])
    
    def Plot3D(self, deck, Liste_Feuille, Liste_Projection, Camera):
        """
        Parameters
        ----------
        deck : Module.deck.data.Deck
            Variable that contains input data
        Liste_Feuille : list
            List of speckle sheets
        Liste_Projection : list
            List of projected speckle sheets
        Camera : Module.Camera.Camera.Camera
            Camera informations

        Methods
        -------
        plan
            Meshgrid of a plane
        cylindre
            Meshgrid of a cylinder
        """

        def plan(a, b, c, Pos, Wingframe):
            """
            Parameters
            ----------
            a : float
                X coordinate of normal vector's plane
            b : float
                Y coordinate of normal vector's plane
            c : float
                Z coordinate of normal vector's plane
            Pos : numpy.ndarray
                3D position a a point that's on the plane
            Wingframe : numpy.ndarray
                4 points that define the frame of the wing
            
            Returns
            -------
            x, y, z : numpy.ndarray
                Meshgrid of a plane
            """
            xmin = min(Wingframe[0,0],Wingframe[1,0])
            xmax = max(Wingframe[0,0],Wingframe[1,0])
            ymin = min(Wingframe[2,1],Wingframe[3,1])
            ymax = max(Wingframe[2,1],Wingframe[3,1])
            stepx = (xmax-xmin)/2
            stepy = (ymax-ymin)/2
            x, y = np.meshgrid(np.arange(xmin, xmax+stepx, stepx), np.arange(ymin, ymax+stepy, stepy))
            z = (Pos[0]*a+Pos[1]*b+Pos[2]*c-b*y-a*x)/c
            return x, y, z

        def cylindre(a, b, c, Pos, R, Wingframe):
            """
            Parameters
            ----------
            a : float
                X coordinate of cylinder axis
            b : float
                Y coordinate of cylinder axis
            c : float
                Z coordinate of cylinder axis
            Pos : numpy.ndarray
                3D position a a point that's on the cylinder axis
            Wingframe : numpy.ndarray
                4 points that define the frame of the wing
            
            Returns
            -------
            x, y, z : numpy.ndarray
                Meshgrid of a plane
            """
            p0 = Wingframe[0,:]
            p1 = Wingframe[1,:]
            #vector in direction of axis
            v = np.array([a,b,c])
            #find magnitude of vector
            mag = np.linalg.norm(p1-p0)
            #unit vector in direction of axis
            v = v / np.linalg.norm(v)
            #make some vector not in the same direction as v
            not_v = np.array([1, 0, 0])
            if (v == not_v).all():
                not_v = np.array([0, 1, 0])
            #make vector perpendicular to v
            n1 = np.cross(v, not_v)
            #normalize n1
            n1 /= np.linalg.norm(n1)
            #make unit vector perpendicular to v and n1
            n2 = np.cross(v, n1)
            #surface ranges over t from 0 to length of axis and 0 to 2*pi
            t = np.linspace(-mag/2,mag/2, 100)
            theta = np.linspace(-np.pi/2, np.pi/2, 50)
            #use meshgrid to make 2d arrays
            t, theta = np.meshgrid(t, theta)
            #generate coordinates for surface
            x, y, z = [Pos[i] + v[i] * t + R * np.sin(theta) * n1[i] + R * np.cos(theta) * n2[i] for i in [0, 1, 2]]
            return x, y, z

        def cone(Wingframe, alpha):
            zend = min(Wingframe[0,2],Wingframe[1,2])
            xend = max(Wingframe[:,0])
            if Wingframe[0,2] == zend:
                v = Wingframe[0,:]
                #deltax = Wingframe[0,0]/Wingframe[1,0]
                #deltay = Wingframe[0,1]/Wingframe[1,1]
                delta = Wingframe[0,0]/Wingframe[1,0]
            if Wingframe[1,2] == zend:
                v = Wingframe[1,:]
                #deltax = Wingframe[1,0]/Wingframe[0,0]
                #deltay = Wingframe[1,1]/Wingframe[0,1]
                delta = Wingframe[1,0]/Wingframe[0,0]
            #rotationz = np.array([[np.cos(-alpha), -np.sin(-alpha), 0],
                                #[np.sin(-alpha), np.cos(-alpha),  0],
                                #[0,            0,                   1]], np.float32)
            def rotationy(alpha):
                return np.array([[np.cos(alpha), 0, -np.sin(alpha)],
                                [0,                 1,              0],
                                [np.sin(alpha),     0,  np.cos(alpha)]], np.float32)
            def rotationz(alpha):
                return np.array([[np.cos(alpha), -np.sin(alpha), 0],
                                [np.sin(alpha), np.cos(alpha),  0],
                                [0,            0,                   1]], np.float32)

            axeoptique = np.array([1,0,0])*xend
            v = np.dot(rotationy(alpha),axeoptique)
            v1 = np.dot(rotationy(-alpha),axeoptique)
            v2 = np.dot(rotationz(alpha), axeoptique)
            v3 = np.dot(rotationz(-alpha),axeoptique)
            return v,v1,v2,v3, axeoptique

        Nbimage = deck.NbImage
        CadreAile = deck.Wingframe
        v,v1,v2,v3,axeoptique = cone(CadreAile, Camera.fov/2)

        fig = plt.figure(Nbimage+1)
        ax = fig.add_subplot(111, projection='3d')

        if deck.Surface_type == 'Plan':
            x,y,z = plan(deck.a, deck.b, deck.c, deck.Position, CadreAile)
            ax.plot_surface(x, y, z, color='r', alpha=0.2)
        if deck.Surface_type == 'Cylindre':
            x,y,z = cylindre(deck.a, deck.b, deck.c, deck.Position, deck.Radius, CadreAile)
            ax.plot_surface(x, y, z, color='r', alpha=0.2)

        ax.scatter(0, 0, 0, color='b', label='Camera center')
        for j in range(Nbimage):
            for i in range (Liste_Feuille[j].debut, len(Liste_Feuille[j].contours), Liste_Feuille[j].saut):
                ax.plot(Liste_Feuille[j].contours3D[i][:, 0], Liste_Feuille[j].contours3D[i][:, 1], Liste_Feuille[j].contours3D[i][:, 2], color='k', marker=None)
                ax.plot(Liste_Projection[j][i][:, 0], Liste_Projection[j][i][:, 1], Liste_Projection[j][i][:, 2], color='k', marker=None)
                ax.scatter([Liste_Feuille[j].d]*4, Liste_Feuille[j].Cadre[:,0], Liste_Feuille[j].Cadre[:,1], color='k', marker='+')
        ax.scatter(CadreAile[:,0], CadreAile[:,1], CadreAile[:,2], color='c', label='Wingframe')
        for i in range (0,len(CadreAile[:,0])):
            ax.text(CadreAile[i,0], CadreAile[i,1], CadreAile[i,2], str(i+1))

        ax.plot([0,v[0]],[0,v[1]], [0,v[2]],  color='g', label='Field of view', linewidth= 1, alpha=0.5)
        ax.plot([0,v1[0]],[0,v1[1]], [0,v1[2]],  color='g', linewidth= 1, alpha=0.5)
        ax.plot([0,v2[0]],[0,v2[1]], [0,v2[2]],  color='g', linewidth= 1, alpha=0.5)
        ax.plot([0,v3[0]],[0,v3[1]], [0,v3[2]],  color='g', linewidth= 1, alpha=0.5)
        ax.plot([0, axeoptique[0]], [0, axeoptique[1]], [0, axeoptique[2]], '--k', linewidth= 1, label='Optical axis')
        ax.quiver(0,0,0,1/2,0,0,color='r', linewidth= 1)
        ax.quiver(0,0,0,0,1/2,0,color='g', linewidth= 1)
        ax.quiver(0,0,0,0,0,1/2,color='b', linewidth= 1)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend()
        ax.set_title('Reference speckle and projected in 3D (m)')
        self.set_aspect_equal_3d(ax)

    def PlotReference(self, deck, Liste_Feuille):
        """
        Parameters
        ----------
        deck : Module.deck.data.Deck
            Variable that contains input data
        Liste_Feuille : list
            List of speckle sheets
        """
        Nbimage =deck.NbImage
        for i in range(Nbimage):
            fig=plt.figure(i+1)
            ax = fig.add_subplot(111, aspect='equal')
            for j in range(Liste_Feuille[i].debut, len(Liste_Feuille[i].contours), Liste_Feuille[i].saut):
                ax.plot(Liste_Feuille[i].contours[j][:, 0][:, 0], Liste_Feuille[i].contours[j][:, 0][:, 1], marker=None, color='k')
                ax.fill(Liste_Feuille[i].contours[j][:, 0][:, 0], Liste_Feuille[i].contours[j][:, 0][:, 1], 'k', zorder=10)

            ax.set_title('Image référence '+ str(i+1) +' (pix)')

    def PlotUnfolded(self, deck, Liste_Feuille, Liste_depliage, CadreAileUnfolded, yf, zf):
        """
        Parameters
        ----------
        deck : Module.deck.data.Deck
            Variable that contains input data
        Liste_Feuille : list
            List of speckle sheets
        Liste_depliage : list
            List of unfolded speckle sheets
        CadreAileUnfolded : numpy.ndarray
            Array of unfolded position of the wing frame
        yf : numpy.ndarray
            Array of a y coordinate of mesh inside the wing frame
        zf : numpy.ndarray
            Array of a x coordinate of mesh inside the wing frame
        """
        Nbimage =deck.NbImage
        fig=plt.figure(Nbimage+2)
        ax = fig.add_subplot(111, aspect='equal')
        for j in range(Nbimage):
            for i in range(Liste_Feuille[j].debut, len(Liste_Feuille[j].contours), Liste_Feuille[j].saut):
                ax.plot(Liste_depliage[j][i][:, 1], Liste_depliage[j][i][:, 2], color='black')
                ax.fill(Liste_depliage[j][i][:, 1], Liste_depliage[j][i][:, 2], color='black')
        ax.scatter(CadreAileUnfolded[:,1], CadreAileUnfolded[:,2], color='c', marker='+')
        for i in range (len(CadreAileUnfolded[:,0])):
            ax.text(CadreAileUnfolded[i,1], CadreAileUnfolded[i,2], str(i+1), color='c')
        ax.scatter( yf, zf, marker='+', color='m')
        for i in range (yf.shape[0]-1):
            for j in range (yf.shape[1]-1):
                ax.text((yf[i,j]+yf[i,j+1])/2, (zf[i,j]+zf[i+1,j])/2, str((i+1)*(j+1)), color='black')
        ax.set_title('Dépliée')
        ax.set_xlim(min(CadreAileUnfolded[:,1]), max(CadreAileUnfolded[:,1]))
        ax.set_ylim(min(CadreAileUnfolded[:,2]), max(CadreAileUnfolded[:,2]))
        ax.grid()

    def Show_plots(self):
        plt.show()