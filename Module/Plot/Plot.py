import matplotlib.pyplot as plt
import numpy as np

class Plot:
    def __init__(self) -> None:
        pass

    def set_aspect_equal_3d(self, ax):
        """Fix equal aspect bug for 3D plots."""

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
        
        def plan(a, b, c, Pos, Wingframe):
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
            rotationz = np.array([[np.cos(-alpha), -np.sin(-alpha), 0],
                                [np.sin(-alpha), np.cos(-alpha),  0],
                                [0,            0,                   1]], np.float32)
            rotation = np.array([[np.cos(alpha), 0, -np.sin(alpha)],
                                [0,                 1,              0],
                                [np.sin(alpha),     0,  np.cos(alpha)]], np.float32)
            axeoptique = v.copy()
            axeoptique = np.dot(rotation,v)
            axeoptiquetemp = axeoptique/np.linalg.norm(axeoptique)
            v1 = np.dot(rotation,axeoptique)/delta
            v2 = v*np.cos(np.pi/2) + np.cross(axeoptiquetemp,v)*np.sin(np.pi/2) + axeoptiquetemp*np.dot(axeoptiquetemp,v)*(1-np.cos(np.pi/2))
            v2 = v2/np.linalg.norm(v2)*np.linalg.norm(v1)
            v3 = np.dot(rotationz,axeoptique)
            v3 = v3/np.linalg.norm(v3)*np.linalg.norm(v1)
            return v,v1,v2,v3, axeoptique

        Nbimage = deck.NbImage
        CadreAile = deck.WingFrame
        v,v1,v2,v3,axeoptique = cone(CadreAile, Camera.fov/2)

        fig = plt.figure(Nbimage+1)
        ax = fig.add_subplot(111, projection='3d')

        if deck.SurfaceType == 'Plan':
            x,y,z = plan(deck.a, deck.b, deck.c, deck.Position, CadreAile)
            ax.plot_surface(x, y, z, color='r', alpha=0.2)
        if deck.SurfaceType == 'Cylindre':
            x,y,z = cylindre(deck.a, deck.b, deck.c, deck.Position, deck.radius, CadreAile)
            ax.plot_surface(x, y, z, color='r', alpha=0.2)

        ax.scatter(0, 0, 0, color='b', label='Camera center')
        for j in range(Nbimage):
            for i in range (Liste_Feuille[j].debut, len(Liste_Feuille[j].contours), Liste_Feuille[j].saut):
                ax.plot(Liste_Feuille[j].contours3D[i][:, 0], Liste_Feuille[j].contours3D[i][:, 1], Liste_Feuille[j].contours3D[i][:, 2], color='k', marker=None)
                ax.plot(Liste_Projection[j][i][:, 0], Liste_Projection[j][i][:, 1], Liste_Projection[j][i][:, 2], color='k', marker=None)
                ax.scatter([Liste_Feuille[j].d]*4, Liste_Feuille[j].Cadre[:,0], Liste_Feuille[j].Cadre[:,1], color='k', marker='+')
        ax.scatter(CadreAile[:,0], CadreAile[:,1], CadreAile[:,2], color='c', label='Wingframe')

        ax.plot([0,v[0]],[0,v[1]], [0,v[2]],  color='g', label='Field of view')
        ax.plot([0,v1[0]],[0,v1[1]], [0,v1[2]],  color='g')
        ax.plot([0,v2[0]],[0,v2[1]], [0,v2[2]],  color='g')
        ax.plot([0,v3[0]],[0,v3[1]], [0,v3[2]],  color='g')
        ax.plot([0, axeoptique[0]], [0, axeoptique[1]], [0, axeoptique[2]], '--k', linewidth= 1, label='Optical axis')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend()
        plt.title('Image référence et projetée 3D (m)')
        self.set_aspect_equal_3d(ax)

    def PlotReference(self, Nbimage, Liste_Feuille):
        for i in range(Nbimage):
            fig=plt.figure(i+1)
            ax = fig.add_subplot(111, aspect='equal')
            for j in range(Liste_Feuille[i].debut, len(Liste_Feuille[i].contours), Liste_Feuille[i].saut):
                plt.plot(Liste_Feuille[i].contours[j][:, 0][:, 0], Liste_Feuille[i].contours[j][:, 0][:, 1], marker=None, color='k')
                ax.fill(Liste_Feuille[i].contours[j][:, 0][:, 0], Liste_Feuille[i].contours[j][:, 0][:, 1], 'k', zorder=10)
            plt.title('Image référence '+ str(i+1) +' (pix)')

    def PlotUnfolded(self, Nbimage, Liste_Feuille, Liste_depliage, CadreAileUnfolded, yf, zf):
        fig=plt.figure(Nbimage+2)
        for j in range(Nbimage):
            for i in range(Liste_Feuille[j].debut, len(Liste_Feuille[j].contours), Liste_Feuille[j].saut):
                plt.plot(Liste_depliage[j][i][:, 1], Liste_depliage[j][i][:, 2], color='black')
                plt.fill(Liste_depliage[j][i][:, 1], Liste_depliage[j][i][:, 2], color='black')
        plt.scatter(CadreAileUnfolded[:,1], CadreAileUnfolded[:,2], color='c', marker='+')
        plt.scatter( yf, zf, marker='+', color='m')
        for i in range (yf.shape[0]-1):
            for j in range (yf.shape[1]-1):
                plt.text((yf[i,j]+yf[i,j+1])/2, (zf[i,j]+zf[i+1,j])/2, str((i+1)*(j+1)), color='black')
        plt.title('Dépliée')
        plt.axis('equal')
        plt.xlim(min(CadreAileUnfolded[:,1]), max(CadreAileUnfolded[:,1]))
        plt.ylim(min(CadreAileUnfolded[:,2]), max(CadreAileUnfolded[:,2]))
        plt.grid()

