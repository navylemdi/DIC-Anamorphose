import matplotlib.pyplot as plt
import Fonction
import Sheets
class Plot:

    def Plot3D(Nbimage, Liste_Feuille, Liste_Projection, CadreAile):
        fig = plt.figure(Nbimage+1)
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(0, 0, 0, color='b')
        for j in range(Nbimage):
            for i in range (Liste_Feuille[j].debut, len(Liste_Feuille[j].contours), Liste_Feuille[j].saut):
                ax.plot(Liste_Feuille[j].contours3D[i][:, 0], Liste_Feuille[j].contours3D[i][:, 1], Liste_Feuille[j].contours3D[i][:, 2], color='k', marker=None)
                ax.plot(Liste_Projection[j][i][:, 0], Liste_Projection[j][i][:, 1], Liste_Projection[j][i][:, 2], color='k', marker=None)
                ax.scatter([Liste_Feuille[j].d]*4, Liste_Feuille[j].Cadre[:,0], Liste_Feuille[j].Cadre[:,1], color='k', marker='+')
        ax.scatter(CadreAile[:,0], CadreAile[:,1], CadreAile[:,2], color='c')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.title('Image référence et projetée 3D (m)')
        Fonction.set_aspect_equal_3d(ax)

    def PlotReference(Nbimage, Liste_Feuille):
        for i in range(Nbimage):
            fig=plt.figure(i+1)
            ax = fig.add_subplot(111, aspect='equal')
            for j in range(Liste_Feuille[i].debut, len(Liste_Feuille[i].contours), Liste_Feuille[i].saut):
                plt.plot(Liste_Feuille[i].contours[j][:, 0][:, 0], Liste_Feuille[i].contours[j][:, 0][:, 1], marker=None, color='k')
                ax.fill(Liste_Feuille[i].contours[j][:, 0][:, 0], Liste_Feuille[i].contours[j][:, 0][:, 1], 'k', zorder=10)
            plt.title('Image référence '+ str(i+1) +' (pix)')

    def PlotUnfolded(Nbimage, Liste_Feuille, Liste_depliage, CadreAileUnfolded, yf, zf):
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