import cv2 as cv2
import numpy as np
import Fonction
import time
import sys
from sympy.solvers import solve
import matplotlib.pyplot as plt
from sympy import Symbol

class Sheets:
    
    def __init__(self, centreH, centreV, d, image, height, width, debut, saut):
        self.centreH = centreH
        self.centreV = centreV
        self.d = d
        self.height = height
        self.width = width
        self.debut = debut
        self.saut = saut
        if type(image) != np.ndarray :
            sys.exit("Image file format has to be .png to be readable")
        Pospix = np.array([[0, 0],
                           [0, image.shape[0]],
                           [image.shape[1], 0],
                           [image.shape[1], image.shape[0]]])
        self.Cadre = Fonction.Pix2Meter(Pospix, image, -width/2, width/2, height/2, -height/2, centreH, centreV)
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        #Conversion en NB
        ret,thresh = cv2.threshold(image_gray, 127, 255, cv2.THRESH_BINARY)
        self.contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, 
                                     cv2.CHAIN_APPROX_TC89_KCOS)#Trouver contours
        
        # Transformation coordonées contours repère 2D en repère 3D
        self.contours3D = [None]*len(self.contours)
        for i in range(debut, len(self.contours), self.saut):
            '''Opencv met l'ordonnée en premiere position des tableau et l'abscisse en seconde
            C'est pour ca que les indices semblent inversés (H<->V) mais c'est normal'''
            self.contours3D[i] = np.empty( [len(self.contours[i]), 3], dtype=np.float32)
            
            temp = Fonction.Pix2Meter(self.contours[i][:, 0], image,  -width/2, width/2,
                              -height/2, height/2, centreH, centreV)
            self.contours3D[i][:, 0] = self.d
            self.contours3D[i][:, 1] = temp[:, 0]
            self.contours3D[i][:, 2] = temp[:, 1]
        self.Centre=np.array([d, 0, 0])
        
    def projection(self, Surface):
        F = Surface.Equation() 
        x, y, z = Symbol('x'), Symbol('y'), Symbol('z')
        delta = Symbol('delta', positive=True)
        #Calcul projection sur plan incliné
        self.Pntprojection = [None]*len(self.contours)
        print('Start projection calculation')
        start = time.time()
        for i in range(self.debut, len(self.contours), self.saut):
            self.Pntprojection[i] = np.empty( [len(self.contours[i]), 3], dtype=np.float32)
            sys.stdout.write('\r' + str(round((i/(len(self.contours)-1))*100,2)) + '% ')#Affichage pourcentage de l'avancement
            for j in range(len(self.contours3D[i])):
                try:
                    sol = max(solve( F.subs([(x, self.contours3D[i][j,0]/delta), (y, self.contours3D[i][j,1]/delta), (z, self.contours3D[i][j,2]/delta)]), delta))#Résolution de l'equation f(x/delta, y/delta, z/delta) = 0 avec delta l'inconnue
                    #On prends le max pour avoir le point le plus proche de la caméra
                    self.Pntprojection[i][j,:] = self.contours3D[i][j,:]/sol# Coordonnées dans l'espace des points projetés
                except (IndexError, ValueError):
                    print("No solution found for the dots.\n")
                    self.Pntprojection[i][j,:] = [None]*3
            sys.stdout.flush()
        print('\nEnd projection calculation')
        end = time.time()
        print('Elapsed time: ', time.strftime("%H:%M:%S", time.gmtime(end-start)))
        sol4 = 0
        try:
            sol4 = max(solve( F.subs([(x, self.Centre[0]/delta), (y, self.Centre[1]/delta), (z, self.Centre[2]/delta)]), delta))
        except:
            print("No solution found for center frame.\n")
        self.PntCentreCadreProjection = (self.Centre/sol4).astype(float)#delta[:,:,None]# Coordonnées des points projetés
        return self.Pntprojection,  self.PntCentreCadreProjection

    def Affichage_reference(self, n, gcolor):
        fig=plt.figure(n)
        ax = fig.add_subplot(111, aspect='equal')
        for i in range(self.debut, len(self.contours), self.saut):
            plt.plot(self.contours[i][:, 0][:, 0], self.contours[i][:, 0][:, 1], marker=None, color=gcolor)
            ax.fill(self.contours[i][:, 0][:, 0], self.contours[i][:, 0][:, 1], gcolor,zorder=10)
        plt.title('Reference image '+ str(n) +' (pix)')
