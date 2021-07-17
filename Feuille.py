#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  6 09:26:15 2021

@author: yvan
"""
import cv2 as cv2
import numpy as np
import Fonction
import time
import sys
from sympy.solvers import solve
import matplotlib.pyplot as plt

class Feuille:
    
    def __init__(self, centreH, centreV, image, height, width, debut, saut, d):
        self.centreH = centreH
        self.centreV = centreV
        self.height = height
        self.width = width
        self.debut = debut
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
        for i in range(debut, len(self.contours), saut):
            '''Opencv met l'ordonnée en premiere position des tableau et l'abscisse en seconde
            C'est pour ca que les indices semblent inversés (H<->V) mais c'est normal'''
            self.contours3D[i] = np.empty( [len(self.contours[i]), 3], dtype=np.float32)
            
            temp = Fonction.Pix2Meter(self.contours[i][:, 0], image,  -height/2, height/2,
                              -width/2, width/2, centreV, centreH)
            self.contours3D[i][:, 0] = d
            self.contours3D[i][:, 1] = temp[:, 1]
            self.contours3D[i][:, 2] = temp[:, 0]
        
    def projection(self, saut, F, x, y, z, delta1): 
        #Calcul projection sur plan incliné
        self.Pntprojection = [None]*len(self.contours)
        print('Début calcul projection')
        start = time.time()
        for i in range(self.debut, len(self.contours), saut):
            self.Pntprojection[i] = np.empty( [len(self.contours[i]), 3], dtype=np.float32)
            sys.stdout.write('\r' + str(round((i/(len(self.contours)-1))*100,2)) + '% ')#Affichage pourcentage de l'avancement
            for j in range(len(self.contours3D[i])):
                try:
                    sol = max(solve( F.subs([(x, self.contours3D[i][j,0]/delta1), (y, self.contours3D[i][j,1]/delta1), (z, self.contours3D[i][j,2]/delta1)]), delta1))#Résolution de l'equation f(x/delta, y/delta, z/delta) = 0 avec delta l'inconnue
                    #On prends le max pour avoir le point le plus proche de la caméra
                    self.Pntprojection[i][j,:] = self.contours3D[i][j,:]/sol# Coordonnées dans l'espace des points projetés
                except (IndexError, ValueError):
                    print("\nIl n'existe pas de solution pour les cercles. Vérifier l'equation de la surface")
                    self.Pntprojection[i][j,:] = [None]*3
            sys.stdout.flush()
        print('\nFin calcul projection')
        end = time.time()
        print('Temps ecoulé: ', time.strftime("%H:%M:%S", time.gmtime(end-start)))
        return self.Pntprojection
    
    def depliage(self, saut, x ,y ,z, GradF, ProjVector):
        self.UnfoldedPnt = [None]*len(self.contours)
        print('Début dépliage')
        start = time.time()
        for i in range(self.debut, len(self.contours), saut):
            self.UnfoldedPnt[i] = np.empty( [len(self.contours[i]), 3], dtype=np.float32)
            sys.stdout.write('\r' + str(round((i/(len(self.contours)-1))*100,2)) + '% ')#Affichage pourcentage de l'avancement
            for j in range(len(self.contours3D[i])):
                NormalVector = np.array(GradF.subs([(x, self.Pntprojection[i][j, 0]), (y, self.Pntprojection[i][j, 1]), (z, self.Pntprojection[i][j, 2])])).astype(np.float64)/np.linalg.norm(np.array(GradF.subs([(x, self.Pntprojection[i][j, 0]), (y, self.Pntprojection[i][j, 1]), (z, self.Pntprojection[i][j, 2])])).astype(np.float64))
                v = np.cross(np.squeeze(NormalVector), ProjVector)
                c = np.dot(np.squeeze(NormalVector), ProjVector)
                kmat = np.array([[0, -v[2], v[1]], 
                                  [v[2], 0, -v[0]], 
                                  [-v[1], v[0], 0]])
                rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * (1/(1+c))
                self.UnfoldedPnt[i][j, :] = np.dot(rotation_matrix, self.Pntprojection[i][j, :])
            sys.stdout.flush()
        print('\nFin dépliage')
        end = time.time()
        print('Temps ecoulé: ', time.strftime("%H:%M:%S", time.gmtime(end-start)))
        return self.UnfoldedPnt
    
    def Affichage_reference(self, saut, n, gcolor):
        fig=plt.figure(n)
        ax = fig.add_subplot(111, aspect='equal')
        for i in range(self.debut, len(self.contours), saut):
            plt.plot(self.contours[i][:, 0][:, 0], self.contours[i][:, 0][:, 1], marker=None, color=gcolor)
            ax.fill(self.contours[i][:, 0][:, 0], self.contours[i][:, 0][:, 1], gcolor,zorder=10)
        plt.title('Image référence '+ str(n) +' (pix)')
        plt.show()