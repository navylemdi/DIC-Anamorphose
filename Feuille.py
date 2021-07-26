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
        self.Centre=np.array([d, 0, 0])
        
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
                    print("Il n'existe pas de solution pour les cercles. Vérifier l'equation de la surface")
                    self.Pntprojection[i][j,:] = [None]*3
            sys.stdout.flush()
        print('\nFin calcul projection')
        end = time.time()
        print('Temps ecoulé: ', time.strftime("%H:%M:%S", time.gmtime(end-start)))
        sol4 = 0
        try:
            sol4 = max(solve( F.subs([(x, self.Centre[0]/delta1), (y, self.Centre[1]/delta1), (z, self.Centre[2]/delta1)]), delta1))
        except:
            print("Il n'existe pas de solution pour le centre du cadre. Vérifier l'equation de la surface")
        self.PntCentreCadreProjection = (self.Centre/sol4).astype(float)#delta[:,:,None]# Coordonnées des points projetés
        return self.Pntprojection,  self.PntCentreCadreProjection
    
    # def depliage(self, saut, x ,y ,z, GradF, ProjVector, Surface):
    #     self.UnfoldedPnt = [None]*len(self.contours)
    #     print('Début dépliage')
    #     start = time.time()
    #     if Surface.SurfaceType == 'Plan':
    #         for i in range(self.debut, len(self.contours), saut):
    #             self.UnfoldedPnt[i] = np.empty( [len(self.contours[i]), 3], dtype=np.float32)
    #             sys.stdout.write('\r' + str(round((i/(len(self.contours)-1))*100,2)) + '% ')#Affichage pourcentage de l'avancement
            
    #             for j in range(len(self.contours3D[i])):
    #                 NormalVector = np.array(GradF.subs([(x, self.Pntprojection[i][j, 0]), (y, self.Pntprojection[i][j, 1]), (z, self.Pntprojection[i][j, 2])])).astype(np.float64)/np.linalg.norm(np.array(GradF.subs([(x, self.Pntprojection[i][j, 0]), (y, self.Pntprojection[i][j, 1]), (z, self.Pntprojection[i][j, 2])])).astype(np.float64))
    #                 v = np.cross(np.squeeze(NormalVector), ProjVector)
    #                 c = np.dot(np.squeeze(NormalVector), ProjVector)
    #                 kmat = np.array([[0, -v[2], v[1]], 
    #                                   [v[2], 0, -v[0]], 
    #                                   [-v[1], v[0], 0]])
    #                 rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * (1/(1+c))
    #                 self.UnfoldedPnt[i][j, :] = np.dot(rotation_matrix, self.Pntprojection[i][j, :])
            
    #     elif Surface.SurfaceType == 'Cylindre':
    #         ProjVector2 = np.array([1, 0, 0])#Vecteur horizontal vers les positifs
    #         ProjVector3 = np.array([0, 0, 1])#Vecteur vertical vers les positifs
    #         # OrientedPnt = np.zeros((len(C),len(t),3))
    #         # RolledPnt = np.zeros((len(C),len(t),3))
    #         # NormalVector = np.zeros((len(C),len(t),3))
    #         CylAxe = np.array([Surface.a, Surface.b, Surface.c])/np.linalg.norm(np.array([Surface.a, Surface.b, Surface.c]))
    #         v = np.cross(CylAxe, ProjVector2)
    #         cos = np.dot(CylAxe, ProjVector2)
    #         kmat = np.array([[0, -v[2], v[1]], 
    #                          [v[2], 0, -v[0]], 
    #                          [-v[1], v[0], 0]])
    #         rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * (1/(1+cos)) #Rotation entre l'axe du cylindre et l'horizontal v
    #         VecteurOrientation = np.squeeze(np.array(GradF.subs([(x, self.PntCentreCadreProjection[0]), (y, self.PntCentreCadreProjection[1]), (z, self.PntCentreCadreProjection[2])])).astype(np.float64)/np.linalg.norm(np.array(GradF.subs([(x, self.PntCentreCadreProjection[0]), (y, self.PntCentreCadreProjection[1]), (z, self.PntCentreCadreProjection[2])])).astype(np.float64)))#Pntprojection[i, j, :]#
    #         VecteurOrientationRotation = np.dot(rotation_matrix, VecteurOrientation)/np.linalg.norm(np.dot(rotation_matrix, VecteurOrientation))
    #         v2 = np.cross(VecteurOrientationRotation, ProjVector3)
    #         cos2 = np.dot(VecteurOrientationRotation, ProjVector3)
    #         kmat2 = np.array([[0, -v2[2], v2[1]], 
    #                           [v2[2], 0, -v2[0]], 
    #                           [-v2[1], v2[0], 0]], dtype='float64')
    #         roulement_matrix = np.eye(3) + kmat2 + kmat2.dot(kmat2) * (1/(1+cos2))
    #         for i in range(self.debut, len(self.contours), saut):
    #             self.UnfoldedPnt[i] = np.empty( [len(self.contours[i]), 3], dtype=np.float32)
    #             sys.stdout.write('\r' + str(round((i/(len(self.contours)-1))*100,2)) + '% ')#Affichage pourcentage de l'avancement
                
    #             #VecteurOrientationRotationRoulement = np.dot(roulement_matrix, VecteurOrientationRotation)/np.linalg.norm(np.dot(roulement_matrix, VecteurOrientationRotation))
    #             for j in range(len(self.contours3D[i])):
    #                     OrientedPnt = np.dot(rotation_matrix, self.Pntprojection[i][j, :])#On tourne le cylindre pour l'aligner avec l'horizontale
    #                     RolledPnt= np.dot(roulement_matrix, OrientedPnt)#On tourne le cylindre pour l'aligner avec l'horizontale
    #                     #print(OrientedPnt[i, j, :])
    #                     NormalVector = np.array(GradF.subs([(x, self.Pntprojection[i][j, 0]), (y, self.Pntprojection[i][j, 1]), (z, self.Pntprojection[i][j, 2])])).astype(np.float64)/np.linalg.norm(np.array(GradF.subs([(x, self.Pntprojection[i][j, 0]), (y, self.Pntprojection[i][j, 1]), (z, self.Pntprojection[i][j, 2])])).astype(np.float64))
    #                     NormalVector = np.dot(rotation_matrix, np.squeeze(NormalVector))#Vecteur normaux à la surface tournée à l'horizontale
    #                     NormalVector = np.dot(roulement_matrix, np.squeeze(NormalVector))#Vecteur normaux à la surface tournée à la verticale
    #                     v2 = np.cross(NormalVector, ProjVector3)#Calcul des angles avec la verticale
    #                     theta = np.arcsin(v2[0])
    #                     #print(theta*180/np.pi)
    #                     #print(v2*180/np.pi)
    #                     #print(np.arcsin(v)[0]*180/np.pi)
    #                     #print(np.arcsin(v)[1]*180/np.pi)
    #                     #print(rotation_matrix)
    #                     #print(rotation_matrix.dot(NormalVector[i,j,:]))
    #                     self.UnfoldedPnt[i][j, :] = [RolledPnt[0], Surface.Radius*theta, 0]#np.squeeze(B.dot(A))#np.dot(rotation_matrix, Pntprojection[i, j, :])#np.dot(rotation_matrix, Pntprojection[i, j, :])#
    #                     #print(UnfoldedPnt[i, j, :])
    #                     #print('-----')
    #         sys.stdout.flush()
    #     print('\nFin dépliage')
    #     end = time.time()
    #     print('Temps ecoulé: ', time.strftime("%H:%M:%S", time.gmtime(end-start)))
    #     return self.UnfoldedPnt
    
    def Affichage_reference(self, saut, n, gcolor):
        fig=plt.figure(n)
        ax = fig.add_subplot(111, aspect='equal')
        for i in range(self.debut, len(self.contours), saut):
            plt.plot(self.contours[i][:, 0][:, 0], self.contours[i][:, 0][:, 1], marker=None, color=gcolor)
            ax.fill(self.contours[i][:, 0][:, 0], self.contours[i][:, 0][:, 1], gcolor,zorder=10)
        plt.title('Image référence '+ str(n) +' (pix)')
