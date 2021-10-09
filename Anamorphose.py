#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  6 09:34:39 2021

@author: yvan
"""
import numpy as np
import matplotlib.pyplot as plt
import sympy as sym
from sympy import Symbol
import cv2 as cv2
from Surface import Surface
from Feuille import Feuille
from Plot import Plot
import Fonction
import os
import glob
import sys

plt.close('all')

##-------------------------------CONSTANTES----------------------------------##

saut = 500 #Taille du saut de point dans la liste contours

debut = 3 #Debut des boucles for pour les projections

height = 27e-2#27e-2 #29.7e-2#hauteur en m de l'image de reference(m)
width = 21e-2#21e-2 #21e-2#largeur en m de l'image de reference(m)
WingWidth = 60e-2 #largeur zone analyse de l'aile (m)
WingHeight = 3 #hauteur zone analyse de l'aile (m)


heightPrintable = 27.9e-2
widthPrintable = 21.6e-2
PrintPath = '/Users/yvan/Desktop/ETS_montreal/Cours/E21/MTR892 - Projet technique/AnamorphosePlane/ImagePrintable'

image1 = cv2.imread("/Users/yvan/Desktop/ETS_montreal/Cours/E21/MTR892 - Projet technique/Banque_Speckle/2mm/Speckle_1.png")
image2 = cv2.imread("/Users/yvan/Desktop/ETS_montreal/Cours/E21/MTR892 - Projet technique/Banque_Speckle/2mm/Speckle_2.png")
List_image = [image1, image2]
Nbimage=len(List_image)

#Angles
#gamma = 80.0 #angle entre capteur et plan aile (deg)
#theta = 90.0-gamma #Angle entre normale capteur et plan aile (deg)
#alpha = 10.0#Angle de champ de vue 
#beta = 180.0-165.0-alpha/2.0 #Angle aigu entre aile et axe optique

#l = np.sqrt(0.9**2 + 2.5**2 + 0.9*2.5*np.cos(105*np.pi/180))

CentreH1 = 0#CadreAile[2,1]-width/2 #Position horizontale du centre du speckle de référence 1
CentreV1 = 0#CadreAile[0,2]+height/2 #Position verticale du centre du speckle de référence 1
CentreH2 = -width + CentreH1 #Position horizontale du centre du speckle de référence 2
CentreV2 = CentreV1 #Position verticale du centre du speckle de référence 2
# CentreH3 = CentreH1 #Position horizontale du centre du speckle de référence 3
# CentreV3 = height + CentreV1 #Position verticale du centre du speckle de référence 3
# CentreH4 = CentreH2 #Position horizontale du centre du speckle de référence 4
# CentreV4 = CentreV3 #Position verticale du centre du speckle de référence 4
# CentreH5 = CentreH2-width #Position horizontale du centre du speckle de référence 4
# CentreV5 = CentreV2 #Position verticale du centre du speckle de référence 4

Feuille_pos = np.array([[CentreH1, CentreV1],
                        [CentreH2,CentreV2]])
#Parametre Position aile
a = 0#-np.sin(theta*np.pi/180)# -0.02#
b = 0#-np.sin(theta*np.pi/180)#np.linalg.solve(D,E)[1]
c = 1#np.cos(theta*np.pi/180)#1#np.linalg.solve(D,E)[2]#
dprim = 1#a*A[0]**1+b*A[1]**1+c*A[2]
Radius = 0.4
Pos = np.array([3, 0, 0])

A = np.array([Pos[0]-Radius, 0, 0.2])#np.array([l*np.cos((alpha/2)*np.pi/180), 0, l*np.sin((-alpha/2)*np.pi/180)])
B = np.array([Pos[0]-Radius, 0, -0.2])#np.array([A[0] + (5.5-2.5)*np.cos(beta*np.pi/180), 0, A[2] + (5.5-2.5)*np.sin(beta*np.pi/180)])
C1 = np.array([[Pos[0], -Radius, 0],#(WingHeight)/2],
               [Pos[0], Radius, 0]])#(-WingHeight)/2]])
CadreAile = np.vstack((A, B, C1))#Points qui definissent les limites spatiales de l'aile

#Parametres Position Mouchetis Reference
xa = 1
ya = 0
za = 0
d = 2#A[0]#7.52928#


##------------------------------FIN CONSTANTES-------------------------------##

##---------------------------------SURFACE----------------------------------##

S = Surface(a, b, c, Pos[0], Pos[1], Pos[2], dprim, Radius, 'Cylindre')

##---------------------------------FEUILLES----------------------------------##

Liste_Feuille=[]
for i in range(Nbimage):
    Liste_Feuille.append(Feuille(Feuille_pos[i,0], Feuille_pos[i,1], List_image[i], height, width, debut, saut, d))

##-----------------------------FIN FEUILLES----------------------------------##

##--------------------------------PROJECTION---------------------------------##

Liste_Projection=[]
for i in range(Nbimage):
    Liste_Projection.append(Liste_Feuille[i].projection(saut, S)[0])#Coordonées des points de projection de la feuille1

##------------------------------FIN PROJECTION-------------------------------##

##--------------------------------DEPLIAGE-----------------------------------##
#On récupère le vecteur normal à la surface en un point donné puis on effectue
#une rotation de ce vecteur pour avoir un vecteur horizontal. Cette matrice de
#rotation est ensuite appliqué sur la position du point donné pour obtenir un
#point déplié sur un plan horizontal

ProjVector = np.array([-1, 0, 0])#Direction de dépliage de la surface 3D
Liste_depliage=[]
for i in range(Nbimage):
    Liste_depliage.append(Fonction.depliage(Liste_Feuille[i], S, saut, ProjVector)[0])
Depliage = Fonction.depliage(Liste_Feuille[0], S, saut, ProjVector)
#UnfoldedPnt1 = Depliage[0]#Coordonées de la déformée des points de projection
rotation_matrix = Depliage[1]
roulement_matrix = Depliage[2]

#Dépliage du cadre de l'aile
CadreAileUnfolded, yf, zf = Fonction.depliage_cadre_objet(CadreAile, S.SurfaceType, S.Gradient(), rotation_matrix, roulement_matrix, ProjVector, widthPrintable, heightPrintable)

##------------------------------FIN DEPLIAGE---------------------------------##

##--------------------------------AFFICHAGE----------------------------------##

Plot.PlotReference(Nbimage, debut, saut, Liste_Feuille)

Plot.Plot3D(Nbimage, debut, saut, Liste_Feuille, Liste_Projection, CadreAile, d)

Plot.PlotUnfolded(Nbimage, debut, saut, Liste_Feuille, Liste_depliage, CadreAileUnfolded, yf, zf)

##-----------------------------FIN AFFICHAGE---------------------------------##

##--------------------------DECOUPAGE IMPRESSION-----------------------------##
#Decoupe la derniere figure en morceau de taille (widthPrintable,heightPrintable)
#pour pouvoir l'imprimer facilement. Sauvegarde dans un folder au format .pdf

Fonction.Print(PrintPath, yf, zf, widthPrintable, heightPrintable,Nbimage, debut, saut, Liste_Feuille, Liste_depliage, CadreAileUnfolded)

##------------------------FIN DECOUPAGE IMPRESSION---------------------------##