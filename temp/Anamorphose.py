import numpy as np
import matplotlib.pyplot as plt
import sympy as sym
from sympy import Symbol
import cv2 as cv2
from Surface import Surface
from Sheets import Sheets
from Plot import Plot
from Speckle import Speckle
from data import Deck
import Fonction
import os
import glob
import sys


deck = Deck('/Users/yvan/Desktop/ETS_montreal/Cours/E21/MTR892 - Projet technique/AnamorphosePlane/temp/deck.yaml')
#print(deck.doc)

plt.close('all')


##-------------------------------CONSTANTES----------------------------------##
'''
step = 500 #Taille du saut de point dans la liste contours

begining = 3 #Debut des boucles for pour les projections

height = 27e-2#27e-2 #29.7e-2#hauteur en m de l'image de reference(m)
width = 21e-2#21e-2 #21e-2#largeur en m de l'image de reference(m)
#WingWidth = 60e-2 #largeur zone analyse de l'aile (m)
#WingHeight = 3 #hauteur zone analyse de l'aile (m)

heightPrintable = 27.9e-2
widthPrintable = 21.6e-2
PrintPath = '/Users/yvan/Desktop/ETS_montreal/Cours/E21/MTR892 - Projet technique/AnamorphosePlane/ImagePrintable'

image1 = cv2.imread("/Users/yvan/Desktop/ETS_montreal/Cours/E21/MTR892 - Projet technique/Banque_Speckle/2mm/Speckle_1.png")
image2 = cv2.imread("/Users/yvan/Desktop/ETS_montreal/Cours/E21/MTR892 - Projet technique/Banque_Speckle/2mm/Speckle_2.png")
List_image = [image1, image2]
Nbimage=len(List_image)

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
d = 2
Sheets_pos = np.array([[CentreH1, CentreV1, d],
                        [CentreH2,CentreV2, d]])'''
#Parametre Position aile
a = -1
b = 0
c = 1
dprim = 1
Radius = 0.4
Pos = np.array([3, 0, 0])

A = np.array([3+27e-2*np.cos(45*np.pi/180), 0, 0.2])#np.array([l*np.cos((alpha/2)*np.pi/180), 0, l*np.sin((-alpha/2)*np.pi/180)])
B = np.array([3-27e-2*np.cos(45*np.pi/180), 0, -0.2])#np.array([A[0] + (5.5-2.5)*np.cos(beta*np.pi/180), 0, A[2] + (5.5-2.5)*np.sin(beta*np.pi/180)])
C1 = np.array([[Pos[0], -20e-2, 0],#(WingHeight)/2],
               [Pos[0], 20e-2, 0]])#(-WingHeight)/2]])
WingFrame = np.vstack((A, B, C1))#Points qui definissent les limites spatiales de l'aile


##------------------------------FIN CONSTANTES-------------------------------##

##---------------------------------SURFACE----------------------------------##

S = Surface(deck.a, deck.b, deck.c, deck.Position, deck.radius, deck.SurfaceType)
##-------------------------------FIN SURFACE--------------------------------##

##---------------------------------FEUILLES----------------------------------##

speckle = Speckle(deck.NbImage, deck.PositionCentre, deck.Images(), deck.height, deck.width, deck.begining, deck.step)

#List_Sheets=[]
#for i in range(Nbimage):
#   List_Sheets.append(Sheets(Sheets_pos[i,0], Sheets_pos[i,1], Sheets_pos[i,2], List_image[i], height, width, begining, step))

##-----------------------------FIN FEUILLES----------------------------------##

##--------------------------------PROJECTION---------------------------------##

Liste_Projection = speckle.ProjectionSpeckle(S)

#Liste_Projection=[]
#for i in range(Nbimage):
    #Liste_Projection.append(List_Sheets[i].projection(S)[0])#Coordonnées des points de projection de la feuille1
#speckle = Speckle(Nbimage, Sheets_pos, List_image, height, width, begining, step)
#Liste_Projection = speckle.ProjectionSpeckle(S)
##------------------------------FIN PROJECTION-------------------------------##

##--------------------------------DEPLIAGE-----------------------------------##
#On récupère le vecteur normal à la surface en un point donné puis on effectue
#une rotation de ce vecteur pour avoir un vecteur horizontal. Cette matrice de
#rotation est ensuite appliqué sur la position du point donné pour obtenir un
#point déplié sur un plan horizontal

List_Unfolded=[]
for i in range(deck.NbImage):
    List_Unfolded.append(Fonction.Unfold(speckle.List_Sheets[i], S)[0])
Depliage = Fonction.Unfold(speckle.List_Sheets[0], S)
#UnfoldedPnt1 = Depliage[0]#Coordonées de la déformée des points de projection
rotation_matrix = Depliage[1]
roulement_matrix = Depliage[2]

#Dépliage du cadre de l'aile
WingFrameUnfolded, yf, zf = Fonction.Unfold_object_frame(WingFrame, S.SurfaceType, S.Gradient(), rotation_matrix, roulement_matrix, deck.widthPrintable, deck.heightPrintable)

##------------------------------FIN DEPLIAGE---------------------------------##

##--------------------------------AFFICHAGE----------------------------------##

Plot.PlotReference(deck.NbImage, speckle.List_Sheets)

Plot.Plot3D(deck.NbImage, speckle.List_Sheets, Liste_Projection, WingFrame)

Plot.PlotUnfolded(deck.NbImage, speckle.List_Sheets, List_Unfolded, WingFrameUnfolded, yf, zf)

##-----------------------------FIN AFFICHAGE---------------------------------##

##--------------------------DECOUPAGE IMPRESSION-----------------------------##
#Decoupe la derniere figure en morceau de taille (widthPrintable,heightPrintable)
#pour pouvoir l'imprimer facilement. Sauvegarde dans un folder au format .pdf

Fonction.Print(deck.PrintPath, yf, zf, deck.widthPrintable, deck.heightPrintable, deck.NbImage, speckle.List_Sheets, List_Unfolded, WingFrameUnfolded)
##------------------------FIN DECOUPAGE IMPRESSION---------------------------##