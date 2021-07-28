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
import Fonction
import os
import glob

plt.close('all')

##-------------------------------CONSTANTES----------------------------------##

saut = 2 #Taille du saut de point dans la liste contours

debut = 2 #Debut des boucles for pour les projections
debut2 = 2
debut3 = 2
debut4 = 2
debut5 = 2

height = 27e-2#27e-2 #29.7e-2#hauteur en m de l'image de reference(m)
width = 21e-2#21e-2 #21e-2#largeur en m de l'image de reference(m)
WingWidth = 8.4e-2 #largeur zone analyse de l'aile (m)
WingHeight = 1 #hauteur zone analyse de l'aile (m)


widthPrintable = 27.9e-2
heightPrintable = 21.6e-2

image1 = cv2.imread("/Users/yvan/Desktop/ETS_montreal/Cours/E21/MTR892/Banque_Speckle/4mm/Speckle_1.png")
image2 = cv2.imread("/Users/yvan/Desktop/ETS_montreal/Cours/E21/MTR892/Banque_Speckle/4mm/Speckle_2.png")

#cv2.imshow('Reference', image)

#Angles
gamma = 80.0 #angle entre capteur et plan aile (deg)
theta = 90.0-gamma #Angle entre normale capteur et plan aile (deg)
alpha = 22.0#Angle de champ de vue 
beta = 180.0-135.0-alpha/2.0 #Angle aigu entre aile et axe optique

l = np.sqrt(1**2 + 1**2)
A = np.array([l*np.cos(alpha/2*np.pi/180), 0, -l*np.sin(alpha/2*np.pi/180)])
B = np.array([A[0] + 1*np.cos(beta*np.pi/180), 0, A[2] + 1*np.sin(beta*np.pi/180)])
C1 = np.array([[(B[0]+A[0])/2, (WingWidth)/2, (B[2]+A[2])/2],
               [(B[0]+A[0])/2, (-WingWidth)/2, (B[2]+A[2])/2]])
CadreAile = np.vstack((A, B, C1))#Points qui definissent les limites spatiales de l'aile

CentreH1 = 0 #CadreAile[2,1]-width/2 #Position horizontale du centre du speckle de référence 1
CentreV1 = CadreAile[0,2]+height/2 #Position verticale du centre du speckle de référence 1
CentreH2 = 0 #CentreH1 #Position horizontale du centre du speckle de référence 2
CentreV2 = CentreV1 + height#Position verticale du centre du speckle de référence 2
CentreH3 = CentreH1 #Position horizontale du centre du speckle de référence 3
CentreV3 = height + CentreV1 #Position verticale du centre du speckle de référence 3
CentreH4 = CentreH2 #Position horizontale du centre du speckle de référence 4
CentreV4 = CentreV3 #Position verticale du centre du speckle de référence 4
CentreH5 = CentreH2-width #Position horizontale du centre du speckle de référence 4
CentreV5 = CentreV2 #Position verticale du centre du speckle de référence 4

#Parametre Position aile
a = (B-A)[0]#-np.sin(theta*np.pi/180)# -0.02#
b = (B-A)[1]#-np.sin(theta*np.pi/180)#np.linalg.solve(D,E)[1]
c = (B-A)[2]#np.cos(theta*np.pi/180)#1#np.linalg.solve(D,E)[2]#
dprim = 1#a*A[0]**1+b*A[1]**1+c*A[2]
Radius = 4.2e-2
Pos = np.array([A[0], A[1], A[2]])


'''
A = np.array([Pos[0]-Radius, 0, 0.2])#np.array([l*np.cos((alpha/2)*np.pi/180), 0, l*np.sin((-alpha/2)*np.pi/180)])
B = np.array([Pos[0]-Radius, 0, -0.2])#np.array([A[0] + (5.5-2.5)*np.cos(beta*np.pi/180), 0, A[2] + (5.5-2.5)*np.sin(beta*np.pi/180)])
C1 = np.array([[Pos[0], -Radius, 0],#(WingHeight)/2],
               [Pos[0], Radius, 0]])#(-WingHeight)/2]])
'''


#Parametres Position Mouchetis Reference
xa = 1
ya = 0
za = 0
d = A[0]#7.52928#


#Creation des plans dans l'espace centré sur le centre optique
# yg1, zg1 = np.meshgrid(np.arange(-width/2, width/2, width/50),
#                        np.arange(-height/2, height/2, height/50))
# xg1 = (d-ya*yg1-za*zg1)/xa
#xgp, ygp = np.meshgrid(np.arange(A[0], B[0], (B[0]-A[0])/2), np.arange(-WingWidth/2, WingWidth/2, WingWidth/10))
#zplane = (dprim-b*ygp**1-a*xgp**1)/c #A modifier avec l'equation de surface

S = Surface(a, b, c, Pos[0], Pos[1], Pos[2], dprim, Radius, 'Cylindre')
x, y, z = Symbol('x'), Symbol('y'), Symbol('z')
F = S.Equation(x,y,z) #a*x**1+ b*(y)**1 + c*z - dprim #Fonction de surface 3D F(x,y,z) = 0 où (x,y,z) appartient à la surface 3D
delta1 = Symbol('delta1', positive=True)


##------------------------------FIN CONSTANTES-------------------------------##

##---------------------------------FEUILLES----------------------------------##
 
Feuille1 = Feuille(CentreH1, CentreV1, image1, height, width, debut, saut, d)
Feuille2 = Feuille(CentreH2, CentreV2, image2, height, width, debut2, saut, d)

Liste_Feuille=[Feuille1, Feuille2]
##-----------------------------FIN FEUILLES----------------------------------##

##--------------------------------PROJECTION---------------------------------##

Pntprojection1 = Feuille1.projection(saut, F, x, y, z, delta1)[0]#Coordonée des points de projectoin de la feuille1
Pntprojection2 = Feuille2.projection(saut, F, x, y, z, delta1)[0]

##------------------------------FIN PROJECTION-------------------------------##

##--------------------------------DEPLIAGE-----------------------------------##
#On récupère le vecteur normal à la surface en un point donné puis on effectue
#une rotation de ce vecteur pour avoir un vecteur horizontal. Cette matrice de
#rotation est ensuite appliqué sur la position du point donné pour obtenir un
#point déplié sur un plan horizontal
GradF = sym.Matrix([sym.diff(F,x), sym.diff(F,y), sym.diff(F,z)]) #Gradient (vecteur normal) de la surface obtenu à partir de l'equation de la surface
ProjVector = np.array([-1, 0, 0])#Direction de dépliage de la surface 3D
Depliage = Fonction.depliage(Feuille1, S, saut, x, y, z, GradF, ProjVector)
UnfoldedPnt1 = Depliage[0]#Coordonées de la déformée des points de projection
UnfoldedPnt2 = Fonction.depliage(Feuille2, S, saut, x, y, z, GradF, ProjVector)[0]
rotation_matrix = Depliage[1]
roulement_matrix = Depliage[2]

#Dépliage du cadre de l'aile
CadreAileUnfolded = np.zeros((4,3))
if S.SurfaceType=='Plan':
    for i in range(4):
        NormalVector = np.array(GradF.subs([(x, CadreAile[i,0]), (y, CadreAile[i, 1]), (z, CadreAile[i, 2])])).astype(np.float64)/np.linalg.norm(np.array(GradF.subs([(x, CadreAile[i,0]), (y, CadreAile[i,1]), (z, CadreAile[i,2])])).astype(np.float64))
        v = np.cross(np.squeeze(NormalVector), ProjVector)
        c = np.dot(np.squeeze(NormalVector), ProjVector)
        kmat = np.array([[0, -v[2], v[1]], 
                         [v[2], 0, -v[0]], 
                         [-v[1], v[0], 0]])
        rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * (1/(1+c))
        CadreAileUnfolded[i,:] = np.dot(rotation_matrix, CadreAile[i,:])
    yf, zf  = np.meshgrid(np.arange(min(CadreAileUnfolded[:,1]), max(CadreAileUnfolded[:,1]) + widthPrintable, widthPrintable), np.arange(min(CadreAileUnfolded[:,2]), max(CadreAileUnfolded[:,2]) + heightPrintable, heightPrintable))

elif S.SurfaceType=='Cylindre':
    for i in range(4):
        CadreAileUnfolded[i,:] = np.dot(rotation_matrix, CadreAile[i,:])
        CadreAileUnfolded[i,:] = np.dot(roulement_matrix, CadreAileUnfolded[i,:])
    yf, zf  = np.meshgrid(np.arange(min(CadreAileUnfolded[:,1]), max(CadreAileUnfolded[:,1]) + widthPrintable, widthPrintable), np.arange(min(CadreAileUnfolded[:,2]), max(CadreAileUnfolded[:,2]) + heightPrintable, heightPrintable))

    
NbFeuille = (yf.shape[1]-1) * (zf.shape[0]-1)

##------------------------------FIN DEPLIAGE---------------------------------##

##--------------------------------AFFICHAGE----------------------------------##

Feuille1.Affichage_reference(saut, 1, 'k')
Feuille2.Affichage_reference(saut, 2, 'b')

fig2 = plt.figure(6)
ax = fig2.add_subplot(111, projection='3d')
ax.scatter(0, 0, 0, color='b')
for i in range (debut, len(Feuille1.contours), saut):
    ax.plot(Feuille1.contours3D[i][:, 0], Feuille1.contours3D[i][:, 1], Feuille1.contours3D[i][:, 2], color='k', marker=None)
    ax.plot(Pntprojection1[i][:, 0], Pntprojection1[i][:, 1], Pntprojection1[i][:, 2], color='k', marker=None)
for i in range(debut2, len(Feuille2.contours), saut):
    ax.plot(Feuille2.contours3D[i][:, 0], Feuille2.contours3D[i][:, 1], Feuille2.contours3D[i][:, 2], color='b', marker=None)
    ax.plot(Pntprojection2[i][:, 0], Pntprojection2[i][:, 1], Pntprojection2[i][:, 2], color='b', marker=None)
    
# for i in range(debut3, len(Feuille3.contours), saut):
#     ax.plot(Feuille3.contours3D[i][:, 0], Feuille3.contours3D[i][:, 1], Feuille3.contours3D[i][:, 2], color='y', marker=None)
#     ax.plot(Pntprojection3[i][:, 0], Pntprojection3[i][:, 1], Pntprojection3[i][:, 2], color='y', marker=None)

# for i in range(debut4, len(Feuille4.contours), saut):
#     ax.plot(Feuille4.contours3D[i][:, 0], Feuille4.contours3D[i][:, 1], Feuille4.contours3D[i][:, 2], color='r', marker=None)
#     ax.plot(Pntprojection4[i][:, 0], Pntprojection4[i][:, 1], Pntprojection4[i][:, 2], color='r', marker=None)
    
# for i in range(debut5, len(Feuille5.contours), saut):
#     ax.plot(Feuille5.contours3D[i][:, 0], Feuille5.contours3D[i][:, 1], Feuille5.contours3D[i][:, 2], color='g', marker=None)
#     ax.plot(Pntprojection5[i][:, 0], Pntprojection5[i][:, 1], Pntprojection5[i][:, 2], color='g', marker=None)
    #ax.plot(UnfoldedPnt[i][:, 0], UnfoldedPnt[i][:, 1], -UnfoldedPnt[i][:, 2], color='r', marker=None)
#ax.plot_surface(xg1, yg1, zg1, color='b', alpha=0.2)
#ax.plot_surface(xgp, ygp, zplane, color='c', alpha=0.2)
ax.scatter(CadreAile[:,0], CadreAile[:,1], CadreAile[:,2], color='c')
ax.scatter([d]*4, Feuille1.Cadre[:,0], Feuille1.Cadre[:,1], color='k', marker='+')
ax.scatter([d]*4, Feuille2.Cadre[:,0], Feuille2.Cadre[:,1], color='b', marker='+')
# ax.scatter([d]*4, Feuille3.Cadre[:,0], Feuille3.Cadre[:,1], color='y', marker='+')
# ax.scatter([d]*4, Feuille4.Cadre[:,0], Feuille4.Cadre[:,1], color='r', marker='+')
# ax.scatter([d]*4, Feuille5.Cadre[:,0], Feuille5.Cadre[:,1], color='g', marker='+')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.title('Image référence et projetée 3D (m)')
Fonction.set_aspect_equal_3d(ax)

fig3=plt.figure(7)
for i in range(debut, len(Feuille1.contours), saut):
      plt.plot(UnfoldedPnt1[i][:, 1], UnfoldedPnt1[i][:, 2], color='black')
      plt.fill(UnfoldedPnt1[i][:, 1], UnfoldedPnt1[i][:, 2], color='black')
for i in range(debut2, len(Feuille2.contours), saut):
    plt.plot(UnfoldedPnt2[i][:, 1], UnfoldedPnt2[i][:, 2], color='blue')
    plt.fill(UnfoldedPnt2[i][:, 1], UnfoldedPnt2[i][:, 2], color='blue')
# for i in range(debut3, len(Feuille3.contours), saut):
#       plt.plot(UnfoldedPnt3[i][:, 1], UnfoldedPnt3[i][:, 2], color='y')
#       plt.fill(UnfoldedPnt3[i][:, 1], UnfoldedPnt3[i][:, 2], color='y')     
# for i in range(debut4, len(Feuille4.contours), saut):
#       plt.plot(UnfoldedPnt4[i][:, 1], UnfoldedPnt4[i][:, 2], color='r')
#       plt.fill(UnfoldedPnt4[i][:, 1], UnfoldedPnt4[i][:, 2], color='r')  
# for i in range(debut5, len(Feuille5.contours), saut):
#       plt.plot(UnfoldedPnt5[i][:, 1], UnfoldedPnt5[i][:, 2], color='g')
#       plt.fill(UnfoldedPnt5[i][:, 1], UnfoldedPnt5[i][:, 2], color='g')  
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

##-----------------------------FIN AFFICHAGE---------------------------------##


##--------------------------DECOUPAGE IMPRESSION-----------------------------##
#Decoupe la derniere figure en morceau de taille (widthPrintable,heightPrintable)
#pour pouvoir l'imprimer facilement. Sauvegarde dans un folder au format .pdf
files = glob.glob('/Users/yvan/Desktop/ETS_montreal/Cours/E21/MTR892/AnamorphosePlane/ImagePrintableTestCirculaire/*.pdf')
for f in files:
    os.remove(f)

for i in range (yf.shape[0]-1):
    for j in range (yf.shape[1]-1):
        fig = plt.figure((i+1)*(j+1)+7)
        fig.set_size_inches(widthPrintable/0.0254, heightPrintable/0.0254)
        ax = fig.add_subplot(111, aspect='equal')
        axe = plt.gca()
        x_axis = axe.axes.get_xaxis()
        x_axis.set_visible(False)
        y_axis = axe.axes.get_yaxis()
        y_axis.set_visible(False)
        for l in range(debut, len(Feuille1.contours), saut):
            plt.plot(UnfoldedPnt1[l][:, 1], UnfoldedPnt1[l][:, 2], color='k')
            plt.fill(UnfoldedPnt1[l][:, 1], UnfoldedPnt1[l][:, 2], color='k')
        for l in range(debut2, len(Feuille2.contours), saut):
            plt.plot(UnfoldedPnt2[l][:, 1], UnfoldedPnt2[l][:, 2], color='k')
            plt.fill(UnfoldedPnt2[l][:, 1], UnfoldedPnt2[l][:, 2], color='k')
        # for l in range(debut3, len(Feuille3.contours), saut):
        #     plt.plot(UnfoldedPnt3[l][:, 1], UnfoldedPnt3[l][:, 2], color='k')
        #     plt.fill(UnfoldedPnt3[l][:, 1], UnfoldedPnt3[l][:, 2], color='k')
        # for l in range(debut4, len(Feuille4.contours), saut):
        #     plt.plot(UnfoldedPnt4[l][:, 1], UnfoldedPnt4[l][:, 2], color='k')
        #     plt.fill(UnfoldedPnt4[l][:, 1], UnfoldedPnt4[l][:, 2], color='k')
        # for l in range(debut5, len(Feuille5.contours), saut):
        #     plt.plot(UnfoldedPnt5[l][:, 1], UnfoldedPnt5[l][:, 2], color='k')
        #     plt.fill(UnfoldedPnt5[l][:, 1], UnfoldedPnt5[l][:, 2], color='k')
        plt.scatter(CadreAileUnfolded[:,1], CadreAileUnfolded[:,2], color='c', marker='+')
        plt.scatter(yf, zf, marker='+', color='m')
        plt.axis('equal')
        plt.xlim(yf[0][j], yf[0][j+1])
        plt.ylim(zf[i][0], zf[i+1][0])
        plt.box(False)
        plt.close(fig)
        fig.tight_layout()#Supprime les marges
        fig.savefig('/Users/yvan/Desktop/ETS_montreal/Cours/E21/MTR892/AnamorphosePlane/ImagePrintableTestCirculaire/Image'+str(i+1)+','+str(j+1)+'.pdf')

plt.show()
##------------------------FIN DECOUPAGE IMPRESSION---------------------------##