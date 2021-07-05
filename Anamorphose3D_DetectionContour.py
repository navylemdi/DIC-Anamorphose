#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 29 15:06:06 2021

@author: yvan
"""
import numpy as np
import matplotlib.pyplot as plt
import sympy as sym
from sympy import Symbol
from sympy.solvers import solve
from IPython import get_ipython
import cv2 as cv2
import time
import sys

get_ipython().magic('reset -sf')
plt.close('all')

##--------------------------------FONCTIONS----------------------------------##

def Pix2Meter(Pospix, image, Lim_inf_H, Lim_max_H, Lim_inf_V, Lim_max_V, CentreH, CentreV):
    Posmet = np.zeros((len(Pospix),2), np.float32)
    Posmet[:, 1] = ((Lim_max_V-Lim_inf_V)*Pospix[:,1])/image.shape[0] + Lim_inf_V + CentreV
    Posmet[:, 0] = ((Lim_max_H-Lim_inf_H)*Pospix[:,0])/image.shape[1] + Lim_inf_H + CentreH
    return Posmet

def Meter2Pix(Posmet, image, Lim_inf_H, Lim_max_H, Lim_inf_V, Lim_max_V):
    Pospix = np.zeros((len(Posmet),2), np.float32)
    Pospix[:, 0] = image.shape[0]*(Posmet[:,0]-Lim_inf_V)/(Lim_max_V-Lim_inf_V)
    Pospix[:, 1] = image.shape[1]*(Posmet[:,1]-Lim_inf_H)/(Lim_max_H-Lim_inf_H)
    return Pospix

##------------------------------FIN FONCTIONS--------------------------------##

##-------------------------------CONSTANTES----------------------------------##

saut = 50 #Taille du saut de point dans la liste contours
debut = 2 #Debut des boucles for pour les projections
height = 29.7e-2 #29.7e-2#hauteur en m de l'image de reference(m)
width = 21e-2 #21e-2#largeur en m de l'image de reference(m)
WingWidth = 60e-2 #largeur zone analyse de l'aile (m)
WingHeight = 3 #hauteur zone analyse de l'aile (m)
CentreH = 0.1 #Position horizontale du centre du speckle de référence
CentreV = 0 #Position verticale du centre du speckle de référence

image = cv2.imread("/Users/yvan/Desktop/ETS_montreal/Cours/E21/MTR892/Speckle_4-65-90-210-270_100pi_cm.png")
#cv2.imshow('Reference', image)

#Angles
alpha = 10.0#Angle de champ de vue
beta = 10 #Angle aigu entre aile et axe optique
gamma = 180-90-beta #angle entre capteur et plan aile (deg)
theta = 90.0-gamma #Angle entre normale capteur et plan aile (deg)

l = np.sqrt(0.9**2 + 2.5**2 + 0.9*2.5*np.cos(105*np.pi/180))
A = np.array([l*np.cos((alpha/2)*np.pi/180), 0, l*np.sin((-alpha/2)*np.pi/180)])
B = np.array([A[0] + (5.5-2.5)*np.cos(beta*np.pi/180), 0, A[2] + (5.5-2.5)*np.sin(beta*np.pi/180)])
C1 = np.array([[(B[0]+A[0])/2, (WingWidth)/2, (B[2]+A[2])/2],
               [(B[0]+A[0])/2, (-WingWidth)/2, (B[2]+A[2])/2]])
CadreAile = np.vstack((A, B, C1))#Points qui definissent les limites spatiales de l'aile

# D = np.array([[CadreAile[0, 0]**2, CadreAile[0, 1]**2, CadreAile[0, 2]],
#               [CadreAile[1, 0]**2, CadreAile[1, 1]**2, CadreAile[1, 2]],
#               [CadreAile[2, 0]**2, CadreAile[2, 1]**2, CadreAile[2, 2]]])
# E = np.array([-1,-1,-1])

#Plane aile - normal vector
a = -np.sin(theta*np.pi/180)# -0.02#
b = 0#-np.sin(theta*np.pi/180)#np.linalg.solve(D,E)[1]
c = np.cos(theta*np.pi/180)#1#np.linalg.solve(D,E)[2]#
dprim = a*A[0]**1+b*A[1]**1+c*A[2]

#Plane 1 - normal vector
xa = 1
ya = 0
za = 0
d = A[0]#7.52928#

# POI = np.array([[d, -width/2, -height/2],
#                 [d, -width/2, height/2],
#                 [d, width/2, height/2],
#                 [d, width/2, -height/2]], np.float32)
Pospix = np.array([[0, 0],
                   [0, image.shape[0]],
                   [image.shape[1], 0],
                   [image.shape[1], image.shape[0]]])

Cadre = Pix2Meter(Pospix, image, -width/2, width/2, height/2, -height/2, CentreH, CentreV)

#Creation des plans dans l'espace centré sur le centre optique
yg1, zg1 = np.meshgrid(np.arange(-width/2, width/2, width/50),
                       np.arange(-height/2, height/2, height/50))
xg1 = (d-ya*yg1-za*zg1)/xa
xgp, ygp = np.meshgrid(np.arange(A[0], B[0], (B[0]-A[0])/10),
                       np.arange(-WingWidth/2, WingWidth/2, WingWidth/10))
zplane = (dprim-b*ygp**1-a*xgp**1)/c #A modifier avec l'equation de surface

##------------------------------FIN CONSTANTES-------------------------------##

##--------------------------------CONTOURS-----------------------------------##

image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#cv2.imshow('Gray scale', image_gray)
#Conversion en NB
ret,thresh = cv2.threshold(image_gray, 127, 255, cv2.THRESH_BINARY)
#cv2.imshow('Threshold', thresh)

contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, 
                                     cv2.CHAIN_APPROX_TC89_L1)#Trouver contours

##--------------------------------FIN CONTOURS-------------------------------##

##--------------------------------PROJECTION---------------------------------##

x, y, z = Symbol('x'), Symbol('y'), Symbol('z')
F = a*x**1+ b*(y)**1 + c*z - dprim #Fonction de surface 3D F(x,y,z) = 0 où (x,y,z) appartient à la surface 3D
delta1 = Symbol('delta1', positive=True)

# Transformation coordonées contours repère 2D en repère 3D
contours3D = [None]*len(contours)
for i in range(debut, len(contours), saut):
    #Opencv met l'ordonnée en premiere position des tableau et l'abscisse en seconde
    #C'est pour ca que les indices semblent inversés (H<->V) mais c'est normal
    contours3D[i] = np.empty( [len(contours[i]), 3], dtype=np.float32)
    temp = Pix2Meter(contours[i][:, 0], image,  -height/2, height/2,
                      -width/2, width/2, CentreV, CentreH)
    contours3D[i][:, 0] = d
    contours3D[i][:, 1] = temp[:, 1]
    contours3D[i][:, 2] = temp[:, 0]

#Calcul projection sur plan incliné
Pntprojection = [None]*len(contours)
print('Début calcul projection')
start = time.time()
for i in range(debut, len(contours), saut):
    Pntprojection[i] = np.empty( [len(contours[i]), 3], dtype=np.float32)
    sys.stdout.write('\r' + str(round((i/(len(contours)-1))*100,2)) + '% ')#Affichage pourcentage de l'avancement
    for j in range(len(contours3D[i])):
        try:
            sol = solve( F.subs([(x, contours3D[i][j,0]/delta1), (y, contours3D[i][j,1]/delta1), (z, contours3D[i][j,2]/delta1)]), delta1)[0]#Résolution de l'equation f(x/delta, y/delta, z/delta) = 0 avec delta l'inconnue
            Pntprojection[i][j,:] = contours3D[i][j,:]/sol# Coordonnées dans l'espace des points projetés
        except IndexError: 
            print("\nIl n'existe pas de solution pour les cercles. Vérifier l'equation de la surface")
    sys.stdout.flush()
print('\nFin calcul projection')
end = time.time()
print('Temps ecoulé: ', time.strftime("%H:%M:%S", time.gmtime(end-start)))

##------------------------------FIN PROJECTION-------------------------------##

##--------------------------------DEPLIAGE---------------------------------##
#On récupère le vecteur normal à la surface en un point donné puis on effectue
#une rotation de ce vecteur pour avoir un vecteur horizontal. Cette matrice de
#rotation est ensuite appliqué sur la position du point donné pour obtenir un
#point déplié sur un plan horizontal
GradF = sym.Matrix([sym.diff(F,x), sym.diff(F,y), sym.diff(F,z)]) #Gradient (vecteur normal) de la surface obtenu à partir de l'equation de la surface
ProjVector = np.array([1, 0, 0])#Direction de dépliage de la surface 3D
UnfoldedPnt = [None]*len(contours)
print('Début dépliage')
start = time.time()
for i in range(debut, len(contours), saut):
    UnfoldedPnt[i] = np.empty( [len(contours[i]), 3], dtype=np.float32)
    sys.stdout.write('\r' + str(round((i/(len(contours)-1))*100,2)) + '% ')#Affichage pourcentage de l'avancement
    for j in range(len(contours3D[i])):
        NormalVector = np.array(GradF.subs([(x, Pntprojection[i][j, 0]), (y, Pntprojection[i][j, 1]), (z, Pntprojection[i][j, 2])])).astype(np.float64)/np.linalg.norm(np.array(GradF.subs([(x, Pntprojection[i][j, 0]), (y, Pntprojection[i][j, 1]), (z, Pntprojection[i][j, 2])])).astype(np.float64))
        v = np.cross(np.squeeze(NormalVector), ProjVector)
        c = np.dot(np.squeeze(NormalVector), ProjVector)
        kmat = np.array([[0, -v[2], v[1]], 
                          [v[2], 0, -v[0]], 
                          [-v[1], v[0], 0]])
        rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * (1/(1+c))
        UnfoldedPnt[i][j, :] = np.dot(rotation_matrix, Pntprojection[i][j, :])
    sys.stdout.flush()
print('\nFin dépliage')
end = time.time()
print('Temps ecoulé: ', time.strftime("%H:%M:%S", time.gmtime(end-start)))

##------------------------------FIN DEPLIAGE---------------------------------##

##--------------------------------AFFICHAGE----------------------------------##

fig1=plt.figure(1)
ax = fig1.add_subplot(111, aspect='equal')
for i in range(debut, len(contours), saut):
    plt.plot(contours[i][:, 0][:, 0], contours[i][:, 0][:, 1], marker=None, color='black')
    ax.fill(contours[i][:, 0][:, 0], contours[i][:, 0][:, 1],'k',zorder=10)
plt.scatter(Pospix[:,0],Pospix[:,1], marker ='+')
plt.title('Image référence (pix)')
plt.show()

fig2 = plt.figure(2)
ax = fig2.add_subplot(111, projection='3d')
ax.scatter(0, 0, 0, color='b')
for i in range (debut, len(contours), saut):
    ax.plot(contours3D[i][:, 0], contours3D[i][:, 1], contours3D[i][:, 2], color='k', marker=None)
    ax.plot(Pntprojection[i][:, 0], Pntprojection[i][:, 1], Pntprojection[i][:, 2], color='k', marker=None)
    #ax.plot(UnfoldedPnt[i][:, 0], UnfoldedPnt[i][:, 1], -UnfoldedPnt[i][:, 2], color='r', marker=None)
ax.plot_surface(xg1, yg1, zg1, color='b', alpha=0.5)
ax.plot_surface(xgp, ygp, zplane, color='r', alpha=0.5)
ax.scatter(CadreAile[:,0], CadreAile[:,1], CadreAile[:,2], color='r')
ax.scatter([d]*4, Cadre[:,0], Cadre[:,1], color='r')
#ax.scatter(POI[:,0], POI[:,1], POI[:,2])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.title('Image référence et projetée 3D (m)')
plt.show()

fig3=plt.figure(3)
for i in range(debut, len(contours), saut):
      plt.plot(UnfoldedPnt[i][:, 1], -UnfoldedPnt[i][:, 2], color='black')
      plt.fill(UnfoldedPnt[i][:, 1], -UnfoldedPnt[i][:, 2], color='black')
plt.title('Dépliée')
#plt.axis('equal')
plt.grid()
plt.show()

# fig4=plt.figure(4)
# for i in range(debut, len(contours), saut):
#       plt.plot(Pntprojection[i][:, 1], Pntprojection[i][:, 0], color='black')
#       plt.fill(Pntprojection[i][:, 1], Pntprojection[i][:, 0], color='black')
# plt.title('Dépliée')
# #plt.axis('equal')
# plt.grid()
# plt.show()
##----------------------------FIN AFFICHAGE----------------------------------##

