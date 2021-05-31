#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 25 11:51:13 2021

@author: yvan
"""

import numpy as np
import matplotlib.pyplot as plt
from IPython import get_ipython
from cv2 import cv2

plt.close('all')
get_ipython().magic('reset -sf')

img = cv2.imread("/Users/yvan/Desktop/ETS_montreal/Cours/E21/MTR892/Speckle_4-65-90-210-270.png")
height = img.shape[0]
width = img.shape[1]

#Angles
gamma = 80 #angle entre capteur et plan aile (deg)
theta = 90-gamma #Angle entre normale capteur et plan aile (deg)
alpha = 10 #Angle de champ de vue 
beta = 180-165-alpha/2 #Angle aigu entre aile et axe optique

#Calcul Point debut champ de vue
l = np.sqrt(0.9**2 + 2.5**2 + 0.9*2.5*np.cos(105*np.pi/180))
A = [l*np.cos((alpha/2)*np.pi/180), 0, l*np.sin((-alpha/2)*np.pi/180)]
B = [A[0] + (5.5-2.5)*np.cos(beta*np.pi/180), 0, A[2] + (5.5-2.5)*np.sin(beta*np.pi/180)]

#Plane aile - normal vector
a = -np.sin(theta*np.pi/180)
b = 0
c = np.cos(theta*np.pi/180)
dprim = a*A[0]+b*A[1]+c*A[2]

#Plane 1 - normal vector
xa = 1
ya = 0
za = 0
d = A[0]

#Taille de l'image de référence
new_height = 2*d*np.tan((alpha/2)*np.pi/180)
new_width = 2*d*np.tan((alpha/2)*np.pi/180)

#Creation des plans dans l'espace centré sur le centre optique
yg1, zg1 = np.meshgrid(np.arange(-new_width/2, new_width/2, new_width/20), np.arange(-new_height/2, new_height/2, new_height/20))
xg1 = (d-ya*yg1-za*zg1)/xa
xgp, ygp = np.meshgrid(np.arange(A[0], B[0], (B[0]-A[0])/20), np.arange(-new_width/2, new_width/2, new_width/20))
zplane = (dprim-b*ygp-a*xgp)/c

#Cadre de l'image de réference
POI = np.array([[d, -new_width/2, -new_height/2],
                [d, -new_width/2, new_height/2],
                [d, new_width/2, new_height/2],
                [d, new_width/2, -new_height/2]], dtype='float32')

delta = (a*POI[:, 0]+b*POI[:, 1]+c*POI[:, 2])/dprim
Pntprojection = POI/delta[:, None]# Coordonnées des points projetés

PntPrjtOnPng = np.zeros((4, 2),dtype='float32')
POIOnPng = np.zeros((4, 2),dtype='float32')
POIOnPng[:, 0] = (height/new_height)*POI[:, 1]+height/2
POIOnPng[:, 1] = (width/new_width)*POI[:, 2]+width/2
PntPrjtOnPng[:, 0] = (height/new_height)*Pntprojection[:, 1]+height/2
PntPrjtOnPng[:, 1] = (width/new_width)*Pntprojection[:, 2]+width/2

#Dimension réelle de l'image
hauteur = np.sqrt((Pntprojection[0,0]-Pntprojection[1,0])**2+(Pntprojection[0,1]
        -Pntprojection[1,1])**2+(Pntprojection[0,2]-Pntprojection[1,2])**2)
largeur = np.sqrt((Pntprojection[1,0]-Pntprojection[2,0])**2+(Pntprojection[1,1]
        -Pntprojection[2,1])**2+(Pntprojection[1,2]-Pntprojection[2,2])**2)

#Matrice de passage reference-deformée
tform = cv2.getPerspectiveTransform(POIOnPng, PntPrjtOnPng)

#Déformation image de reference
tf_img_warp = cv2.warpPerspective(img, tform, (int(width), int(height)))
cv2.imwrite('/Users/yvan/Desktop/ETS_montreal/Cours/E21/MTR892/Deformeecv2.png', tf_img_warp)#Enregistrement de l'image déformée

##------------------------------AFFICHAGE-----------------------------------##
fig = plt.figure(1)
ax = fig.add_subplot(111, projection='3d')
ax.scatter(0, 0, 0, color='b')
ax.scatter(A[0], A[1], A[2], color='r')
ax.scatter(B[0], B[1], B[2], color='r')
ax.scatter(POI[:, 0], POI[:, 1], POI[:, 2], color='k')
ax.scatter(Pntprojection[:, 0], Pntprojection[:, 1], Pntprojection[:, 2], color='k')
ax.plot_surface(xg1, yg1, zg1, rstride=10, cstride=10, color='b', alpha=0.2)
ax.plot_surface(xgp, ygp, zplane, rstride=10, cstride=10, color='r', alpha=0.2)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
#ax.axes.set_xlim3d(left=0, right=8)
ax.axes.set_ylim3d(bottom=-1, top=1)
ax.axes.set_zlim3d(bottom=-0.6, top=1)
plt.show()

plt.figure(2)
plt.imshow(img, origin='lower')
plt.plot(POIOnPng[:, 1], POIOnPng[:, 0], marker='+', color='red')
plt.xlabel('Columns')
plt.ylabel('Rows')
plt.title('Réference')
plt.show()

plt.figure(3)
plt.imshow(tf_img_warp, origin='lower')
plt.xlabel('Columns')
plt.ylabel('Rows')
plt.title('Déformée')
plt.show()
