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

img = cv2.imread("/Users/yvan/Desktop/ETS_montreal/Cours/E21/MTR892/Speckle_4-65-90-210-270_100pi_cm.png")
heightpi = img.shape[0]
widthpi = img.shape[1]
height=29.7e-2#hauteur en m de l'image de reference
width=21e-2#largeur en m de l'image de reference
#Angles
gamma = 80.0 #angle entre capteur et plan aile (deg)
theta = 90.0-gamma #Angle entre normale capteur et plan aile (deg)
alpha = 10.0#Angle de champ de vue 
beta = 180.0-165.0-alpha/2.0 #Angle aigu entre aile et axe optique

#Calcul Point debut champ de vue
l = np.sqrt(0.9**2 + 2.5**2 + 0.9*2.5*np.cos(105*np.pi/180))
A = np.array([l*np.cos((alpha/2)*np.pi/180), 0, l*np.sin((-alpha/2)*np.pi/180)])
B = np.array([A[0] + (5.5-2.5)*np.cos(beta*np.pi/180), 0, A[2] + (5.5-2.5)*np.sin(beta*np.pi/180)])
C1 = np.array([[(B[0]+A[0])/2, (60e-2)/2, (B[2]+A[2])/2],
 [(B[0]+A[0])/2, (-60e-2)/2, (B[2]+A[2])/2]])
CadreAile = np.vstack((A, B, C1))#Points qui definissent les limites spatiales de l'aile

#Plane aile - normal vector
a = -np.sin(theta*np.pi/180)
b = 0
c = np.cos(theta*np.pi/180)
dprim = a*A[0]+b*A[1]+c*A[2]

#Plane 1 - normal vector
xa = 1
ya = 0
za = 0
d = A[0]#7.52928#

#Taille de l'image de référence
new_height = 2*d*np.tan((alpha/2)*np.pi/180)
new_width = 2*d*np.tan((alpha/2)*np.pi/180)

#Creation des plans dans l'espace centré sur le centre optique
yg1, zg1 = np.meshgrid(np.arange(-new_width/2, new_width/2, new_width/20), np.arange(-new_height/2, new_height/2, new_height/20))
xg1 = (d-ya*yg1-za*zg1)/xa
xgp, ygp = np.meshgrid(np.arange(A[0], B[0], (B[0]-A[0])/20), np.arange(-(60e-2)/2, (60e-2)/2, (60e-2)/20))
zplane = (dprim-b*ygp-a*xgp)/c

#Cadre de l'image de réference
POI = np.array([[d, -new_width/2, -new_height/2],
                [d, -new_width/2, new_height/2],
                [d, new_width/2, new_height/2],
                [d, new_width/2, -new_height/2]], np.float32)

originB = np.array([d, 0, 0], np.float32)#milieu du plan de l'image de reference
delta = (a*POI[:, 0]+b*POI[:, 1]+c*POI[:, 2])/dprim
Pntprojection = POI/delta[:, None]# Coordonnées des points projetés

delta3=(a*originB[0]+b*originB[1]+c*originB[2])/dprim
originR=originB/delta3 #Origine du plan rouge (projection de l'origine bleu sur plan incliné)

PntPrjtOnPng = np.zeros((4, 2),np.float32)
POIOnPng = np.zeros((4, 2),np.float32)
POIOnPng[:, 0] = (heightpi/new_height)*POI[:, 1]+heightpi/2
POIOnPng[:, 1] = (widthpi/new_width)*POI[:, 2]+widthpi/2
PntPrjtOnPng[:, 0] = (heightpi/new_height)*Pntprojection[:, 1]+heightpi/2
PntPrjtOnPng[:, 1] = (widthpi/new_width)*Pntprojection[:, 2]+widthpi/2

passage_horizontal_incline=np.array([[np.cos(gamma*np.pi/180), 0, -np.sin(gamma*np.pi/180)],
                                     [0,                      1,                       0],
                                     [np.sin(gamma*np.pi/180), 0,  np.cos(gamma*np.pi/180)]], np.float32)

PntprojCoorplanR = np.zeros((4, 3),np.float32)
PntprojCoorplanR = Pntprojection[:, :] - originR
for i in range(0,4):
    PntprojCoorplanR[i,:]=np.dot(passage_horizontal_incline,PntprojCoorplanR[i,:])
    
#PntprojCoorplanR.dtype='float32'
#Dimension réelle de l'image
hauteur = np.sqrt((Pntprojection[0,0]-Pntprojection[1,0])**2+(Pntprojection[0,1]
        -Pntprojection[1,1])**2+(Pntprojection[0,2]-Pntprojection[1,2])**2)
largeur = np.sqrt((Pntprojection[1,0]-Pntprojection[2,0])**2+(Pntprojection[1,1]
        -Pntprojection[2,1])**2+(Pntprojection[1,2]-Pntprojection[2,2])**2)

C=np.array([[widthpi/2, (B[0]-A[0])/hauteur * heightpi],
           [(60e-2)/largeur * widthpi + widthpi/2, (B[0]-A[0])/hauteur * heightpi/2],
           [widthpi/2, 0],
           [-(60e-2)/largeur * widthpi + widthpi/2, (B[0]-A[0])/hauteur * heightpi/2]])

PntPrjtOnPng2 = np.zeros((4, 2),np.float32)
POIOnPng2 = np.zeros((4, 2),np.float32)
CadreAileOnPng = np.zeros((4, 2),np.float32)
POIOnPng2[:, 0] = (heightpi/new_height)*POI[:, 1]+heightpi/2
POIOnPng2[:, 1] = (widthpi/new_width)*POI[:, 2]+widthpi/2
PntPrjtOnPng2[:, 0] = (heightpi/new_height)*PntprojCoorplanR[:, 1]+heightpi/2
PntPrjtOnPng2[:, 1] = (widthpi/new_width)*PntprojCoorplanR[:, 2]+widthpi/2
CadreAileOnPng[:, 0] = (heightpi/new_height)*CadreAile[:, 1]+heightpi/2
CadreAileOnPng[:, 1] = (widthpi/new_width)*CadreAile[:, 2]+widthpi/2

# Matrice de passage reference-deformée
tform = cv2.getPerspectiveTransform(POIOnPng, PntPrjtOnPng)
tform2 = cv2.getPerspectiveTransform(POIOnPng2, PntPrjtOnPng2)

delta2 = (xa*B[0]+ya*B[1]+za*B[2])/d
D=B/delta2
ratio=(np.sqrt((D[0]-A[0])**2+(D[1]-A[1])**2+(D[2]-A[2])**2))/(np.sqrt((B[0]-A[0])**2+(B[1]-A[1])**2+(B[2]-A[2])**2))
ratio2=hauteur/np.sqrt((POI[2,0]-POI[2,1])**2)
#Déformation image de reference
new_heightpi = heightpi*hauteur/height
new_widthpi = widthpi*largeur/width

tf_img_warp = cv2.warpPerspective(img, tform, (int(widthpi), int(heightpi)))
tf_img_warp2 = cv2.warpPerspective(img, tform2, (int(new_widthpi), int(new_heightpi)))

#cv2.imwrite('/Users/yvan/Desktop/ETS_montreal/Cours/E21/MTR892/Deformeecv2100pi_cmBoutdaile2.png', tf_img_warp)#Enregistrement de l'image déformée
imgresize=tf_img_warp[int(C[2,1]):int(C[0,1]), int(C[3,0]):int(C[1,0])]
imgresize2=tf_img_warp2[int(C[2,1]):int(C[0,1]), int(C[3,0]):int(C[1,0])]

#cv2.imwrite('/Users/yvan/Desktop/ETS_montreal/Cours/E21/MTR892/Deformeecv2100pi_cmBoutdAileResize2.png', imgresize)#Enregistrement de l'image déformée
##------------------------------AFFICHAGE-----------------------------------##
fig = plt.figure(1)
ax = fig.add_subplot(111, projection='3d')
ax.scatter(0, 0, 0, color='b')
ax.scatter(originB[0], originB[1], originB[2], color='b')
ax.scatter(originR[0], originR[1], originR[2], color='b')
ax.scatter(CadreAile[:,0], CadreAile[:,1], CadreAile[:,2], color='r')
ax.scatter(D[0], D[1], D[2], color='g')
ax.plot(POI[:, 0], POI[:, 1], POI[:, 2], color='k')
ax.scatter(Pntprojection[:, 0], Pntprojection[:, 1], Pntprojection[:, 2], color='k')
ax.plot(PntprojCoorplanR[:, 0], PntprojCoorplanR[:, 1], PntprojCoorplanR[:, 2], color='k')
ax.plot_surface(xg1, yg1, zg1, rstride=10, cstride=10, color='b', alpha=0.2)
ax.plot_surface(xgp, ygp, zplane, rstride=10, cstride=10, color='r', alpha=0.2)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.axes.set_xlim3d(left=0, right=7)
ax.axes.set_ylim3d(bottom=-1, top=1)
ax.axes.set_zlim3d(bottom=-0.6, top=4)
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
plt.plot(C[:,0],C[:,1], marker='+', color='red')
plt.xlabel('Columns')
plt.ylabel('Rows')
plt.title('Déformée')
plt.show()

plt.figure(4)
plt.imshow(imgresize, origin='lower')
plt.scatter(C[:,0],C[:,1], marker='+', color='red')
plt.xlabel('Columns')
plt.ylabel('Rows')
plt.title('imgresize')
plt.show()

plt.figure(5)
plt.imshow(tf_img_warp2, origin='lower')
plt.scatter(C[:,0],C[:,1], marker='+', color='red')
plt.xlabel('Columns')
plt.ylabel('Rows')
plt.title('Avec rotation')
plt.show()

plt.figure(6)
plt.imshow(imgresize2, origin='lower')
plt.scatter(C[:,0],C[:,1], marker='+', color='red')
plt.xlabel('Columns')
plt.ylabel('Rows')
plt.title('Avec coordonnées réelles projetées resize')
plt.show()