#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 14 15:18:59 2021

@author: yvan
"""
import cv2 as cv2
import matplotlib.pyplot as plt
import numpy as np
from IPython import get_ipython

get_ipython().magic('reset -sf')
plt.close('all')
cv2.destroyAllWindows()

##--------------------------------FONCTIONS----------------------------------##

def Pix2Meter(Pospix, image, Lim_inf_H, Lim_max_H, Lim_inf_V, Lim_max_V):
    Posmet = np.zeros((len(Pospix),2), np.float32)
    Posmet[:, 0] = (Lim_max_V-Lim_inf_V)*Pospix[:,1]/image.shape[0]+Lim_inf_V
    Posmet[:, 1] = (Lim_max_H-Lim_inf_H)*Pospix[:,0]/image.shape[1]+Lim_inf_H
    return Posmet

def Meter2Pix(Posmet, image, Lim_inf_H, Lim_max_H, Lim_inf_V, Lim_max_V):
    Pospix = np.zeros((len(Posmet),2), np.float32)
    Pospix[:, 0] = image.shape[0]*(Posmet[:,1]-Lim_inf_V)/(Lim_max_V-Lim_inf_V)
    Pospix[:, 1] = image.shape[1]*(Posmet[:,0]-Lim_inf_H)/(Lim_max_H-Lim_inf_H)
    return Pospix
##------------------------------FIN FONCTIONS--------------------------------##

##-------------------------------CONSTANTES----------------------------------##

saut = 500 #Taille du saut de point dans la liste contours
debut = 2 #Debut des boucles for pour les projections
height = 27.9e-2#29.7e-2#hauteur en m de l'image de reference(m)
width = 21.6e-2#21e-2#largeur en m de l'image de reference(m)
WingWidth = 60e-2 #largeur zone analyse de l'aile (m)
WingHeight = 3 #hauteur zone analyse de l'aile (m)
#Angles
alpha = 10.0#Angle de champ de vue 
beta = 45 #Angle aigu entre aile et axe optique
gamma = 180-90-beta #angle entre capteur et plan aile (deg)
theta = 90.0-gamma #Angle entre normale capteur et plan aile (deg)

passage_horizontal_incline=np.array([[np.cos(gamma*np.pi/180), 0, -np.sin(gamma*np.pi/180)],
                                     [0,                      1,                       0],
                                     [np.sin(gamma*np.pi/180), 0,  np.cos(gamma*np.pi/180)]], np.float32)#Matrice de rotation
#Calcul Point debut champ de vue
# l = np.sqrt(1**2 + 1**2)
# A = np.array([l*np.cos((alpha/2)*np.pi/180), 0, l*np.sin((-alpha/2)*np.pi/180)])
# B = np.array([A[0] + (WingHeight)*np.cos(beta*np.pi/180), 0, A[2] + (WingHeight)*np.sin(beta*np.pi/180)])
l = np.sqrt(0.9**2 + 2.5**2 + 0.9*2.5*np.cos(105*np.pi/180))
A = np.array([l*np.cos((alpha/2)*np.pi/180), 0, l*np.sin((-alpha/2)*np.pi/180)])
B = np.array([A[0] + (5.5-2.5)*np.cos(beta*np.pi/180), 0, A[2] + (5.5-2.5)*np.sin(beta*np.pi/180)])
C1 = np.array([[(B[0]+A[0])/2, (WingWidth)/2, (B[2]+A[2])/2],
 [(B[0]+A[0])/2, (-WingWidth)/2, (B[2]+A[2])/2]])
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
d = 0.3#A[0]#7.52928#

POI = np.array([[d, -width/2, -height/2],
                [d, -width/2, height/2],
                [d, width/2, height/2],
                [d, width/2, -height/2]], np.float32)

#Creation des plans dans l'espace centré sur le centre optique
yg1, zg1 = np.meshgrid(np.arange(-WingWidth/2, WingWidth/2, WingWidth/50), np.arange(-WingHeight/2, WingHeight/2, WingHeight/50))
xg1 = (d-ya*yg1-za*zg1)/xa
xgp, ygp = np.meshgrid(np.arange(A[0], B[0], (B[0]-A[0])/50), np.arange(-WingWidth/2, WingWidth/2, WingWidth/50))
zplane = (dprim-b*ygp-a*xgp)/c

##------------------------------FIN CONSTANTES-------------------------------##

##--------------------------------CONTOURS-----------------------------------##
image = cv2.imread("/Users/yvan/Desktop/ETS_montreal/Cours/E21/MTR892/Banque_Speckle/speckle_1.png")
#cv2.imshow('Reference', image)
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#cv2.imshow('Gray scale', image_gray)
ret,thresh = cv2.threshold(image_gray, 127, 255, cv2.THRESH_BINARY)#Conversion en NB
#cv2.imshow('Threshold', thresh)

contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_L1)#Trouver contours

#image_copy = image.copy()
#cv2.drawContours(image_copy, contours, -1, (0, 255, 0), thickness=1, lineType=cv2.LINE_AA)
#cv2.imshow('None_approximation',image_copy)

##--------------------------------FIN CONTOURS-------------------------------##

##--------------------------------PROJECTION---------------------------------##

# Transformation coordonées contours repère 2D en repère 3D
contours3D = [None]*len(contours)
for i in range(debut, len(contours), saut):
    contours3D[i] = np.empty( [len(contours[i]), 3], dtype=np.float32)
    temp = Pix2Meter(contours[i][:, 0], image,  -height/2, height/2, -width/2, width/2)
    contours3D[i][:, 0] = d
    contours3D[i][:, 1] = temp[:, 0]
    contours3D[i][:, 2] = temp[:, 1]
    
#Calcul projection sur plan incliné
Pntprojection = [None]*len(contours)
for i in range(debut, len(contours), saut):
    Pntprojection[i] = np.empty( [len(contours[i]), 3], dtype=np.float32)
    delta = (a*contours3D[i][:, 0]+b*contours3D[i][:, 1]+c*contours3D[i][:, 2])/dprim
    Pntprojection[i] = contours3D[i]/delta[:, None]# Coordonnées dans l'espace des points projetés

#Calcul coordonées projection dans le plan incliné
PntprojCoorplanR = [None]*len(contours)
for i in range(debut, len(contours), saut):
    PntprojCoorplanR[i] = np.empty( [len(contours[i]), 3], dtype=np.float32)
    for j in range(0,len(contours[i])):
        PntprojCoorplanR[i][j,:] = np.dot(passage_horizontal_incline, Pntprojection.copy()[i][j,:])

#Calcul 
CadreAileCoorPlanR = np.zeros((4,3))
for i in range (0, 4):
    CadreAileCoorPlanR[i] = np.dot(passage_horizontal_incline, CadreAile[i, :])
CadreAileOnPng = Meter2Pix(CadreAile[:, 1:3], image, -height/2, height/2, -width/2, width/2)# np.zeros((4, 2),np.float32)

##------------------------------FIN PROJECTION-------------------------------##

##--------------------------------AFFICHAGE----------------------------------##

fig1=plt.figure(1)
ax = fig1.add_subplot(111, aspect='equal')
for i in range(debut, len(contours), saut):
    plt.plot(contours[i][:, 0][:, 0], contours[i][:, 0][:, 1], marker=None, color='black')
    ax.fill(contours[i][:, 0][:, 0], contours[i][:, 0][:, 1],'k',zorder=10)
plt.title('Image référence (pix)')
plt.show()

fig2 = plt.figure(2)
ax = fig2.add_subplot(111, projection='3d')
ax.scatter(0, 0, 0, color='b')
for i in range (debut, len(contours), saut):
    ax.plot(contours3D[i][:, 0], contours3D[i][:, 1], contours3D[i][:, 2], color='k', marker=None)
    ax.plot(Pntprojection[i][:, 0], Pntprojection[i][:, 1], Pntprojection[i][:, 2], color='k', marker=None)
    #ax.plot(PntprojCoorplanR[i][:, 0], PntprojCoorplanR[i][:, 1], PntprojCoorplanR[i][:, 2], color='r', marker=None)
ax.plot_surface(xg1, yg1, zg1, color='b', alpha=0.5)
ax.plot_surface(xgp, ygp, zplane, color='r', alpha=0.5)
ax.scatter(CadreAile[:,0], CadreAile[:,1], CadreAile[:,2], color='r')
ax.scatter(POI[:,0], POI[:,1], POI[:,2])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.title('Image référence et projetée 3D (m)')
plt.show()

fig3 = plt.figure(3)
fig3.set_size_inches(WingWidth/0.0254, WingHeight/0.0254)
ax = fig3.add_subplot(111, aspect='equal')
axe = plt.gca()
x_axis = axe.axes.get_xaxis()
x_axis.set_visible(False)
y_axis = axe.axes.get_yaxis()
y_axis.set_visible(False)
for i in range (debut,len(contours), saut):
    plt.plot(PntprojCoorplanR[i][:,1], PntprojCoorplanR[i][:,2], color='k')
    ax.fill(PntprojCoorplanR[i][:,1], PntprojCoorplanR[i][:,2], 'k',zorder=10)
plt.scatter(CadreAileCoorPlanR[:,1], CadreAileCoorPlanR[:,2], marker='+', color='b')
#ax.axis('equal')
plt.xlim( min(CadreAileCoorPlanR[:,1]), max(CadreAileCoorPlanR[:,1]) )
plt.ylim( min(CadreAileCoorPlanR[:,2]), max(CadreAileCoorPlanR[:,2]) ) 
#plt.title('Image projetée (m)')
plt.box(False)
plt.grid()
plt.show()
fig3.tight_layout()
fig3.savefig("SpeckleAnamorphose(m)LETTERCTA.pdf")

# fig4 = plt.figure(4)
# ax = fig4.add_subplot(111, aspect='equal')
# axe = plt.gca()
# for i in range (debut,len(contours), saut):
#     temp=Meter2Pix(PntprojCoorplanR[i][:, 1:],image, -height/2, height/2, 
#              -width/2, width/2)
#     plt.plot(temp[:, 0], temp[:, 1], color='k')
#     ax.fill(temp[:, 0], temp[:, 1], 'k',zorder=10)
# # x_axis = axe.axes.get_xaxis()
# # x_axis.set_visible(False)
# # y_axis = axe.axes.get_yaxis()
# # y_axis.set_visible(False)
# plt.scatter(CadreAileOnPng[:, 0], CadreAileOnPng[:, 1], marker='+')
# plt.title('Image projetée (pix)')
# plt.box(False)
# plt.grid()
# plt.show()
# #fig4.savefig("SpeckleAnamorphose", bbox_inches='tight')

##----------------------------FIN AFFICHAGE----------------------------------##
