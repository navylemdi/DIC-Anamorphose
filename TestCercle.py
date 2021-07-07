#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 13 11:47:08 2021

@author: yvan
"""
import numpy as np
import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d.axes3d import Axes3D
import sympy as sym
from sympy import Symbol
from sympy.solvers import solve
from IPython import get_ipython

get_ipython().magic('reset -sf')
plt.close('all')
##--------------------------------FONCTIONS----------------------------------##

def generate_circle_by_vectors(t, C, r, n, u):
    n = n/np.linalg.norm(n)
    u = u/np.linalg.norm(u)
    P_circle = r*np.cos(t)[:,np.newaxis]*u + r*np.sin(t)[:,np.newaxis]*np.cross(n,u) + C
    return P_circle

def generate_circle_by_angles(t, C, r, theta, phi):
    # Orthonormal vectors n, u, <n,u>=0
    n = np.array([np.cos(phi)*np.sin(theta), np.sin(phi)*np.sin(theta), np.cos(theta)])
    u = np.array([-np.sin(phi), np.cos(phi), 0])
    # P(t) = r*cos(t)*u + r*sin(t)*(n x u) + C
    P_circle = r*np.cos(t)[:,np.newaxis]*u + r*np.sin(t)[:,np.newaxis]*np.cross(n,u) + C
    return P_circle
##------------------------------FIN FONCTIONS--------------------------------##

##-------------------------------CONSTANTES----------------------------------##

dim = (0.25,0.25)
#Plane 1 - normal vector
xa = 1
ya = 0
za = 0
d = 0.25
#Plane aile - normal vector
beta = 45 #angle entre horizontale et plan aile (deg) Rx
gamma = 10 #angle entre verticale et plan aile (deg) Ry
a = -np.sin(beta*np.pi/180)
b = 0.5#-np.sin(gamma*np.pi/180)
c = np.cos(beta*np.pi/180)#*np.cos(gamma*np.pi/180)
dprim = -1
# a=dim[1]/(dim[1]/np.tan(beta*np.pi/180)-d)
# b=0
# c=1
# dprim=-dim[1]/2+a*d

r = 1e-1   # Radius
nbCercles = 1
C = np.zeros((nbCercles,3))
C[:,0] = d #Axe x
C[:,1] = 0*0.1#(dim[0]-r)*np.random.rand(nbCercles)-(dim[0]-r)/2 #Axe y
C[:,2] = 0*0.25#(dim[1]-r)*np.random.rand(nbCercles)-(dim[1]-r)/2 #Axe z
theta = 90*np.pi/180     # Azimuth
phi   = 0    # Zenith

# Create a grid of x and y
yg1, zg1 = np.meshgrid(np.arange(-dim[0]/2, dim[0]/2, 0.01), np.arange(-dim[1]/2, dim[1]/2, 0.01))
xgp, ygp = np.meshgrid(np.arange(0, 2.5, 0.001), np.arange(-0.6, 0.6, 0.01))

x1 = (d-ya*yg1-za*zg1)/xa # Plan capteur
zplane = (dprim - b*(ygp)**2 - a*((xgp)**2))/c# Plan aile
#zplane = (dprim-b*ygp-a*xgp)/c

points_of_interest = np.array([[d, -dim[0]/2, -dim[1]/2],
                               [d, -dim[0]/2, dim[1]/2],
                               [d, dim[0]/2, -dim[1]/2], 
                               [d, dim[0]/2, dim[1]/2]])# Coordonnées des points sur l'image de référence
##------------------------------FIN CONSTANTES-------------------------------##

##--------------------------------PROJECTION---------------------------------##

t = np.linspace(0, 2*np.pi, 50)
P_gen = np.zeros((len(C),len(t),3))
for i in range(len(C)):
     P_gen[i] = generate_circle_by_angles(t, C[i], r, theta, phi)

x, y, z = Symbol('x'), Symbol('y'), Symbol('z')
F = a*x**2+ b*(y)**2 + c*z - dprim #Fonction de surface 3D F(x,y,z) = 0 où (x,y,z) appartient à la surface 3D
delta1 = Symbol('delta1', positive=True)
sol = np.zeros((len(C),len(t)))
sol3 = np.zeros(len(C))
for j in range(len(C)):
    try:
        sol3[j] = solve( F.subs([(x, C[j,0]/delta1), (y, C[j,1]/delta1), (z,C[j,2]/delta1)]), delta1)[0]
    except IndexError:
        print("Il n'existe pas de solution pour le centre des cercles. Vérifier l'equation de la surface")
        
    for i in range(len(t)):
        try:
            sol[j,i] = solve( F.subs([(x, P_gen[j,i,0]/delta1), (y, P_gen[j,i,1]/delta1), (z, P_gen[j,i,2]/delta1)]), delta1)[0]#a*((P_gen[j,i,0]/delta1)**(2)) + b*(2*P_gen[j,i,1]/delta1)**(2) + c*(P_gen[j,i,2]/delta1) - dprim , delta1)[0]
        except IndexError:
            print("Il n'existe pas de solution pour les cercles. Vérifier l'equation de la surface")
            
sol2 = np.zeros(4)
for i in range(4):
    try:
        sol2[i] = solve( F.subs([(x, points_of_interest[i,0]/delta1), (y, points_of_interest[i,1]/delta1), (z, points_of_interest[i,2]/delta1)]), delta1)[0]
    except IndexError:
        print("Il n'existe pas de solution pour le cadre. Vérifier l'equation de la surface")
#delta=(np.sqrt(-4*(-dprim)*a*P_gen[:,:,0]**2+(b*P_gen[:,:,1]+c*P_gen[:,:,2])**2)-(b*P_gen[:,:,1]+c*P_gen[:,:,2]))/(2*(-dprim))
#delta = (a*P_gen[:,:,0]+b*P_gen[:,:,1]+c*P_gen[:,:,2])/dprim
Pntprojection = P_gen[:]/sol[:,:,None]#delta[:,:,None]# Coordonnées des points projetés
Pntcadreprojection = points_of_interest[:]/sol2[:,None]
Cprojected = C[:]/sol3[:, None]

##------------------------------FIN PROJECTION-------------------------------##

##--------------------------------DEPLIAGE---------------------------------##

GradF = sym.Matrix([sym.diff(F,x), sym.diff(F,y), sym.diff(F,z)]) #Gradient (vecteur normal) de la surface obtenu à partir de l'equation de la surface
ProjVector = np.array([1, 0, 0])#Direction de dépliage de la surface 3D
UnfoldedPnt = np.zeros((len(C),len(t),3))
for i in range (len(C)):
    for j in range(len(t)):
        NormalVector = np.array(GradF.subs([(x, Pntprojection[i, j, 0]), (y, Pntprojection[i, j, 1]), (z, Pntprojection[i, j, 2])])).astype(np.float64)/np.linalg.norm(np.array(GradF.subs([(x, Pntprojection[i, j, 0]), (y, Pntprojection[i, j, 1]), (z, Pntprojection[i, j, 2])])).astype(np.float64))
        v = np.cross(np.squeeze(NormalVector), ProjVector)
        c = np.dot(np.squeeze(NormalVector), ProjVector)
        kmat = np.array([[0, -v[2], v[1]], 
                         [v[2], 0, -v[0]], 
                         [-v[1], v[0], 0]])
        rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * (1/(1+c))
        UnfoldedPnt[i, j, :] = np.dot(rotation_matrix, Pntprojection[i, j, :])
        
##------------------------------FIN DEPLIAGE---------------------------------##

##-------------------------------AFFICHAGE----------------------------------##

plt.figure(1)
for i in range(len(C)):
    plt.plot(P_gen[i,:,1], P_gen[i,:,2], color='blue')
plt.scatter(C[:,1],C[:,2],color='k', marker='+')
plt.scatter(points_of_interest[:,1],points_of_interest[:,2],color='r', marker='+')
plt.title('Réference')
plt.grid()
plt.xlim([-dim[0]/2, dim[0]/2])
plt.ylim([-dim[1]/2, dim[1]/2])
plt.axis('equal')
plt.show()

fig = plt.figure(2)
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x1, yg1, zg1, rstride=10, cstride=10, color='b', alpha=0.2)
ax.plot_surface(xgp, ygp, zplane, rstride=10, cstride=10, color='r', alpha=0.2)
ax.scatter(0,0,0,color='b')
ax.scatter(C[:,0],C[:,1],C[:,2],color='r', marker='+')
ax.scatter(Pntcadreprojection[:,0],Pntcadreprojection[:,1],Pntcadreprojection[:,2],color='red', marker='+')
ax.scatter(Cprojected[:,0], Cprojected[:,1], Cprojected[:,2], color='black', marker='+')
for i in range(len(C)):
    ax.plot(P_gen[i,:,0], P_gen[i,:,1], P_gen[i,:,2], color='blue')
    ax.plot(Pntprojection[i,:,0], Pntprojection[i,:,1], Pntprojection[i,:,2], color='blue')
    ax.plot(UnfoldedPnt[i,:,0], UnfoldedPnt[i,:,1], -UnfoldedPnt[i,:,2], color='red')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()

plt.figure(3)
for i in range(len(C)):
      plt.plot(UnfoldedPnt[i, :, 1], -UnfoldedPnt[i, :, 2], color='blue')
plt.title('Dépliée')
plt.grid()
plt.axis('equal')
plt.show()
##------------------------------FIN AFFICHAGE--------------------------------##
