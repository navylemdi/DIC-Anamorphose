#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  6 09:33:40 2021

@author: yvan
"""

import numpy as np
import sys
import time

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

def set_aspect_equal_3d(ax):
    """Fix equal aspect bug for 3D plots."""

    xlim = ax.get_xlim3d()
    ylim = ax.get_ylim3d()
    zlim = ax.get_zlim3d()

    from numpy import mean
    xmean = mean(xlim)
    ymean = mean(ylim)
    zmean = mean(zlim)

    plot_radius = max([abs(lim - mean_)
                       for lims, mean_ in ((xlim, xmean),
                                           (ylim, ymean),
                                           (zlim, zmean))
                       for lim in lims])

    ax.set_xlim3d([xmean - plot_radius, xmean + plot_radius])
    ax.set_ylim3d([ymean - plot_radius, ymean + plot_radius])
    ax.set_zlim3d([zmean - plot_radius, zmean + plot_radius])
    
def depliage(feuille, surface, saut, x ,y ,z, GradF, ProjVector):
        UnfoldedPnt = [None]*len(feuille.contours)
        print('Début dépliage')
        start = time.time()
        if surface.SurfaceType == 'Plan':
            for i in range(feuille.debut, len(feuille.contours), saut):
                UnfoldedPnt[i] = np.empty( [len(feuille.contours[i]), 3], dtype=np.float32)
                sys.stdout.write('\r' + str(round((i/(len(feuille.contours)-1))*100,2)) + '% ')#Affichage pourcentage de l'avancement
            
                for j in range(len(feuille.contours3D[i])):
                    NormalVector = np.array(GradF.subs([(x, feuille.Pntprojection[i][j, 0]), (y, feuille.Pntprojection[i][j, 1]), (z, feuille.Pntprojection[i][j, 2])])).astype(np.float64)/np.linalg.norm(np.array(GradF.subs([(x, feuille.Pntprojection[i][j, 0]), (y, feuille.Pntprojection[i][j, 1]), (z, feuille.Pntprojection[i][j, 2])])).astype(np.float64))
                    v = np.cross(np.squeeze(NormalVector), ProjVector)
                    c = np.dot(np.squeeze(NormalVector), ProjVector)
                    kmat = np.array([[0, -v[2], v[1]], 
                                      [v[2], 0, -v[0]], 
                                      [-v[1], v[0], 0]])
                    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * (1/(1+c))
                    UnfoldedPnt[i][j, :] = np.dot(rotation_matrix, feuille.Pntprojection[i][j, :])
            roulement_matrix = None
        elif surface.SurfaceType == 'Cylindre':
            ProjVector2 = np.array([0, 1, 0])#Vecteur horizontal vers les positifs
            ProjVector3 = np.array([-1, 0, 0])#Vecteur vertical vers les positifs
            CylAxe = np.array([surface.a, surface.b, surface.c])/np.linalg.norm(np.array([surface.a, surface.b, surface.c]))
            v = np.cross(CylAxe, ProjVector2)
            cos = np.dot(CylAxe, ProjVector2)
            kmat = np.array([[0, -v[2], v[1]], 
                             [v[2], 0, -v[0]], 
                             [-v[1], v[0], 0]])
            rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * (1/(1+cos)) #Rotation entre l'axe du cylindre et l'horizontal v
            VecteurOrientation = np.squeeze(np.array(GradF.subs([(x, feuille.PntCentreCadreProjection[0]), (y, feuille.PntCentreCadreProjection[1]), (z, feuille.PntCentreCadreProjection[2])])).astype(np.float64)/np.linalg.norm(np.array(GradF.subs([(x, feuille.PntCentreCadreProjection[0]), (y, feuille.PntCentreCadreProjection[1]), (z, feuille.PntCentreCadreProjection[2])])).astype(np.float64)))#Pntprojection[i, j, :]#
            VecteurOrientationRotation = np.dot(rotation_matrix, VecteurOrientation)/np.linalg.norm(np.dot(rotation_matrix, VecteurOrientation))
            v2 = np.cross(VecteurOrientationRotation, ProjVector3)
            cos2 = np.dot(VecteurOrientationRotation, ProjVector3)
            kmat2 = np.array([[0, -v2[2], v2[1]], 
                              [v2[2], 0, -v2[0]], 
                              [-v2[1], v2[0], 0]], dtype='float64')
            roulement_matrix = np.eye(3) + kmat2 + kmat2.dot(kmat2) * (1/(1+cos2))
            for i in range(feuille.debut, len(feuille.contours), saut):
                UnfoldedPnt[i] = np.empty( [len(feuille.contours[i]), 3], dtype=np.float32)
                sys.stdout.write('\r' + str(round((i/(len(feuille.contours)-1))*100,2)) + '% ')#Affichage pourcentage de l'avancement    
                for j in range(len(feuille.contours3D[i])):
                        OrientedPnt = np.dot(rotation_matrix, feuille.Pntprojection[i][j, :])#On tourne le cylindre pour l'aligner avec l'horizontale
                        RolledPnt= np.dot(roulement_matrix, OrientedPnt)#On tourne le cylindre pour l'aligner avec l'horizontale
                        NormalVector = np.array(GradF.subs([(x, feuille.Pntprojection[i][j, 0]), (y, feuille.Pntprojection[i][j, 1]), (z, feuille.Pntprojection[i][j, 2])])).astype(np.float64)/np.linalg.norm(np.array(GradF.subs([(x, feuille.Pntprojection[i][j, 0]), (y, feuille.Pntprojection[i][j, 1]), (z, feuille.Pntprojection[i][j, 2])])).astype(np.float64))
                        NormalVector = np.dot(rotation_matrix, np.squeeze(NormalVector))#Vecteur normaux à la surface tournée à l'horizontale
                        NormalVector = np.dot(roulement_matrix, np.squeeze(NormalVector))#Vecteur normaux à la surface tournée à la verticale
                        v2 = np.cross(NormalVector, ProjVector3)#Calcul des angles avec la verticale
                        theta = np.arcsin(v2[1])
                        UnfoldedPnt[i][j, :] = [0, RolledPnt[1], -surface.Radius*theta]
            sys.stdout.flush()
        print('\nFin dépliage')
        end = time.time()
        print('Temps ecoulé: ', time.strftime("%H:%M:%S", time.gmtime(end-start)))
        return UnfoldedPnt, rotation_matrix, roulement_matrix
    
