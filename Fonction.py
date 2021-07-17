#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  6 09:33:40 2021

@author: yvan
"""

import numpy as np

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
    
def Translation(x, y,z):
    return np.array([[1, 0, 0, x],
                     [0, 1, 0, y],
                     [0, 0, 1, z],
                     [0, 0, 0, 1]])
