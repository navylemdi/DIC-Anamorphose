#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 21 09:36:31 2021

@author: yvan
"""

class Surface:
    
    def __init__(self, a, b, c, Posx, Posy, Posz, d, Radius, SurfaceType):
        self.a = a
        self.b = b
        self.c = c
        self.Posx = Posx
        self.Posy = Posy
        self.Posz = Posz
        self.d = d
        self.Radius = Radius
        self.SurfaceType = SurfaceType
        
    def Equation(self, x, y, z):
        if self.SurfaceType =='Plan':
            F = self.a*x + self.b*y + self.c*z -self.d
            
        elif self.SurfaceType == 'Cylindre':
            F = (x-self.Posx)**2 + (z-self.Posz)**2 + (y-self.Posy)**2 - (self.a*(x-self.Posx) + self.b*(y-self.Posy) + self.c*(z-self.Posz))**2/(self.a**2+self.b**2+self.c**2) - self.Radius**2
        
        else:
            print('Surface non prise en compte')
        return F     
         