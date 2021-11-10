import sympy as sym
from sympy import Symbol

class Surface:
    
    def __init__(self, a, b, c, Pos, Radius, SurfaceType):
        self.a = a
        self.b = b
        self.c = c
        self.Posx = Pos[0]
        self.Posy = Pos[1]
        self.Posz = Pos[2]
        self.d = self.a*self.Posx + self.b*self.Posy + self.c*self.Posz
        self.Radius = Radius
        self.SurfaceType = SurfaceType

    def Equation(self):
        if self.SurfaceType =='Plan':
            F = self.a*Symbol('x') + self.b*Symbol('y') + self.c*Symbol('z') -self.d
            
        elif self.SurfaceType == 'Cylindre':
            F = (Symbol('x')-self.Posx)**2 + (Symbol('z')-self.Posz)**2 + (Symbol('y')-self.Posy)**2 - (self.a*(Symbol('x')-self.Posx) + self.b*(Symbol('y')-self.Posy) + self.c*(Symbol('z')-self.Posz))**2/(self.a**2+self.b**2+self.c**2) - self.Radius**2
        
        else:
            print('Surface non prise en compte')
        return F     
    
    def Gradient(self):
        return sym.Matrix([sym.diff(self.Equation(),Symbol('x')), sym.diff(self.Equation(),Symbol('y')), sym.diff(self.Equation(),Symbol('z'))])
