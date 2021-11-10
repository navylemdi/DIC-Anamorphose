import sympy as sym
from sympy import Symbol

class Surface:
    
    def __init__(self, deck):
        self.a = deck.a
        self.b = deck.b
        self.c = deck.c
        self.Posx = deck.Position[0]
        self.Posy = deck.Position[1]
        self.Posz = deck.Position[2]
        self.d = self.a*self.Posx + self.b*self.Posy + self.c*self.Posz
        self.Radius = deck.Radius
        self.Surface_type = deck.Surface_type

    def Equation(self):
        if self.Surface_type =='Plan':
            F = self.a*Symbol('x') + self.b*Symbol('y') + self.c*Symbol('z') -self.d
            
        elif self.Surface_type == 'Cylindre':
            F = (Symbol('x')-self.Posx)**2 + (Symbol('z')-self.Posz)**2 + (Symbol('y')-self.Posy)**2 - (self.a*(Symbol('x')-self.Posx) + self.b*(Symbol('y')-self.Posy) + self.c*(Symbol('z')-self.Posz))**2/(self.a**2+self.b**2+self.c**2) - self.Radius**2
        
        else:
            print('Surface non prise en compte')
        return F     
    
    def Gradient(self):
        return sym.Matrix([sym.diff(self.Equation(),Symbol('x')), sym.diff(self.Equation(),Symbol('y')), sym.diff(self.Equation(),Symbol('z'))])