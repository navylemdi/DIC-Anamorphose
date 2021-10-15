from Module import *
import sys
import numpy as np

deck = Deck('/Users/yvan/Desktop/ETS_montreal/Cours/E21/MTR892 - Projet technique/AnamorphosePlane/deck.yaml')

#Creation of the surface object
S = Surface(deck.a, deck.b, deck.c, deck.Position, deck.radius, deck.SurfaceType)

speckle = Speckle(deck.NbImage, deck.PositionCentre, deck.Images(), deck.height, deck.width, deck.begining, deck.step)

Liste_Projection = speckle.ProjectionSpeckle(S)

List_Unfolded = speckle.UnfoldSpeckle(S)
#Depliage = List_Unfolded[0]
rotation_matrix = List_Unfolded[1]
roulement_matrix = List_Unfolded[2]

# print(np.shape(List_Unfolded))
# print(np.shape(Depliage))
# print(rotation_matrix)
# print(roulement_matrix)

WingFrameUnfolded, yf, zf = Fonction.Unfold_object_frame(deck.WingFrame, S.SurfaceType, S.Gradient(), rotation_matrix, roulement_matrix, deck.widthPrintable, deck.heightPrintable)

##--------------------------------AFFICHAGE----------------------------------##

p=Plot()
p.PlotReference(deck.NbImage, speckle.List_Sheets)

p.Plot3D(deck.NbImage, speckle.List_Sheets, Liste_Projection, deck.WingFrame)

p.PlotUnfolded(deck.NbImage, speckle.List_Sheets, List_Unfolded[0], WingFrameUnfolded, yf, zf)

##-----------------------------FIN AFFICHAGE---------------------------------##

##--------------------------DECOUPAGE IMPRESSION-----------------------------##

Fonction.Print(deck.PrintPath, yf, zf, deck.widthPrintable, deck.heightPrintable, deck.NbImage, speckle.List_Sheets, List_Unfolded[0], WingFrameUnfolded)
##------------------------FIN DECOUPAGE IMPRESSION---------------------------##