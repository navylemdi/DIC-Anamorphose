from Module import *

deck = Deck('./TestETS/deck_ETS.yaml')

Camera = Camera(deck)
#Creation of the surface object
S = Surface(deck)
#Creation of the speckle
speckle = Speckle(deck)
#Projection of the speckle
Liste_Projection = speckle.ProjectionSpeckle(S)

#Unfolding of the speckle
List_Unfolded = speckle.UnfoldSpeckle(S)
rotation_matrix = List_Unfolded[1]
roulement_matrix = List_Unfolded[2]

#Unfolding of the wingframe and meshing of the anamorphosed speckle for print
WingFrameUnfolded, yf, zf = Fonction.Unfold_object_frame(deck, S, rotation_matrix, roulement_matrix)

##--------------------------------AFFICHAGE----------------------------------##

p=Plot()
p.PlotReference(deck, speckle.List_Sheets)#Plot the loaded speckle

p.Plot3D(deck, speckle.List_Sheets, Liste_Projection, Camera)#Plot in 3D the loaded speckle and the anamorphosed

p.PlotUnfolded(deck, speckle.List_Sheets, List_Unfolded[0], WingFrameUnfolded, yf, zf)#Plot the unfolded speckle

##-----------------------------FIN AFFICHAGE---------------------------------##

##--------------------------DECOUPAGE IMPRESSION-----------------------------##

Fonction.Print(deck, yf, zf, speckle.List_Sheets, List_Unfolded[0], WingFrameUnfolded)
##------------------------FIN DECOUPAGE IMPRESSION---------------------------##

project = Project()

project.save('TestETS', r'./TestETS', deck, List_Unfolded[0], WingFrameUnfolded, yf, zf, Liste_Projection)

p.Show_plots()