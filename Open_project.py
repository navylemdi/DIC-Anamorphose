from Module import *

p = Project()
p.open(r'/Users/yvan/Desktop/ETS_montreal/Cours/E21/MTR892 - Projet technique/AnamorphosePlane/New_project.yaml')
p.PlotReference()
p.Plot3D()
p.PlotUnfolded()
#Plot().PlotReference(p.deck, p.List_Sheets)
p.Show_plots()
