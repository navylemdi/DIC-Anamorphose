from Sheets import Sheets

class Speckle():
    
    def __init__(self, Nbimage, Sheets_pos, List_image, height, width, begining, step):
        self.Nbimage = int(Nbimage)
        self.List_Sheets=[]
        for i in range(self.Nbimage):
            self.List_Sheets.append(Sheets(Sheets_pos[i, 0], Sheets_pos[i, 1], Sheets_pos[i, 2], List_image[i], height, width, begining, step))
    
    def ProjectionSpeckle(self, S):
        Liste_Projection = []
        for i in range(self.Nbimage):
            Liste_Projection.append(self.List_Sheets[i].projection(S)[0])
        return Liste_Projection