from Module.Speckle.Sheets import Sheets


class Speckle(Sheets):
    
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

    def UnfoldSpeckle(self, S):
        List_Unfolded = []
        B = []
        for i in range(self.Nbimage):
            if i ==0:
                A, rotation_matrix,  roulement_matrix= self.List_Sheets[0].Unfold(S)
                List_Unfolded.append(A)
            else:
                List_Unfolded.append(self.List_Sheets[i].Unfold(S)[0])
        return List_Unfolded, rotation_matrix, roulement_matrix