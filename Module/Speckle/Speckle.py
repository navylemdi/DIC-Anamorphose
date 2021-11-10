from Module.Speckle.Sheets import Sheets


class Speckle(Sheets):
    
    def __init__(self, deck):
        self.Nbimage = int(deck.NbImage)
        Sheets_pos = deck.Position_centre
        List_image = deck.Images()
        height = deck.Height
        width = deck.Width
        begining = deck.Begining
        step = deck.Step
        self.List_Sheets=[]
        self.Generic_name = deck.Generic_name
        for i in range(self.Nbimage):
            self.List_Sheets.append(Sheets(Sheets_pos[i, 0], Sheets_pos[i, 1], Sheets_pos[i, 2],  List_image[i], height, width, begining, step))
    
    def ProjectionSpeckle(self, S):
        Liste_Projection = []
        for i in range(self.Nbimage):
            Compute_information = self.Generic_name + str(i+1)
            print("------------------------\n")
            print(Compute_information+':\n')
            Liste_Projection.append(self.List_Sheets[i].projection(S)[0])
        return Liste_Projection

    def UnfoldSpeckle(self, S):
        List_Unfolded = []
        B = []
        for i in range(self.Nbimage):
            Compute_information = self.Generic_name + str(i+1)
            print("------------------------\n")
            print(Compute_information+':\n')
            if i ==0:
                A, rotation_matrix,  roulement_matrix= self.List_Sheets[0].Unfold(S)
                List_Unfolded.append(A)
            else:
                List_Unfolded.append(self.List_Sheets[i].Unfold(S)[0])
        return List_Unfolded, rotation_matrix, roulement_matrix