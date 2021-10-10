# Anamorphosis

## Detection of edges and deformations on a surface of several sheets of speckles

### Installation
To begin, you have to download some packages listed in the requirements.txt file
To use Anamorphose.py, you also have to download Fonction.py, Feuille.py, Surface.py and Plot.py files.

### Constant section

In the Anamorphose.py, parameters of the anamorphosis as the surface properties and the speckle files have to be set in the *CONSTANTES* section.

Due to the multitude of points in the speckle, to test the code, we only anamorphose a few points. The variable *saut* represents the anamorphic step. `saut=2` will anamorphose half of all the points of a sheet of speckles.

*debut* variable represents the first index of the contour list calculated by OpenCV to be considered ine the anamorphosis. Usually `debut=3` is sufficient to avoid black filling of the result by the algorithm.

You need to implement the size of the speckle sheets you want to anamorphose and the size of the anamorphosed speckle sheets you want to get. Use the *height*, *width*, *heightPrintable* and *widthPrintable* variables to do so. The unit is meter.

*PrintPath* is the path of the anamorphosed speckle sheets folder.

In the *List_image* list you must put all the images read by OpenCV you want to use. It has to be a numpy array.

In the *Feuille_pos* list you must put the center position of all your sheet. With the height and the width respectively in the first and second position.

If needed, you can also represent the frame of the wing with the *CadreAile* numpy array.

### Surface implementation

Then you must to implement the surface properties with the `Surface(a, b, c, Posx, Posy, Posz, d, Radius, SurfaceType)` class :
- *(a, b, c)* vector represents the normal vector in case of a plane surface and the axis of a cylinder in the case of a cylinder surface case.
- `Pos=[Posx, Posy, Posz]` array represents the position relative to the camera of a point belongs to the axis of the cylinder (if the surface is a cylinder).
-  *d* is the constant in your plane equation *a.X + b.Y + c.Z=d* in the case of plane. The position relative to the camera
-  *Radius* is the radius of the cylinder in the case of a cylinder.
-  *SurfaceType* is a string to tell the programm that you want to anamorphose on a `Cylindre` or on a `Plan`. Only cylinder or plane surface case are implemented.

### Feuille implementation

Then you must to create your `Feuille(centreH, centreV, image, height, width, debut, saut, d)` object:
- *centreH*, *centreV* and *d* represents the center of your speckle sheet in respectively in the Y, Z, and X directions relative to the camera.
- *image* is the numpy array created by OpenCV.
- *height* and *width* are the size of your speckle sheets in meter.

Use `projection(saut, S)` to calculate the 3D projection coordinates of each dot of your speckle (in the first index of the output). The second index is the 3D coordinates of the center of your speckle sheets.

### Unfolding section

To unfold your anamorphed speckle, you have to use the `depliage(feuille, surface, saut, ProjVector)` function of the Fonction.py file:
- *feuille* is your speckle sheet object 
- *surface* is your surface object

### Plot section

Then you finally can plot all the results.

To plot your speckle sheets use the `PlotReference(Nbimage, debut, saut, Liste_Feuille)` function of the Plot.py file.
To plot the anamorphosed speckle in 3D use the `Plot3D(Nbimage, debut, saut, Liste_Feuille, Liste_Projection, CadreAile, d)` function of the Plot.py file.
To plot the unfolded speckle use the `PlotUnfolded(Nbimage, debut, saut, Liste_Feuille, Liste_depliage, CadreAileUnfolded, yf, zf)` function of the Plot.py file.

### Print section

To save your results use Fonction.Print(PrintPath, yf, zf, widthPrintable, heightPrintable,Nbimage, debut, saut, Liste_Feuille, Liste_depliage, CadreAileUnfolded)

## AnamorphosePlanaire.py

Image warping on a plane with cv2.warpPerspective.
It was just a test program.

## Trucs à faire

- Améliorer temps de traitements : Parallélisation du calcul des projections de chaque feuille.
                                   Utilisation des equations analytiques pour s'affranchir de sympy.solve.
                                   Approximation numérique de la solution de la projection.
- Déplier une surface polynomiale quelconque.
- Effectuer l'anamorphose dans un espace où la caméra n'est plus le centre et orienté dans n'importe quelle direction.
