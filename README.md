# Anamorphosis

## Detection of edges and deformations on a surface of several sheets of speckles
### Installation
To begin, you have to download some packages listed in the requirements.txt file
To use Anamorphose.py, you also have to download Fonction.py, Feuille.py, Surface.py and Plot.py files.

### Constant section

In the Anamorphose.py, parameters of the anamorphosis as the surface properties and the speckle files have to be set in the *CONSTANTES* section.

Due to the multitude of points in the speckle, to test the code, we only anamorphose a few points. The variable *saut* represents the anamorphic step. `saut=2` will anamorphose half of all the points of a sheet of speckles.

*debut* variable represents the first index of the contour list calculated by OpenCV to be considered ine the anamorphosis. Usually `debut=3` is sufficient to avoid black filling of the result by the algorithm.

You need to implement the size of the speckle sheets you want to anamorphose and the size of the anamorphosed speckle sheets you want to get. Use the *height*, *width*, *heightPrintable* and *widthPrintable* variables to do so.

*PrintPath* is the path of the anamorphosed speckle sheets folder.

In the *List_image* list you must put all the images read by openCV you want to use. It has to be a numpy array.

In the *Feuille_pos* list you must put the center position of all your sheet. With the height and the width respectively in the first and second position.

Then you must to implement the surface properties. *(a, b, c)* vector represents the normal vector in case of a plane surface and the axis of a cylinder in the case of a cylinder surface case.

If needed, you can also represent the frame of the wing with the *CadreAile* numpy array.
Seuls les cas plan et cylindre sont implémentés. 

## AnamorphosePlanaire.py

Image warping on a plane with cv2.warpPerspective.
It was just a test program.

## Trucs à faire

- Améliorer temps de traitements : Parallélisation du calcul des projections de chaque feuille.
                                   Utilisation des equations analytiques pour s'affranchir de sympy.solve.
                                   Approximation numérique de la solution de la projection.
- Déplier une surface polynomiale quelconque.
- Effectuer l'anamorphose dans un espace où la caméra n'est plus le centre et orienté dans n'importe quelle direction.
