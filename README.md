# Anamorphosis

## Detection of edges and deformations on a surface of several sheets of speckles

### Installation
To begin, you have to download some packages listed in the requirements.txt file
To use main.py, you also have to download Module folder.

### deck.yaml initialisation

Main structure for `deck.yaml` file:

```
Camera:
  focal_length: 50 #mm
  sensor_height: 29 #mm
  
Input_Speckle: 
  step: 500
  begining: 3
  height: 27e-2
  width: 21e-2
  Path: '.'
  Generic_Name: 'Speckle_'
  NbImage: 3
  PositionCentre: [[0, 0, 2], [-21e-2, 0, 2], [21e-2, 0, 2]]

Surface : 
  a: 0
  b: 0
  c: 1
  Radius : 0.4
  Position : [3, 0, 0]
  Surface_Type : 'Cylindre'
  Wingframe : [[ 2.6,  0.,   0.2],
               [ 2.6,  0.,  -0.2],
               [ 3.,  -0.4,  0. ],
               [ 3.,   0.4,  0. ]]

Output_Speckle:
  heightPrintable: 27.9e-2
  widthPrintable: 21.6e-2
  PrintPath: './ImagePrintable'
```
#### Camera section
```
Camera:
  focal_length: 50 #mm
  sensor_height: 29 #mm
 ```
 Enter your focal length and sensor height in mm in these variable. Field of view in degrees and radians will be calculated and printed out.
 It is also used to plot the field of view cone in the 3D viewer.
#### Input_Speckle section
```
Input_Speckle: 
  step: 500
  begining: 3
  height: 27e-2
  width: 21e-2
  Path: './'
  Generic_Name: 'Speckle_'
  NbImage: 3
  PositionCentre: [[0, 0, 2], [-21e-2, 0, 2], [21e-2, 0, 2]]
```
Due to the multitude of points in the speckle, to test the code, we only anamorphose a few points. The variable *step* represents the anamorphic step. `step=2` will anamorphose half of all the points of a sheet of speckles.

*begining* variable represents the first index of the contour list calculated by OpenCV to be considered ine the anamorphosis. Usually `begining=3` is often sufficient to avoid black filling of the result by the algorithm.

You need to implement the size of the speckle sheets you want to anamorphose and the size of the anamorphosed speckle sheets you want to get. Use the *height*, *width*. The unit is meter.

*Path* is the path of the classic speckle sheets folder.

*Generic_Name* is the name of your speckle sheets. It as to be the shape of *Generic_Name_XX* with XX a number.

*NbImage* is the number of sheets you want to anamorphose.

In the *PositionCentre* list you must put the center position of all your sheets. It is organised like [x, y, z] refering to the figure.

<img width="796" alt="Capture d’écran 2021-10-18 à 11 40 58" src="https://user-images.githubusercontent.com/84194324/137764384-164a5440-43dc-4f38-8fa8-75deda7809c8.png">

#### Surface section
```
Surface : 
  a: 0
  b: 0
  c: 1
  Radius : 0.4
  Position : [3, 0, 0]
  Surface_Type : 'Cylindre'
  Wingframe : [[ 2.6,  0.,   0.2],
               [ 2.6,  0.,  -0.2],
               [ 3.,  -0.4,  0. ],
               [ 3.,   0.4,  0. ]]
```
Then you must to implement the surface properties :
- *(a, b, c)* vector represents the normal vector in case of a plane surface and the axis of a cylinder in the case of a cylinder surface case.
- `Pos=[Posx, Posy, Posz]` array represents the position relative to the camera of a point belongs to the axis of the cylinder (if the surface is a cylinder).
- *Radius* is the radius of the cylinder in the case of a cylinder.
- *SurfaceType* is a string to tell the programm that you want to anamorphose on a `Cylindre` or on a `Plan`. Only cylinder or plane surface case are implemented.
- *Wingframe* is the border of your wing represented by 4 points. It will be useful to represent your wing in 3D and print the anamorphosed speckle. It is organised as [x, y, z].

#### Output_Speckle section
```
Output_Speckle:
  heightPrintable: 27.9e-2
  widthPrintable: 21.6e-2
  PrintPath: './ImagePrintable'
```
*heightPrintable* and *widthPrintable* variables are the height and the width of the anarmophosed speckle you want to print. 
*PrintPath* is the path of the anamorphosed speckle sheets folder.

## AnamorphosePlanaire.py

Image warping on a plane with cv2.warpPerspective.
It was just a test program.

## Trucs à faire

- Améliorer temps de traitements : Parallélisation du calcul des projections de chaque feuille.
                                   Utilisation des equations analytiques pour s'affranchir de sympy.solve.
                                   Approximation numérique de la solution de la projection.
- Déplier une surface polynomiale quelconque.
- Effectuer l'anamorphose dans un espace où la caméra n'est plus le centre et orienté dans n'importe quelle direction.
