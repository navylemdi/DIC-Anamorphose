# Anamorphosis

## Detection of edges and deformations on a surface of several sheets of speckles

<p align="center">
  <img width="700" alt="Capture d’écran 2021-11-10 à 09 02 57" src="https://user-images.githubusercontent.com/84194324/141127089-da3f65a4-66ff-4667-b41b-1c5d69b1e156.png">
</p>

### Installation
To begin, you have to download some packages listed in the requirements.txt file
To use main.py, you also have to download Module folder.

### deck.yaml initialisation

Main structure for `deck.yaml` file:

```
Camera:
  Focal_length: 50 #mm
  Sensor_height: 29 #mm

Input_speckle: 
  Step: 500
  Begining: 3
  Height: 27e-2
  Width: 21e-2
  Path: './Banque_Speckle/2mm'
  Generic_name: 'Speckle_'
  NbImage: 1
  Position_centre: [[2, 0, 0], [2, -21e-2, 0], [2, 21e-2, 0]]

Surface : 
  a: 0
  b: 0
  c: 1
  Radius : 0.4
  Position : [3, 0, 0]
  Surface_type : 'Cylindre'
  Wingframe : [[ 2.6,  0.,   .5],
               [ 2.6,  0.,  -.5],
               [ 3.,  -0.4,  0. ],
               [ 3.,   0.4,  0. ]]

Output_speckle:
  Height_printable: 27.9e-2
  Width_printable: 21.6e-2
  Print_path: './ImagePrintable'
```
#### Camera section
```
Camera:
  Focal_length: 50 #mm
  Sensor_height: 29 #mm
 ```
 Enter your focal length and sensor height in mm in these variable. Field of view in degrees and radians will be calculated and printed out.
 It is also used to plot the field of view cone in the 3D viewer.
#### Input_Speckle section
```
Input_speckle: 
  Step: 500
  Begining: 3
  Height: 27e-2
  Width: 21e-2
  Path: './Banque_Speckle/2mm'
  Generic_name: 'Speckle_'
  NbImage: 1
  Position_centre: [[2, 0, 0], [2, -21e-2, 0], [2, 21e-2, 0]]
```
Due to the multitude of points in the speckle, to test the code, we only anamorphose a few points. The variable *Step* represents the anamorphic step. `step=2` will anamorphose half of all the points of a sheet of speckles.

*Begining* variable represents the first index of the contour list calculated by OpenCV to be considered ine the anamorphosis. Usually `begining=3` is often sufficient to avoid black filling of the result by the algorithm.

You need to implement the size of the speckle sheets you want to anamorphose and the size of the anamorphosed speckle sheets you want to get. Use the *Height*, *Width*. The unit is meter.

*Path* is the path of the classic speckle sheets folder.

*Generic_name* is the name of your speckle sheets. It as to be the shape of *Generic_Name_XX* with XX a number.

*NbImage* is the number of sheets you want to anamorphose.

In the *Position_centre* list you must put the center position of all your sheets. It is organised like [x, y, z] refering to the figure.

<p align="center">
  <img width="628" alt="Capture d’écran 2021-11-09 à 20 24 42" src="https://user-images.githubusercontent.com/84194324/141032568-872ec514-2716-4acb-a321-eb7dfd5d4731.png">
</p>

#### Surface section
```
Surface : 
  a: 0
  b: 0
  c: 1
  Radius : 0.4
  Position : [3, 0, 0]
  Surface_type : 'Cylindre'
  Wingframe : [[ 2.6,  0.,   .5],
               [ 2.6,  0.,  -.5],
               [ 3.,  -0.4,  0. ],
               [ 3.,   0.4,  0. ]]
```
Then you must to implement the surface properties :
- *(a, b, c)* vector represents the normal vector in case of a plane surface and the axis of a cylinder in the case of a cylinder surface case.
- `Position=[Posx, Posy, Posz]` array represents the position relative to the camera of a point belongs to the axis of the cylinder (if the surface is a cylinder).
- *Radius* is the radius of the cylinder in the case of a cylinder.
- *Surface_type* is a string to tell the programm that you want to anamorphose on a `Cylindre` or on a `Plan`. Only cylinder or plane surface case are implemented.
- *Wingframe* is the border of your wing represented by 4 points. It will be useful to represent your wing in 3D and print the anamorphosed speckle. It is organised as [x, y, z].

#### Output_Speckle section
```
Output_speckle:
  Height_printable: 27.9e-2
  Width_printable: 21.6e-2
  Print_path: './ImagePrintable'
```
*Height_printable* and *Width_printable* variables are the height and the width of the anarmophosed speckle you want to print. 
*Print_path* is the path of the anamorphosed speckle sheets folder.

## AnamorphosePlanaire.py

Image warping on a plane with cv2.warpPerspective.
It was just a test program.

## Trucs à faire

- Améliorer temps de traitements : Parallélisation du calcul des projections de chaque feuille.
                                   Utilisation des equations analytiques pour s'affranchir de sympy.solve.
                                   Approximation numérique de la solution de la projection.
- Déplier une surface polynomiale quelconque.
- Effectuer l'anamorphose dans un espace où la caméra n'est plus le centre et orienté dans n'importe quelle direction.
