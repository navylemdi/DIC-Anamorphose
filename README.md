# AnamorphosePlane

## Detection de contour multi-feuille

Telecharger main.py, Fonction.py et Feuille.py

Modifier la partie Constantes avec la position de chacune des feuilles. 

Modifier la variable saut pour choisir le nombre de point sautés (saut=1 <=> Tout les points sont pris en compte, saut=10 <=> 1/10 des points sont pris en compte). Cela permet de limiter le temps de calcul (~30 minutes pour une feuille complète de 4mm/ ~75 minutes pour une feuille complète de 2mm). 
Seul les cas plan et cylindre sont implémentés. 

## AnamorphosePlanaire.py

Déformation de l'image sur un plan grace à cv2.warpPerspective

## Trucs à faire

- Améliorer temps de traitements : Parallélisation du calcul des projections de chaque feuille.
                                   Utilisation des equations analytiques pour s'affranchir de sympy.solve.
                                   Approximation numérique de la solution de la projection.
- Déplier une surface polynomiale quelconque
