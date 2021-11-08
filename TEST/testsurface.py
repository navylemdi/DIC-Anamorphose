import numpy as np
import matplotlib.pyplot as plt

a = -1
b = 0
c = 1
d,e,f = 1,0,1
Pos = np.array([1,0,-1])
R=0.5
xmin = 0
ymin = -1
xmax = 1
ymax = 1
ProjVector2 = np.array([1, 0, 0])#Vecteur horizontal vers les positifs
ProjVector3 = np.array([-1, 0, 0])#Vecteur vertical vers les positifs
CylAxe = np.array([d, e, f])/np.linalg.norm(np.array([d, e, f]))
v = np.cross(CylAxe, ProjVector2)
cos = np.dot(CylAxe, ProjVector2)
kmat = np.array([[0, -v[2], v[1]], 
                 [v[2], 0, -v[0]], 
                 [-v[1], v[0], 0]])
rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * (1/(1+cos)) #Rotation entre l'axe du cylindre et l'horizontal v

Wingframe = np.array([[ Pos[0]+1, Pos[1], Pos[2]+0.2],
               [ Pos[0],  Pos[1],  Pos[2]-0.2],
               [ 1,  Pos[1]+R,  Pos[2]+0. ],
               [ 1,   Pos[1]-R,  Pos[2]+0. ]])
Wingframe2 = Wingframe.copy()
for i in range (0,4):
   Wingframe2[i,:] = np.dot(rotation_matrix, Wingframe[i,:])
def plan(a,b,c,Pos,xmin,ymin,xmax,ymax):
    stepx = (xmax-xmin)/2
    stepy = (ymax-ymin)/2
    x, y = np.meshgrid(np.arange(xmin, xmax+stepx, stepx), np.arange(ymin, ymax+stepy, stepy))
    z = (Pos[0]*a+Pos[1]*b+Pos[2]*c-b*y-a*x)/c
    return x, y, z

def cylindre(a,b,c,Pos,R,Wingframe):
    p0 = Wingframe[0,:]
    p1 = Wingframe[1,:]
    #vector in direction of axis
    v = np.array([a,b,c])
    #find magnitude of vector
    mag = np.linalg.norm(p1-p0)
    #unit vector in direction of axis
    v = v / np.linalg.norm(v)
    #make some vector not in the same direction as v
    not_v = np.array([1, 0, 0])
    if (v == not_v).all():
        not_v = np.array([0, 1, 0])
    #make vector perpendicular to v
    n1 = np.cross(v, not_v)
    #normalize n1
    n1 /= np.linalg.norm(n1)
    #make unit vector perpendicular to v and n1
    n2 = np.cross(v, n1)
    #surface ranges over t from 0 to length of axis and 0 to 2*pi
    t = np.linspace(-mag/2,mag/2, 100)
    theta = np.linspace(-np.pi/2, np.pi/2, 50)
    #use meshgrid to make 2d arrays
    t, theta = np.meshgrid(t, theta)
    #generate coordinates for surface
    x, y, z = [Pos[i] + v[i] * t + R * np.sin(theta) * n1[i] + R * np.cos(theta) * n2[i] for i in [0, 1, 2]]
    return x, y, z

def cone(Wingframe, alpha):
    zend = min(Wingframe[0,2],Wingframe[1,2])
    if Wingframe[0,2] == zend:
        v = Wingframe[0,:]
        delta = Wingframe[0,2]/Wingframe[1,2]
    if Wingframe[1,2] == zend:
        v = Wingframe[1,:]
        delta = Wingframe[1,2]/Wingframe[0,2]
    rotationy = np.array([[np.cos(2*alpha), 0, -np.sin(2*alpha)],
                     [0,                 1,              0],
                     [np.sin(2*alpha),     0,  np.cos(2*alpha)]], np.float32)
    rotationx = np.array([[1,               0,                0],
                          [0, np.cos(np.pi/2), -np.sin(np.pi/2)],
                          [0, np.sin(np.pi/2),  np.cos(np.pi/2)]], np.float32)
    rotationz = np.array([[np.cos(-2*alpha), -np.sin(-2*alpha), 0],
                          [np.sin(-2*alpha), np.cos(-2*alpha),  0],
                          [0,            0,                   1]], np.float32)
    v1 = np.dot(rotationy,v)
    v2 = np.dot(rotationx,v)
    v3 = np.dot(rotationz,v2)
    v4 = v/delta
    return v,v1,v2,v3,v4

x,y,z = plan(a,b,c,Pos,xmin,ymin,xmax,ymax)
x2,y2,z2 = cylindre(d,e,f,Pos,R,Wingframe)
v,v1,v2,v3,v4 = cone(Wingframe, np.pi/4)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(0, 0, 0, color='b')
ax.scatter(Pos[0], Pos[1], Pos[2], color='k')
ax.plot_surface(x, y, z, color='b', alpha=0.2)
ax.plot_surface(x2, y2, z2, color='r', alpha=0.2)
ax.plot([0, v[0]], [0, v[1]], [0, v[2]], color='g')
ax.plot([0, v1[0]], [0, v1[1]], [0, v1[2]], color='g')
ax.plot([0, v4[0]], [0, v4[1]], [0, v4[2]], color='r')
#ax.plot([0, v2[0]], [0, v2[1]], [0, v2[2]], color='g')
#ax.plot([0, v3[0]], [0, v3[1]], [0, v3[2]], color='g')
ax.scatter(Wingframe[:,0],Wingframe[:,1],Wingframe[:,2])
ax.scatter(Wingframe2[:,0],Wingframe2[:,1],Wingframe2[:,2])
plt.show()
