%matplotlib inline
import numpy as np
import cv2 as cv2
from matplotlib import pyplot as plt
import pandas as pd
import os
import numpy as np
from scipy import ndimage 
from scipy.stats import moment
momentos = 10 #numero de momentos estimados
win_rows, win_cols = 10, 10 #tamanio de la ventana
os.chdir('C:\\Users\\fou-f\\Desktop\\CD2EDUARDOFOO\\Retina\\DataETL')
y = pd.read_csv('C:\\Users\\fou-f\\Desktop\\CD2EDUARDOFOO\\Retina\\Code\\metadata.csv')
ojos = os.listdir()[0:20]
rows, cols = 1378, 1378       #dimension de la imagen
features = dict()
for i in ojos:
    features[str(i)] = list()
n = 0

def Momentos(canal, momentos, features):
    for l in range(momentos):
        features[ojos[n]].append(moment(canal.reshape(rows*cols), moment=l + 1))
        return(features)

def MomentosEspaciales(canal, momentos, features):
    win_mean = ndimage.uniform_filter(canal, (win_rows, win_cols))
    win_sqr_mean = ndimage.uniform_filter(canal**2, (win_rows, win_cols))
    win_var = win_sqr_mean - win_mean**2
    for l in range(momentos):
        features[ojos[n]].append(moment(canal.reshape(rows*cols), moment=l + 1))
    return(win_var, features)
    

def Main(ojos, n, features):
    ojo = cv2.imread(ojos[n])
    # division en canales 
    b,g,r = cv2.split(ojo)       # get b,g,r
    # eliminacion de pixeles negros
    mask = r< 5
    r[mask] = 255 
    r = cv2.medianBlur(r, 5)
    #calculo de momentos
    #momentos del canal natural en r
    features = Momentos(r, momentos ,features)
    mask = g< 5
    g[mask] = 255 
    g = cv2.medianBlur(g,5)
    #momentos del canal natural en green
    features = Momentos(g, momentos, features)
    mask = b< 5
    b[mask] = 255 
    b = cv2.medianBlur(b, 5)
    #momentos del canal natural en blue
    features = Momentos(b, momentos, features)
        #momentos del canal despues del filtro de correlacion espacial
    var_r, features = MomentosEspaciales(r, momentos, features )
    var_g, features = MomentosEspaciales(g, momentos, features )
    var_b, features = MomentosEspaciales(b, momentos, features )

    fig=plt.figure(figsize=(6, 12))
    fig.add_subplot(1, 4, 1)
    plt.title(str(ojos[n])+ '       danio: ' + str(y[ 'y'][n]))
    plt.imshow(var_r)
    plt.xticks([]), plt.yticks([])   # to hide tick values on X and Y axis
    plt.xlabel('')
    fig.add_subplot(1, 4, 2)
    plt.imshow(var_g)
    plt.xticks([]), plt.yticks([])   # to hide tick values on X and Y axis
    plt.xlabel('')
    fig.add_subplot(1, 4, 3)
    plt.imshow(var_b)
    plt.xticks([]), plt.yticks([])   # to hide tick values on X and Y axis
    plt.xlabel('')
    ojo = cv2.merge([r,g,b])
    fig.add_subplot(1, 4, 4)
    plt.imshow(ojo)
    plt.xticks([]), plt.yticks([])   # to hide tick values on X and Y axis
    plt.show()
    return(features)
##############################
ojo_gris = cv2.cvtColor(ojo, cv2.COLOR_BGR2GRAY)
laplacian = cv2.Laplacian(ojo_gris,cv2.CV_64F)
sobelx = cv2.Sobel(ojo_gris,cv2.CV_64F,1,0,ksize=3)
sobely = cv2.Sobel(ojo_gris,cv2.CV_64F,0,1,ksize=3)

plt.subplot(2,2,1),plt.imshow(ojo,cmap = 'gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,2),plt.imshow(laplacian,cmap = 'gray')
plt.title('Laplacian'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,3),plt.imshow(sobelx,cmap = 'gray')
plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,4),plt.imshow(sobely,cmap = 'gray')
plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])
plt.show()
################

edges = cv2.Canny(ojo,50,200)

plt.subplot(121),plt.imshow(ojo,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(edges,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

plt.show()

#####################
ojo = cv2.imread(ojos[0])
b,g,r = cv2.split(ojo)       # get b,g,r
# eliminacion de pixeles negros
mask = r< 5
r[mask] = 255 
r = cv2.medianBlur(r, 5)
g[mask] = 255 
g = cv2.medianBlur(g,5)
mask = b< 5
b[mask] = 255 
b = cv2.medianBlur(b, 5)
ojo = cv2.merge([r,g,b])
plt.imshow(ojo)
##########################
import scipy as sp
r_s =sp.sparse.csc_matrix(r).asfptype()
vals = sp.sparse.linalg.eigs(r_s, k=10, which='LM',return_eigenvectors=False )
vals = np.abs(vals)
print(vals)
vals = sp.sparse.linalg.eigs(r_s, k=10, which='SM',return_eigenvectors=False )
vals = np.abs(vals)
print(vals)
##############################
help(sp.sparse)
help(sp.sparse.linalg.eigs)