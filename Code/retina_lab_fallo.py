import time
start_time = time.time()
import numpy as np
import cv2 as cv2
from matplotlib import pyplot as plt
import pandas as pd
import os
import numpy as np
from scipy import ndimage 
from scipy.stats import moment
from scipy import sparse as sp
# parametros
n_vals = 1
tolerancia = 0.3
momentos = 10 #numero de momentos estimados
win_rows, win_cols = 10, 10 #tamanio de la ventana
os.chdir('C:\\Users\\fou-f\\Desktop\\CD2EDUARDOFOO\\Retina\\DataETL')
y = pd.read_csv('C:\\Users\\fou-f\\Desktop\\CD2EDUARDOFOO\\Retina\\Code\\metadata.csv')
y = y['y']
y = list(map(lambda x: str(x), y))[0:101]
y = np.array(y)
ojos = os.listdir()[0:101]
rows, cols = 1378, 1378       #dimension de la imagen
features = dict()
for i in ojos:
    features[str(i)] = list()
n = 0

def Momentos(canal, momentos, features, n):
    for l in range(momentos):
        features[ojos[n]].append(moment(canal.reshape(rows*cols), moment=l + 1))
        return(features)

def MomentosEspaciales(canal, momentos, features, n):
    win_mean = ndimage.uniform_filter(canal, (win_rows, win_cols))
    win_sqr_mean = ndimage.uniform_filter(canal**2, (win_rows, win_cols))
    win_var = win_sqr_mean - win_mean**2
    for l in range(momentos):
        features[ojos[n]].append(moment(canal.reshape(rows*cols), moment=l + 1))
    return(win_var, features)

def EigenFoo(canal, n_vals, features, n):
    r_s =sp.csc_matrix(canal).asfptype()
    vals = sp.linalg.eigs(r_s, k=n_vals, which='LM',return_eigenvectors=False , tol=tolerancia)
    vals = np.abs(vals)
    for l in vals:
        features[ojos[n]].append(l)
    vals = sp.linalg.eigs(r_s, k=n_vals, which='SM',return_eigenvectors=False , tol= tolerancia)
    vals = np.abs(vals)
    for l in vals:
        features[ojos[n]].append(l)
    return(features)

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
    features = Momentos(r, momentos ,features, n)
    features = EigenFoo(r, n_vals, features, n)
    mask = g< 5
    g[mask] = 255 
    g = cv2.medianBlur(g,5)
    #momentos del canal natural en green
    features = Momentos(g, momentos, features, n)
    features = EigenFoo(g, n_vals, features, n)
    mask = b< 5
    b[mask] = 255 
    b = cv2.medianBlur(b, 5)
    #momentos del canal natural en blue
    features = Momentos(b, momentos, features, n)
    features = EigenFoo(b, n_vals, features, n)

        #momentos del canal despues del filtro de correlacion espacial
    var_r, features = MomentosEspaciales(r, momentos, features, n )
    var_g, features = MomentosEspaciales(g, momentos, features, n )
    var_b, features = MomentosEspaciales(b, momentos, features, n )
    features = EigenFoo(var_r, n_vals, features, n)
    features = EigenFoo(var_g, n_vals, features, n)
    features = EigenFoo(var_b, n_vals, features, n)
    
    fig=plt.figure(figsize=(6, 12))
    fig.add_subplot(1, 4, 1)
    plt.title(str(ojos[n])+ '       danio: ' + str(y[n])+ 'numero'+ str(n))
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
############################################
for i in range(len(ojos)):
    features = Main(ojos, i, features)
dataFrame = pd.DataFrame.from_dict(features, orient='index')
from sklearn.model_selection import train_test_split
columnas= []
for i in range(dataFrame.shape[1]):
    columnas.append('X'+str((i+1)))
dataFrame.columns = columnas
from sklearn.neighbors import KNeighborsClassifier
errores_knn_class = []
for i in range(round(len(train)**.5)):
    knn = KNeighborsClassifier(n_neighbors= (i+1))
    knn.fit(train,   y_train) 
    predictions = knn.predict(test)
    prediccionesCorrectas = [predictions == y_test ] 
    accuracy= np.sum(prediccionesCorrectas) / len(predictions)
    errores_knn_class.append(accuracy)
plt.plot(errores_knn_class, c='#5C4684') #el color pantone 2018
errores_knn_class = np.array(errores_knn_class)
cotaSuperior = 15
errores_knn_class < cotaSuperior
plt.plot(errores_knn_class[errores_knn_class < cotaSuperior] , c='#5C4684')
from sklearn.ensemble import RandomForestClassifier
acc_RF = []
for i in range(300):
    trees = i + 1 
    rf = RandomForestClassifier(n_estimators=trees) # inicializamos el clasificador
    rf.fit(train, y_train) #otra linea que nos libra de cuentas
    predictions = rf.predict(test) # otra linea que se agradece
    prediccionesCorrectas=[predictions==y_test] #de nuevo el error consiste en solo contar :D
    accuracy= np.sum(prediccionesCorrectas) / len(predictions)
    acc_RF.append(accuracy)
    print('Presición con Random forest: ' + str(accuracy) + ' con ' + str(trees) +' árboles')
plt.plot(acc_RF, c='#5C4684') #el color pantone 2018
acc_RF = np.array(acc_RF)
print("--- %s seconds ---" % (time.time() - start_time))      