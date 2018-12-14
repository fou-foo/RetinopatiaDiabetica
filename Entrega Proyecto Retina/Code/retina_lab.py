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

def MomentosDerivadas(imagen, momentos, features, n):
    imagen_gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(imagen_gris, 13)
    features = Momentos(laplacian, momentos, features, n)
    sobelx = cv2.Sobel(imagen_gris, cv2.CV_64F, 1, 0, ksize=31)
    features = Momentos(sobelx, momentos, features, n)
    sobely = cv2.Sobel(imagen_gris, cv2.CV_64F, 0, 1, ksize=31)
    features = Momentos(sobely, momentos, features, n)
    edges = cv2.Canny(imagen,10,50)
    features = Momentos(edges, momentos, features, n)
 
    
    fig=plt.figure(figsize=(6, 12))
    fig.add_subplot(1, 5, 1)
    plt.title(str(ojos[n])+ '       danio: ' + str(y[n])+ 'numero'+ str(n))
    plt.imshow(laplacian)
    plt.xticks([]), plt.yticks([])   # to hide tick values on X and Y axis
    plt.xlabel('')
    fig.add_subplot(1, 5, 2)
    plt.imshow(sobelx)
    plt.xticks([]), plt.yticks([])   # to hide tick values on X and Y axis
    plt.xlabel('')
    fig.add_subplot(1, 5, 3)
    plt.imshow(sobely)
    plt.xticks([]), plt.yticks([])   # to hide tick values on X and Y axis
    plt.xlabel('')
    fig.add_subplot(1, 5, 4)
    plt.imshow(edges)
    plt.xticks([]), plt.yticks([])   # to hide tick values on X and Y axis
    fig.add_subplot(1, 5, 5)
    plt.imshow(imagen)
    plt.xticks([]), plt.yticks([])   # to hide tick values on X and Y axis
    plt.show()
    return(features)



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

def MainRGB(ojos, n, features):
    print(i)
    ojo = cv2.imread(ojos[n])
     # filtros direccionales
    features = MomentosDerivadas(ojo, momentos, features, n)
    ## TRABAJO EN ESPACIO RGB
    # division en canales 
    b,g,r = cv2.split(ojo)       # get b,g,r
    # eliminacion de pixeles negros
    mask = r< 5
    r[mask] = 255 
    r = cv2.medianBlur(r, 5)
    #calculo de momentos
    #momentos del canal natural en r
    features = Momentos(r, momentos ,features, n)
    #features = EigenFoo(r, n_vals, features, n)
    #print('first eiegen')
    mask = g< 5
    g[mask] = 255 
    g = cv2.medianBlur(g,5)
    #momentos del canal natural en green
    features = Momentos(g, momentos, features, n)
    #features = EigenFoo(g, n_vals, features, n)
    #print('2 eiegen')
    mask = b< 5
    b[mask] = 255 
    b = cv2.medianBlur(b, 5)
    #momentos del canal natural en blue
    features = Momentos(b, momentos, features, n)
    #features = EigenFoo(b, n_vals, features, n)
    #print('3 eiegen')
        #momentos del canal despues del filtro de correlacion espacial
    var_r, features = MomentosEspaciales(r, momentos, features, n )
    var_g, features = MomentosEspaciales(g, momentos, features, n )
    var_b, features = MomentosEspaciales(b, momentos, features, n )
    #features = EigenFoo(var_r, n_vals, features, n)
    #print('4 eiegen')
    #features = EigenFoo(var_g, n_vals, features, n)
    #print('5 eiegen')
    #features = EigenFoo(var_b, n_vals, features, n)
    #print('6 eiegen')
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

def MainHSV(ojos, n, features):
    print(i)
    ojo = cv2.imread(ojos[n])
    hsv = cv2.cvtColor(ojo, cv2.COLOR_BGR2HSV)
    h,s,v = cv2.split(hsv)

     # filtros direccionales
    features = MomentosDerivadas(hsv, momentos, features, n)
    ## TRABAJO EN ESPACIO HSV
    # division en canales 
    #calculo de momentos
    #momentos del canal natural en r
    features = Momentos(h, momentos ,features, n)
    features = Momentos(s, momentos, features, n)
    features = Momentos(b, momentos, features, n)
    var_r, features = MomentosEspaciales(h, momentos, features, n )
    var_g, features = MomentosEspaciales(s, momentos, features, n )
    var_b, features = MomentosEspaciales(v, momentos, features, n )
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

def MainPCA(ojos, n, features):
    print(i)
    ojo = cv2.imread(ojos[n])
    b,g,r = cv2.split(ojo)
    r_v = r.reshape((1378**2,1))
    g_v = g.reshape((1378**2,1))
    b_v = b.reshape((1378**2,1))
    data = np.hstack([r_v, g_v, b_v])
    pca = PCA(n_components=3, ).fit((data))
    for i in pca.explained_variance_:
        features[ojos[n]].append(i)
    numeros = pca.transform(data)
    rota = (numeros - np.min(numeros))/(np.max(numeros)-np.min(numeros))
    rota = np.round(rota)*255
    r_v = rota[:, 0].reshape((1378, 1378))
    g_v = rota[:, 1].reshape((1378, 1378))
    b_v = rota[:, 2].reshape((1378, 1378))
    fig=plt.figure(figsize=(6, 12))
    fig.add_subplot(1, 4, 1)
    plt.title(str(ojos[n])+ '       danio: ' + str(y[n])+ 'numero'+ str(n))
    plt.imshow(r_v, cmap='gray')
    plt.xticks([]), plt.yticks([])   # to hide tick values on X and Y axis
    plt.xlabel('')
    fig.add_subplot(1, 4, 2)
    plt.imshow(g_v, cmap='gray')
    plt.xticks([]), plt.yticks([])   # to hide tick values on X and Y axis
    plt.xlabel('')
    fig.add_subplot(1, 4, 3)
    plt.imshow(b_v, cmap='gray')
    plt.xticks([]), plt.yticks([])   # to hide tick values on X and Y axis
    plt.xlabel('')
    ojo = cv2.merge([r,g,b])
    fig.add_subplot(1, 4, 4)
    plt.imshow(ojo)
    plt.xticks([]), plt.yticks([])   # to hide tick values on X and Y axis
    plt.show()
     # filtros direccionales
    features = MomentosDerivadas(ojo, momentos, features, n)
    ## TRABAJO EN ESPACIO PCA
    features = Momentos(r_v, momentos ,features, n)
    features = Momentos(g_v, momentos, features, n)
    features = Momentos(b_v, momentos, features, n)
    var_r, features = MomentosEspaciales(r_v, momentos, features, n )
    var_g, features = MomentosEspaciales(g_v, momentos, features, n )
    var_b, features = MomentosEspaciales(b_v, momentos, features, n )
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
