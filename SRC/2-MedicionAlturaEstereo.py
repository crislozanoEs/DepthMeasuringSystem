import numpy as np
import cv2
import os
import time

#Leer el archivo de calibracion
calibrationFile='resultadoscalibracion.npz'   #Archivo de calibracion

#Obtener todos los datos en el archivo de calibracion
calibration = np.load(calibrationFile)
imageSize = tuple(calibration["imageSize"])
leftMapX = calibration["leftMapX"]
leftMapY = calibration["leftMapY"]
leftROI = tuple(calibration["leftROI"])
rightMapX = calibration["rightMapX"]
rightMapY = calibration["rightMapY"]
rightROI = tuple(calibration["rightROI"])
matrixCamera = calibration["MatrizCamara"]
rotationMatrix = tuple(calibration["rotationMatrix"])
Q = calibration["Q"]

fx = matrixCamera[0,0]        # Destancia focal
baseline = 70     # Distancia entra ambas camaras [mm]
centrox = 320     #Coordenada en eje X del pixel de interes
centroy = 2*240 #Coordenada en eje Y del pixel de interes

#Parametros para disparidad
window_size = 3                   # wsize default 3; 5; 7 for SGBM reduced size image; 15 for SGBM full size image (1300px and above); 5 Works nicely
min_disp = 0 
num_disp = 2*16 
pre_filter = 63 
uniqueness_ratio = 15
speckle_windows_size = 0 
speckle_range = 2 
disp_12_mx = 1 
left_matcher = cv2.StereoSGBM_create(
        minDisparity=min_disp,
        numDisparities=num_disp,             # max_disp has to be dividable by 16 f. E. HH 192, 256
        blockSize=5,
        P1=8 * 5 * window_size ** 2,    # wsize default 3; 5; 7 for SGBM reduced size image; 15 for SGBM full size image (1300px and above); 5 Works nicely
        P2=32 * 5 * window_size ** 2,
        disp12MaxDiff=disp_12_mx,
        uniquenessRatio=uniqueness_ratio,
        speckleWindowSize=speckle_windows_size,
        speckleRange=speckle_range,
        preFilterCap=pre_filter,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
        )
#Parametros iniciales
cantidadImagenes = 30   #Cantidad de imagenes a tomar por medida
sumDisparidad = 0      #Suma de disparidad de adaptacion
alturaInicial = 0      #Altura que el usuario midio
puntos = 0      #Mediciones de disparidad donde la disparidad es diferente de 0

resultadosFile =  'resultadosadaptacion.npz'

#Proceso de adaptacion
if (os.path.isfile(archivo)==False):
    for i in range(cantidadImagenes):
        alturaInicial = input("Digite la altura inicial medida en cm")
        cam = cv2.VideoCapture(0) #El id de la camara ls -l /dev/video*
        ret, imgL = cam.read() #Tomar fotonombre1 = 'Ii'+str(i)+'.jpg'
        cam.release() ##Soltar la camara
        cam1 = cv2.VideoCapture(1)
        ret1, imgR = cam1.read()
        cam1.release() ##Soltar la camara
        imgL= cv2.imread(imgL,cv2.IMREAD_GRAYSCALE)
        imgR= cv2.imread(imgR,cv2.IMREAD_GRAYSCALE)
        right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)
        # Parametros al filtro
        lmbda = 80000
        sigma = 1.2
        visual_multiplier = 0.5 ##es 1
        wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
        wls_filter.setLambda(lmbda)
        wls_filter.setSigmaColor(sigma)
        print('Computing disparity...')
        displ = left_matcher.compute(imgL, imgR)  # .astype(np.float32)/16
        dispr = right_matcher.compute(imgR, imgL)  # .astype(np.float32)/16
        displ = np.int16(displ)
        dispr = np.int16(dispr)
        filteredImg = wls_filter.filter(displ, imgL, None, dispr)  # important to put "imgL" here!!!
        filteredImg = cv2.normalize(src=filteredImg, dst=filteredImg, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX);
        filteredImg = np.uint8(filteredImg)
        if (filteredImg[centrox,centroy]!=0):
            disparidad = (fx * baseline) / (units * filteredImg[centrox,centroy])
            print(disparidad)
            puntos = puntos + 1
            sumDisparidad = sumDisparidad+disparidad
    print("ADAPTACION FINALIZADO")
    disparidadInicial =  sumDisparidad/puntos
    np.savez_compressed(resultadosFile, disparidadInicial=disparidadInicial, alturaInicial=alturaInicial)
else:
    adaptacion = np.load(resultadosFile)
    disparidadInicial = adaptacion["disparidadInicial"]
    alturaInicial = adaptacion["alturaInicial"]

print(disparidadInicial)
#Parametros para medicion
sumDisparidad = 0
cantidadMedidas = 1
tiempoEspera=10*60
salidaFile = "salida.txt"
f = open (salidaFile,'a')
while(1):
    time.sleep(tiempoEspera)
    sumDisparidad = 0
    cantidadPuntosD = 0
    cantidadImagenes = 30
    print("MEDIDA %d",cantidadMedidas)
    for i in range(cantidadImagenes):
        cam = cv2.VideoCapture(0) #El id de la camara ls -l /dev/video*
        ret, imgL = cam.read() #Tomar fotonombre1 = 'Ii'+str(i)+'.jpg'
        cam.release() ##Soltar la camara
        cam1 = cv2.VideoCapture(1)
        ret1, imgR = cam1.read()
        cam1.release() ##Soltar la camara
        imgL= cv2.imread(imgL,cv2.IMREAD_GRAYSCALE)
        imgR= cv2.imread(imgR,cv2.IMREAD_GRAYSCALE)
        right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)
        # FILTER Parameters
        lmbda = 80000
        sigma = 1.2
        visual_multiplier = 0.5 ##es 1
        wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
        wls_filter.setLambda(lmbda)
        wls_filter.setSigmaColor(sigma)
        print('Computing disparity...')
        displ = left_matcher.compute(imgL, imgR)  # .astype(np.float32)/16
        dispr = right_matcher.compute(imgR, imgL)  # .astype(np.float32)/16
        displ = np.int16(displ)
        dispr = np.int16(dispr)
        filteredImg = wls_filter.filter(displ, imgL, None, dispr)  # important to put "imgL" here!!!
        filteredImg = cv2.normalize(src=filteredImg, dst=filteredImg, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX);
        filteredImg = np.uint8(filteredImg)
        if (filteredImg[centrox][centroy]!=0):
            disparidadE = baseline*fx/filteredImg[centrox][centroy]
            #Correcion
            disparidadCorregida = disparidadInicial*alturaInicial/disparidadE
            sumDisparidad = sumDisparidad+disparidadCorregida
            #Suma puntos verdaderos
            cantidadPuntosD = cantidadPuntosD+1
        print("Disparidad")
        print(disparidadE)
        print(disparidadCorregida)
    cantidadMedidas = cantidadMedidas+1
    disparidadPromedio = sumDisparidad/cantidadPuntosD
    #Escribir en el archivo de salida
    cadena = 'Altura %d: %.2f cm \n'%(cantidadMedidas, disparidadPromedio)
    print(cadena)
    f.write(cadena)
    