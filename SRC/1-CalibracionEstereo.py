import numpy as np
import cv2
import glob

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
bx=8
by=6 

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((bx*by,3), np.float32)
objp[:,:2] = np.mgrid[0:by,0:bx].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

images = glob.glob('*.jpg')

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (by,bx),None)

    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)

        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        imgpoints.append(corners2)
        ##TamImagen = img.shape[:2]

        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, (by,bx), corners2,ret)
        cv2.imshow('img',img)
        cv2.waitKey(500)

cv2.destroyAllWindows()

et, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
TamImagen = (640,480) #Resolucion imagen
# Las  camaras son iguales
MatrizCamaraD = mtx
MatrizCamaraI = mtx
DistorcionCamaraD = dist
DistorcionCamaraI = dist
OPTIMIZE_ALPHA = 1.0

(_, _, _, _, _, rotationMatrix, translationVector, _, _) = cv2.stereoCalibrate(
         objpoints, imgpoints, imgpoints,
         MatrizCamaraD, DistorcionCamaraD,
         MatrizCamaraI, DistorcionCamaraI,
         TamImagen, None, None, None, None,
         flags = cv2.CALIB_FIX_INTRINSIC, criteria =criteria)


translationVector = (70,0,0) ##Es siempre 70 mm por montaje

(leftRectification, rightRectification, leftProjection, rightProjection,
        dispartityToDepthMap, leftROI, rightROI) = cv2.stereoRectify(
                MatrizCamaraD, DistorcionCamaraD,
                MatrizCamaraI, DistorcionCamaraI,
                TamImagen, rotationMatrix, translationVector,
                None, None, None, None, None,
                cv2.CALIB_ZERO_DISPARITY, OPTIMIZE_ALPHA)

leftMapX, leftMapY = cv2.initUndistortRectifyMap(
        MatrizCamaraI, DistorcionCamaraI, leftRectification,
        leftProjection, TamImagen, cv2.CV_16SC2) ##cv2.CV_32FC1)

rightMapX, rightMapY = cv2.initUndistortRectifyMap(
        MatrizCamaraD, DistorcionCamaraD, rightRectification,
        rightProjection, TamImagen, cv2.CV_16SC2) ##cv2.CV_32FC1)

outputFile = 'resultadoscalibracion'
np.savez_compressed(outputFile, imageSize=TamImagen,
        leftMapX=leftMapX, leftMapY=leftMapY, leftROI=leftROI,
        rightMapX=rightMapX, rightMapY=rightMapY, rightROI=rightROI,
        MatrizCamara=MatrizCamaraD,rotationMatrix=rotationMatrix,
        Q=dispartityToDepthMap,leftRectification=leftRectification,
        leftProjection=leftProjection,rightRectification=rightRectification,
        rightProjection=rightProjection,DistorcionCamara=DistorcionCamaraD)