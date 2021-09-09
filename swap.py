import numpy as np
import cv2 
from matplotlib import pyplot as plt
import math


def convolve(shiftedImg):
    #15 10
    cx, cy = shiftedImg.shape[0]/2, shiftedImg.shape[1]/2
    x = np.linspace(0, shiftedImg.shape[0], shiftedImg.shape[0])
    sigmax, sigmay =(cx-1)/2,(cy-1)/2
    # int(math.sqrt(shiftedImg.shape[0])),int(math.sqrt(shiftedImg.shape[0]))
    y = np.linspace(0, shiftedImg.shape[1], shiftedImg.shape[1])
    X, Y = np.meshgrid(x, y)
    mask = np.exp(-(((X-cx)/sigmax)**2 + ((Y-cy)/sigmay)**2))
    mask= (mask - np.min(mask))/np.ptp(mask).astype('float64')

    conv=shiftedImg * mask
    ifft=np.fft.ifftshift(conv)
    ishift=np.fft.ifft2(ifft)
    magnum = np.abs(ishift)
    return magnum


def laplacianPy(A, B, m, num_levels = 6):
    # generate Gaussian pyramid for A,B and mask
    GA = A.copy()
    GB = B.copy()
    GM = m.copy()
    GA= (GA - np.min(GA))/np.ptp(GA).astype('float64')
    GB= (GB - np.min(GB))/np.ptp(GB).astype('float64')
    gpA = [GA]
    gpB = [GB]
    gpM = [GM]

    for i in range(num_levels):


        fftA = np.fft.fft2(GA)
        fshiftA = np.fft.fftshift(fftA)
        GA=convolve(fshiftA)

        GA= (GA - np.min(GA))/np.ptp(GA).astype('float64')
        GA=GA[::2,::2]

        fftB = np.fft.fft2(GB)
        fshiftB = np.fft.fftshift(fftB)
        GB=convolve(fshiftB)
        GB= (GB - np.min(GB))/np.ptp(GB).astype('float64')
        GB=GB[::2,::2]
        
        fftM = np.fft.fft2(GM)
        fshiftM = np.fft.fftshift(fftM)
        GM=convolve(fshiftM)
        GM= (GM - np.min(GM))/np.ptp(GM).astype('float64')
        GM=GM[::2,::2]
  
        gpA.append(np.float64(GA))
        gpB.append(np.float64(GB))
        gpM.append(np.float64(GM))
    # generate Laplacian Pyramids for A,B and masks
    # the bottom of the Lap-pyr holds the last (smallest) Gauss level


    lpA = [gpA[num_levels-1]]
    lpB = [gpB[num_levels-1]]
    gpMr = [gpM[num_levels-1]]

    for i in range(num_levels-1,0,-1):
        iShape=np.zeros_like(gpA[i-1])
        iShape[0::2,0::2]=gpA[i]

        fftA = np.fft.fft2(iShape)
        fshiftA = np.fft.fftshift(fftA)
        iShape=convolve(fshiftA)

        iShape = ((iShape - np.min(iShape)))/np.ptp(iShape).astype('float64')

        iShapeB=np.zeros_like(gpB[i-1])
        iShapeB[0::2,0::2]=gpB[i]
        fftB = np.fft.fft2(iShapeB)
        fshiftB = np.fft.fftshift(fftB)
        iShapeB=convolve(fshiftB)
        iShapeB = (iShapeB - np.min(iShapeB))/np.ptp(iShapeB).astype('float64')


        LA = np.subtract(gpA[i-1], iShape)
        LB = np.subtract(gpB[i-1],iShapeB)

        lpA.append(LA)
        lpB.append(LB)
        gpMr.append(gpM[i-1]) # also reverse the masks

    LS = []

    for la,lb,gm in zip(lpA,lpB,gpMr):
        ls = la * gm + lb * (1.0 - gm)
        ls.dtype = np.float64
        LS.append(ls)
        # now reconstruct

    ls_ = LS[0]
    ls_.dtype = np.float64

    for i in range(1,num_levels):
        iShape=np.zeros_like(LS[i])
        iShape[::2,::2]=ls_

        fftA = np.fft.fft2(iShape)
        fshiftA = np.fft.fftshift(fftA)

        iShape=convolve(fshiftA)
        iShape = ((iShape - np.min(iShape)))/np.ptp(iShape).astype('float64') 

        ls_=iShape +LS[i]

    return ls_

if __name__ == "__main__":
    #opening images 
    me = cv2.imread("me.png",0)
    amani= cv2.imread("amani.png",0)
    mask = cv2.imread("mask.png",0)
    #normalize 
    mask = (mask - np.min(mask))/np.ptp(mask).astype('float64')

    laplac=laplacianPy(me,amani,mask,5)
    plt.subplot(1,3,1)
    plt.imshow(amani, 'gray')
    plt.title('First')
    plt.axis('off')

    plt.subplot(1,3,2)
    plt.imshow(me,'gray')
    plt.title('Second')
    plt.axis('off')

    plt.subplot(1,3,3)
    plt.imshow(laplac,'gray')
    plt.title('Result')
    plt.axis('off')


    plt.tight_layout()
    plt.show() 
