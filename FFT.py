import cv2
import numpy as np
from matplotlib import pyplot as plt

# read in a image
img = cv2.imread('elephant.jpeg',0)
rows,cols=img.shape
crow = rows//2    #center
ccol = cols//2

###apply FFT through numpy
#f = np.fft.fft2(img)
#fshift=np.fft.fftshift(f)
#magnitude_spectrum=20*np.log(np.abs(fshift))
#phase=np.angle(fshift)

#apply FFT through OpenCV
dft = cv2.dft(np.float32(img),flags = cv2.DFT_COMPLEX_OUTPUT)  #type=float32 mxnx2
dft_shift = np.fft.fftshift(dft)                               #type=float32 mxnx2 
magnitude_spectrum = 20*np.log(cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1]))  #type=float32 mxn
dft_complex = dft_shift[:,:,0] + 1j*dft_shift[:,:,1]    #type=complex64  mxn
phase=np.angle(dft_complex)  # this function take 1-D complex matrix as input

## plot Fourier Transform (magnitude and phase) 
#plt.subplot(131),plt.imshow(img, cmap = 'gray')
#plt.title('Input Image'), plt.xticks([]), plt.yticks([])
#plt.subplot(132),plt.imshow(magnitude_spectrum, cmap = 'gray')
#plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
#plt.subplot(133),plt.imshow(phase,cmap='gray')
#plt.title('Phase'),plt.xticks([]),plt.yticks([])


## applie low-pass filter
## create a mask first, center square is 1, remaining all zeros
#mask = np.zeros((rows, cols, 2), np.uint8)  #generate a rowsxcolsx2 matrix, data type uint8 
#mask[crow-30:crow+30, ccol-30:ccol+30,:] = 1
## apply mask and inverse DFT
#fshift = dft_shift*mask
#f_ishift = np.fft.ifftshift(fshift)
#img_back = cv2.idft(f_ishift)
#img_back = cv2.magnitude(img_back[:,:,0],img_back[:,:,1])

## apply high-pass filter
#mask=np.ones((rows, cols, 2), np.uint8)
#mask[crow-10:crow+10, ccol-10:ccol+10,:] = 0
#fshift = dft_shift*mask
#f_ishift = np.fft.ifftshift(fshift)
#img_back = cv2.idft(f_ishift)
#img_back = cv2.magnitude(img_back[:,:,0],img_back[:,:,1])

## apply band-pass filter
#mask=np.zeros((rows, cols, 2), np.uint8)
#mask[crow-10:crow+10, :,:] = 1
#fshift = dft_shift*mask
#f_ishift = np.fft.ifftshift(fshift)
#img_back = cv2.idft(f_ishift)
#img_back = cv2.magnitude(img_back[:,:,0],img_back[:,:,1])

# apply band-pass filter
mask=np.zeros((rows, cols, 2), np.uint8)
mask[:, ccol-10:ccol+10,:] = 1  # type=uint8
fshift = dft_shift*mask         # type=float32  mxnx2
f_ishift = np.fft.ifftshift(fshift)   # type=float32  mxnx2
img_back = cv2.idft(f_ishift)  #
img_back = cv2.magnitude(img_back[:,:,0],img_back[:,:,1])  # type=float32 mxnx2
cv2.imwrite('new_elephant.jpg',img_back)

plt.subplot(121),plt.imshow(img, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(img_back, cmap = 'gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])

plt.show()
plt.show()