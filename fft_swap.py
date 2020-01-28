#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 17:19:39 2020

@author: liuxiaomeng
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt

img_1=cv2.imread('apple.jpeg',0)
img_2=cv2.imread('orange.jpeg',0)

plt.subplot(121),plt.imshow(img_1, cmap = 'gray')
plt.title('Input Image_1'), plt.xticks([]), plt.yticks([])

plt.subplot(122),plt.imshow(img_2, cmap = 'gray')
plt.title('Input Image_2'), plt.xticks([]), plt.yticks([])
plt.show()
# fft on image_1
dft_1 = cv2.dft(np.float32(img_1),flags = cv2.DFT_COMPLEX_OUTPUT)
dft_shift_1 = dft_1
magnitude_spectrum_1 = cv2.magnitude(dft_shift_1[:,:,0],dft_shift_1[:,:,1])
dft_complex_1 = dft_shift_1[:,:,0] + 1j*dft_shift_1[:,:,1]
phase_1=np.angle(dft_complex_1) 



#plt.subplot(132),plt.imshow(magnitude_spectrum_1, cmap = 'gray')
#plt.title('Magnitude Spectrum_1'), plt.xticks([]), plt.yticks([])
#plt.subplot(133),plt.imshow(phase_1,cmap='gray')
#plt.title('Phase_1'),plt.xticks([]),plt.yticks([])


#fft on image_2
dft_2 = cv2.dft(np.float32(img_2),flags = cv2.DFT_COMPLEX_OUTPUT)
dft_shift_2 = dft_2
magnitude_spectrum_2 = cv2.magnitude(dft_shift_2[:,:,0],dft_shift_2[:,:,1])
dft_complex_2 = dft_shift_2[:,:,0] + 1j*dft_shift_2[:,:,1]
phase_2=np.angle(dft_complex_2)


#plt.subplot(132),plt.imshow(img_2, cmap = 'gray')
#plt.title('Magnitude Image_2'), plt.xticks([]), plt.yticks([])
#plt.subplot(133),plt.imshow(phase_2,cmap='gray')
#plt.title('Phase_2'),plt.xticks([]),plt.yticks([])


#swap phase
new_complex_1=np.zeros([300,300,2],dtype=float)
new_complex_2=np.zeros([300,300,2],dtype=float)

new_phase_1=phase_2
new_phase_2=phase_1

#new_complex_1=magnitude_spectrum_1*(np.cos(new_phase_2+1j*np.sin(new_phase_2)))
new_complex_1[:,:,0]=magnitude_spectrum_1*np.cos(new_phase_1)
new_complex_1[:,:,1]=magnitude_spectrum_1*np.sin(new_phase_1)

#new_complex_2=magnitude_spectrum_2*(np.cos(new_phase_1+1j*np.sin(new_phase_1)))
new_complex_2[:,:,0]=magnitude_spectrum_2*np.cos(new_phase_2)
new_complex_2[:,:,1]=magnitude_spectrum_2*np.sin(phase_1)

f_ishift_1 = new_complex_1
img_back_1 = cv2.idft(f_ishift_1)
img_back_1 = cv2.magnitude(img_back_1[:,:,0],img_back_1[:,:,1])

f_ishift_2 = new_complex_2
img_back_2 = cv2.idft(f_ishift_2)
img_back_2 = cv2.magnitude(img_back_2[:,:,0],img_back_2[:,:,1])

plt.subplot(121),plt.imshow(img_back_1, cmap = 'gray')
plt.title('Output Image_1'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(img_back_2, cmap = 'gray')
plt.title('output image_2'), plt.xticks([]), plt.yticks([])

