import cv2
import numpy as np
from matplotlib import pyplot as plt

Orig_Mia=cv2.imread('Mia2.jpg',0)
Orig_Milin=cv2.imread('Milin.jpg',0)
rows,cols=Orig_Mia.shape
crow = rows//2    #center
ccol = cols//2
dimension=(rows,cols)
ratio=cols/rows   #<1

new_dimension=(cols,rows)
input_Milin=cv2.resize(Orig_Milin,dimension)
input_Mia=cv2.resize(Orig_Mia,dimension)

# FFT on two input images
dft_Mia = cv2.dft(np.float32(input_Mia),flags = cv2.DFT_COMPLEX_OUTPUT)
dft_shift_Mia = dft_Mia
magnitude_Mia = cv2.magnitude(dft_shift_Mia[:,:,0],dft_shift_Mia[:,:,1])
enlarge_mag_Mia=20*np.log(np.abs(magnitude_Mia))
dft_complex_Mia = dft_shift_Mia[:,:,0] + 1j*dft_shift_Mia[:,:,1]
phase_Mia=np.angle(dft_complex_Mia) 

dft_Milin = cv2.dft(np.float32(input_Milin),flags = cv2.DFT_COMPLEX_OUTPUT)
dft_shift_Milin = dft_Milin
magnitude_Milin = cv2.magnitude(dft_shift_Milin[:,:,0],dft_shift_Milin[:,:,1])
enlarge_mag_Milin=20*np.log(np.abs(magnitude_Milin))
dft_complex_Milin = dft_shift_Milin[:,:,0] + 1j*dft_shift_Milin[:,:,1]
phase_Milin=np.angle(dft_complex_Milin) 

# create masks
num_row=1700
num_col=np.int(num_row*rows/cols)
HPF_size=(cols,rows,2)
Empty_mat=np.zeros(HPF_size,np.uint8)
HPF=Empty_mat
HPF[crow-num_row:crow+num_row,ccol-num_col:ccol+num_col,:]=1
Full_mat=np.ones(HPF_size,np.uint8)
LPF=Full_mat-HPF

masked_Mia=dft_shift_Mia*LPF
masked_Milin=dft_Milin*HPF
new_image=(masked_Mia+masked_Milin)//2

# iFFT on two images
Mia_back = cv2.idft(new_image)  #
img_Mia_back = cv2.magnitude(Mia_back[:,:,0],Mia_back[:,:,1])  #
img_Mia_back=cv2.resize(img_Mia_back,new_dimension)
#output=np.uint8(img_Mia_back)
#out_put=cv2.cvtColor(img_Mia_back,cv2.COLOR_GRAY2RGB)
img_Mia_back = img_Mia_back / img_Mia_back.max()
img_Mia_back = img_Mia_back * 255
#color_out=cv2.cvtColor(img_Mia_back,cv2.COLOR_GRAY2RGB)


cv2.imwrite('hybrid.jpg',img_Mia_back)


plt.subplot(131),plt.imshow(img_Mia_back)
plt.title('Input Mia'), plt.xticks([]), plt.yticks([])

#plt.subplot(132),plt.imshow(enlarge_mag_Milin)
#plt.title('Magnitude_mia'),plt.xticks([]),plt.yticks([])
#plt.subplot(133),plt.imshow(phase_Milin)
#plt.title('Phase'),plt.xticks([]),plt.yticks([])
