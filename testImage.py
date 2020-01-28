import cv2
import matplotlib.pyplot as plt
import numpy as np


image_path='/Users/liuxiaomeng/Desktop/apple.jpeg'
original=cv2.imread(image_path,1)
dimension=original.shape
height=dimension[0]
width=dimension[1]
channels=dimension[2]
dim_big=(10*height,10*width)
dim=(height,width)
#image=cv2.cvtColor(original,cv2.COLOR_BGR2RGB)
big_image=cv2.resize(original,dim_big)

new_1=cv2.resize(big_image,dim,interpolation=cv2.INTER_NEAREST)
new_2=cv2.resize(big_image,dim,interpolation=cv2.INTER_CUBIC)


difference1=cv2.absdiff(original,new_1)
difference2=cv2.absdiff(original,new_2)

#cv2.imwrite('elephant 10xup nearest.png',big_image1)
#cv2.imwrite('elephant 10xup bicubic.png',big_image2)

error_1=np.sum(difference1[:,:,:])
error_2=np.sum(difference2[:,:,:])
print(error_1)
print(error_2)



#print('original:')
#print(image)

#
#image=original+256
##print('after ')
##print(image)
##plt.imshow(np.uint8(image))
#
#(b,g,r)=cv2.split(original)
#b=cv2.add(b,256)
#g=cv2.add(g,256)
#r=cv2.add(r,256)
#new_image=cv2.merge((b,g,r))
#
#plt.imshow(new_image)

# display a image

#cv2.imshow('output',image)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

#plt.imshow(image)
#cv2.imwrite(image)

#crop out the baby elephant
# new_image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
# small_elephant=image[400:1000,0:600,:]
# plt.imshow(small_elephant)
# cv2.imwrite('babyElephant.png',small_elephant)