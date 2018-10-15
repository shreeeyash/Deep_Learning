import cv2
import numpy as np

image = cv2.imread('img2.jpg')
image = np.asarray(image,dtype = 'float64')

original_int = cv2.imread('img2.jpg')
original = np.asarray(original_int,dtype = 'float64')



#=======rescaling====================
reshaped_img = np.reshape(image,(-1,3))
reshaped_img -= np.mean(reshaped_img,0)
reshaped_img /= np.std(reshaped_img,0)

#=============PCA====================
cov_matrix = np.cov(reshaped_img,rowvar=False)
lambdas, p = np.linalg.eig(cov_matrix) 
a = np.random.normal(0,0.1,3)

#==========augmentation=============
aug = np.dot(p, a*lambdas)
aug_img = reshaped_img + aug
aug_img = np.reshape(aug_img,(original.shape[0],original.shape[1],3))

x = (aug_img*255).astype('int8')

cv2.imshow('original', original_int)
add1=cv2.add(original,aug_img)
cv2.imshow('image+aug_img',(add1).astype('int8'))
add2=cv2.add((original).astype('int8'),x)
cv2.imshow('image+x',(add2).astype('int8'))
pca_color_image = np.maximum(np.minimum((original_int).astype('int') + x, 255), 0).astype('uint8')
cv2.imshow('pca',pca_color_image)

cv2.waitKey()