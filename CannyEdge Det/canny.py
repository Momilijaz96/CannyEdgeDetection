from math import pi, sqrt, exp, atan2
from numpy.lib.type_check import imag
import cv2
import numpy as np
from skimage import  filters



#Read the gray scale image
img = cv2.imread("img3.png",cv2.IMREAD_GRAYSCALE)
print("Image Shape: ",img.shape)

#Create a 1d guassian filter of length n and std sgima
def gauss(n=11,sigma=1):
    r = range(-int(n/2),int(n/2)+1)
    return [1 / (sigma * sqrt(2*pi)) * exp(-float(x)**2/(2*sigma**2)) for x in r]

sigma=3
filter_length=11
G=gauss(filter_length,sigma)

#Create a 1d mask for 1st derivate of g in x and y
Gx=[1,-1]
Gy=[[1],[-1]]

#convole img with G along rows, to get Ix
Ix=np.array([np.convolve(row,G) for row in img])

#convole img with G along cols, to get Iy
Iy=np.array([np.convolve(col,G) for col in Ix.transpose()])

print("Ix: ",Ix.shape)

#Convolve Ix with Gx to get Ix'
Ix_dash=np.array([np.convolve(row,Gx) for row in Ix])
print("Ix dash:",Ix_dash.shape)

#Convolve Iy with Gy to get Iy'
Iy_dash=np.array([np.convolve(col,Gx) for col in Iy])
print("Iy: ",Iy.shape)

print("Iy dash:",Iy_dash.shape)

#Adjust the dimensions before magnitude cal
Ix_dash=np.append(Ix_dash,np.zeros((filter_length,img.shape[1]+filter_length)),axis=0)
print("Ix dash new:",Ix_dash.shape)

#Iy_dash=np.append(Iy_dash,np.zeros((filter_length,img.shape[0]+filter_length)),axis=0)
Iy_dash=np.append(Iy_dash,np.zeros((1,img.shape[0]+filter_length)),axis=0)

Iy_dash=Iy_dash.T
print("Iy dash new:", Iy_dash.shape)

#Compute magnitude at each pixel
grad_mag=np.sqrt(np.square(Ix_dash)+np.square(Iy_dash))


#Compute orientation of gradient
grad_dir=np.zeros((Iy_dash.shape[0],Iy_dash.shape[1]),dtype=float)
print(grad_dir.shape)
for i in range(Iy_dash.shape[0]):
    for j in range(Iy_dash.shape[1]): 
        grad_dir[i][j]=atan2(Iy_dash[i][j],Ix_dash[i][j])

#Apply non-max suppression
def non_max_suppression(img, D):
    M, N = img.shape
    Z = np.zeros((M,N), dtype=np.int32)
    angle = D * 180. / np.pi
    angle[angle < 0] += 180


    for i in range(1,M-1):
        for j in range(1,N-1):
            try:
                q = 255
                r = 255
                
                #angle 0
                if (0 <= angle[i,j] < 22.5) or (157.5 <= angle[i,j] <= 180):
                    q = img[i, j+1]
                    r = img[i, j-1]
                #angle 45
                elif (22.5 <= angle[i,j] < 67.5):
                    q = img[i+1, j-1]
                    r = img[i-1, j+1]
                #angle 90
                elif (67.5 <= angle[i,j] < 112.5):
                    q = img[i+1, j]
                    r = img[i-1, j]
                #angle 135
                elif (112.5 <= angle[i,j] < 157.5):
                    q = img[i-1, j-1]
                    r = img[i+1, j+1]

                if (img[i,j] >= q) and (img[i,j] >= r):
                    Z[i,j] = img[i,j]
                else:
                    Z[i,j] = 0

            except IndexError as e:
                pass

    return Z

#Apply Hysterisis thresholding on the thin edges
nm_img=non_max_suppression(grad_mag,grad_dir)

#Set low and high values for the edge thresholding
low = 0.1
high = 0.35
#Perform hysterisis thresholding
hyst = filters.apply_hysteresis_threshold(nm_img, low, high)

#Visualization

cv2.imshow("Ix_dash",Ix_dash)
cv2.imshow("Iy_dash",Iy_dash)

cv2.imshow("Non max suppresion Image",nm_img.astype(np.float))
cv2.imshow("Grad Magnitude",grad_mag)
cv2.imshow("Grad direction",grad_dir)

cv2.imshow('Hysterisis thresholding',hyst.astype(float))
cv2.imshow("Image",img)


cv2.waitKey(0) # waits until a key is pressed
cv2.destroyAllWindows() # destroys the window showing image
