# CannyEdgeDetection
<p align='center'>
<img src='https://user-images.githubusercontent.com/53872365/147762201-4bb1b250-14b5-4544-9e41-3858ab62e67e.gif' width=25%/>
</p>


This repo performs canny edge detection from scratch to understand how this powerful function of OpenCV works, which is used in a lot of image pre and post processing modules. We are using 1d Guassian filters to perform the task.

<h3>Quick SetUp</h3>
The main code for following steps is in CannyEdgeDetection/canny.py. Simply run following with your images in the same directory.

```
python3 canny.py
```
<h3> Steps </h3>

1. Read a gray scale image you can find from Berkeley Segmentation Dataset, Training images, store it as a matrix named I.
2. Create a one-dimensional Gaussian mask G to convolve with I. The standard deviation(s) of this Gaussian is a parameter to the edge detector (call it σ > 0).
3. Create a one-dimensional mask for the first derivative of the Gaussian in the x and y directions; call these Gx andGy. The same σ>0 value is used as in step2.
4. Convolve the image I with G along the rows to give the x component image (Ix), and down the columns to give the y component image (Iy)
5. Convolve Ix with Gx to give Ix′ , the x component of I convolved with the derivative of the Gaussian, and convolve Iy with Gy to give Iy′ , y component of I convolved with the derivative of the Gaussian
6. Compute the magnitude of the edge response by combining the x and y components. The magnitude of the result can be computed at each pixel (x, y) as: M(x, y) = Ix(x, y) + Iy(x, y) .
7. Implement non-maximum suppression algorithm that we discussed in the lecture. Pixels that are not local maxima should be removed with this method. In other words, not all the pixels indicating strong magnitude are edges in fact. We need to remove false-positive edge locations from the image.
8. Apply Hysteresis thresholding to obtain final edge-map. You may use any existing library function to compute connected components if you want.


<h3>DataSet</h3>
Sample images are available in the repo and are taken from Berkeley Segmentation Dataset.
