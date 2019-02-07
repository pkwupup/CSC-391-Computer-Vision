# Created by Kewen Peng
# Project 1
import cv2
import numpy as np
import matplotlib.pyplot as plt
import skimage as ski

# ******************************************************** Spatial Filtering
img=cv2.imread("DSC_9259.JPG")
cv2.imshow('original', img)
cv2.waitKey()
imgdata = np.zeros(img.shape, np.float64)
if len(img.shape) == 2:
    imgdata = img
else:
    imgdata[:, :, 0] = img[:, :, 0]
    imgdata[:, :, 1] = img[:, :, 1]
    imgdata[:, :, 2] = img[:, :, 2]

# define the filter size here
filtersize=5
# the sum adds up to 1
w = np.random.randn(filtersize, filtersize)
w = w / w.sum()
print("Filter matrix: ")
print(w)
# show the matrix as a fig
plt.matshow(w)
plt.show()

filtered = np.zeros(imgdata.shape, np.float64)  # array for filtered image
# Apply the filter to each channel
filtered[:, :, 0] = cv2.filter2D(imgdata[:, :, 0], -1, w)
filtered[:, :, 1] = cv2.filter2D(imgdata[:, :, 1], -1, w)
filtered[:, :, 2] = cv2.filter2D(imgdata[:, :, 2], -1, w)
filtered=filtered.astype(np.uint8)
cv2.imshow('spatial', filtered.astype(np.uint8))
cv2.imwrite("spatialFiltering.jpg",filtered)
cv2.waitKey()
cv2.destroyAllWindows()


# ******************************************************** Gaussian vs Median Filtering
img=cv2.imread("DSC_9259-0.50.JPG")
gaussian = cv2.GaussianBlur(img,(5,5),0,0,0,0)
cv2.imshow("Gaussian_5",gaussian)
cv2.imwrite("Gaussian_5.jpg",gaussian)
cv2.waitKey()

gaussian = cv2.GaussianBlur(img,(11,11),0,0,0,0)
cv2.imshow("Gaussian_11",gaussian)
cv2.imwrite("Gaussian_11.jpg",gaussian)
cv2.waitKey()

median_5=cv2.medianBlur(img,5,0)
cv2.imshow("Median_5",median_5)
cv2.imwrite("Median_5.jpg",median_5)
cv2.waitKey()

median=cv2.medianBlur(img,11,0)
cv2.imshow("Median_11",median)
cv2.imwrite("Median_11.jpg",median)
cv2.waitKey()
cv2.destroyAllWindows()

#******************************************************** Edge detection
img=cv2.imread("DSC_9259.JPG")
edge=cv2.Canny(img,100,200)
cv2.imshow("edge",edge)
cv2.imwrite("edge.jpg",edge)
cv2.waitKey()

noisy=cv2.imread("DSC_9259-0.50.JPG")
noisy=cv2.GaussianBlur(noisy,(3,3),0,0,0,0)
cv2.imshow('noisy',noisy)
cv2.waitKey()
edge=cv2.Canny(noisy,150,250)
cv2.imshow("edge_noisy",edge)
cv2.imwrite("edge_noisy.jpg",edge)
cv2.waitKey()

other=cv2.imread("window-06-04.jpg")
cv2.imshow("other",other)
other=other+50
cv2.waitKey()
edge_other=cv2.Canny(other,200,300)
cv2.imshow("edge_other",edge_other)
cv2.imwrite("edge_other.jpg",edge_other)
cv2.waitKey()
cv2.destroyAllWindows()

# ******************************************************** DFT
img=cv2.imread("DSC_9259.JPG")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imwrite("grayImg.jpg", gray)
xx, yy = np.mgrid[0:gray.shape[0], 0:gray.shape[1]]
# show the original gray picture
cv2.imshow("original",gray)
cv2.waitKey()

# The 2-D DFT in the form of a 2-D image with low freq in the center
F2_gray = np.fft.fft2(gray.astype(float))
logMagnitudeImage = np.fft.fftshift(np.log(np.abs(F2_gray)+1))
logMagnitudeImage = logMagnitudeImage / logMagnitudeImage.max()   # scale to [0, 1]
logMagnitudeImage = ski.img_as_ubyte(logMagnitudeImage)
cv2.imshow('Log Magnitude plot', logMagnitudeImage)
cv2.imwrite('log_magnitude.jpg',logMagnitudeImage)
cv2.waitKey()
cv2.destroyAllWindows()

magnitudeImage = np.fft.fftshift(np.abs(F2_gray))
magnitudeImage = magnitudeImage / magnitudeImage.max()   # scale to [0, 1]
magnitudeImage = ski.img_as_ubyte(magnitudeImage)
cv2.imwrite("magnitude.jpg",magnitudeImage)

# compare dft of the noisy image
img=cv2.imread("DSC_9259-0.50.JPG")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
F2_gray = np.fft.fft2(gray.astype(float))
logMagnitudeImage = np.fft.fftshift(np.log(np.abs(F2_gray)+1))
logMagnitudeImage = logMagnitudeImage / logMagnitudeImage.max()   # scale to [0, 1]
logMagnitudeImage = ski.img_as_ubyte(logMagnitudeImage)
cv2.imwrite('log_magnitude_noisy.jpg',logMagnitudeImage)

# ******************************************************** Frequency filtering low pass
img=cv2.imread("DSC_9259.JPG")
img = cv2.resize(img, (0, 0), fx=0.25, fy=0.25) #subsample of the original image
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

V = (np.linspace(-int(gray.shape[0] / 2), int(gray.shape[0] / 2) - 1, gray.shape[0]))
U = (np.linspace(-int(gray.shape[1] / 2), int(gray.shape[1] / 2) - 1, gray.shape[1]))
U, V = np.meshgrid(U, V)
# The function over (U,V) is distance between each point (u,v) to (0,0)
D = np.sqrt(V*V + U*U)
# create x-points for plotting
xval = np.linspace(-int(gray.shape[1] / 2), int(gray.shape[1] / 2) - 1, gray.shape[1])
# Specify a frequency cutoff value as a function of D.max()
D0 = 0.20 * D.max()

idealLowPass = D <= D0

FTgray = np.fft.fft2(gray.astype(float))
FTgrayFiltered = FTgray * np.fft.fftshift(idealLowPass)
graySmallFiltered = np.abs(np.fft.ifft2(FTgrayFiltered))

graySmallFiltered = ski.img_as_ubyte(graySmallFiltered / graySmallFiltered.max())
cv2.imwrite('originalGray.jpg',gray)
cv2.imwrite('graylow.jpg',graySmallFiltered)

cv2.imshow("grayImageLowpassFiltered.jpg", graySmallFiltered)

orilogMagnitudeImage = np.fft.fftshift(np.log(np.abs(FTgray)+1))
orilogMagnitudeImage = orilogMagnitudeImage / orilogMagnitudeImage.max()   # scale to [0, 1]
orilogMagnitudeImage = ski.img_as_ubyte(orilogMagnitudeImage)
cv2.imwrite('original.jpg',orilogMagnitudeImage)
cv2.waitKey()

logMagnitudeImage = np.fft.fftshift(np.log(np.abs(FTgrayFiltered)+1))
logMagnitudeImage = logMagnitudeImage / logMagnitudeImage.max()   # scale to [0, 1]
logMagnitudeImage = ski.img_as_ubyte(logMagnitudeImage)
cv2.imwrite('lowpass.jpg',logMagnitudeImage)

#Plot the ideal filter and then create and plot Butterworth filters of order
#n = 1, 2, 3, 4
idealLowPass = ski.img_as_ubyte(idealLowPass / idealLowPass.max())
plt.plot(xval, idealLowPass[int(idealLowPass.shape[0]/2), :], 'c--', label='ideal')
colors='brgkmc'
for n in range(1, 5):
    H = 1.0 / (1 + (np.sqrt(2) - 1)*np.power(D/D0, 2*n))
    FTgrayFiltered = FTgray * np.fft.fftshift(H)
    graySmallFiltered = np.abs(np.fft.ifft2(FTgrayFiltered))
    graySmallFiltered = ski.img_as_ubyte(graySmallFiltered / graySmallFiltered.max())
    cv2.imwrite("grayImageButterworth-low-" + str(n) + ".jpg", graySmallFiltered)
    H = ski.img_as_ubyte(H / H.max())
    cv2.imwrite("butter-" + str(n) + ".jpg", H)
    # Get a slice through the center of the filter to plot in 2-D
    slice = H[int(H.shape[0]/2), :]
    plt.plot(xval, slice, colors[n-1], label='n='+str(n))
    plt.legend(loc='upper left')
    plt.show()
cv2.destroyAllWindows()

# ******************************************************** Frequency filtering high pass
img=cv2.imread("DSC_9259.JPG")
img = cv2.resize(img, (0, 0), fx=0.25, fy=0.25) #subsample of the original image
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

V = (np.linspace(-int(gray.shape[0] / 2), int(gray.shape[0] / 2) - 1, gray.shape[0]))
U = (np.linspace(-int(gray.shape[1] / 2), int(gray.shape[1] / 2) - 1, gray.shape[1]))
U, V = np.meshgrid(U, V)
# The function over (U,V) is distance between each point (u,v) to (0,0)
D = np.sqrt(V*V + U*U)
# create x-points for plotting
xval = np.linspace(-int(gray.shape[1] / 2), int(gray.shape[1] / 2) - 1, gray.shape[1])
# Specify a frequency cutoff value as a function of D.max()
D0 = 0.20* D.max()
idealHighPass = D >= D0
FTgray = np.fft.fft2(gray.astype(float))
FTgrayFiltered = FTgray * np.fft.fftshift(idealHighPass)
grayFiltered = np.abs(np.fft.ifft2(FTgrayFiltered))

idealHighPass = ski.img_as_ubyte(idealHighPass / idealHighPass.max())
grayFiltered = ski.img_as_ubyte(grayFiltered / grayFiltered.max())
cv2.imshow("idealHighPass.jpg", idealHighPass)
cv2.waitKey()
cv2.imwrite("grayImageIdealHighpassFiltered.jpg", grayFiltered)

# Plot the ideal filter and then create and plot Butterworth filters of order
# n = 1, 2, 3, 4
plt.plot(xval, idealHighPass[int(idealHighPass.shape[0] / 2), :], 'c--', label='ideal')
colors='brgkmc'
for n in range(1, 5):
    H = 1.0 / (1 + (np.sqrt(2) - 1)*np.power(D/D0, 2*n))
    H=1-H
    FTgrayFiltered = FTgray * np.fft.fftshift(H)
    grayFiltered = np.abs(np.fft.ifft2(FTgrayFiltered))
    grayFiltered = ski.img_as_ubyte(grayFiltered / grayFiltered.max())
    cv2.imwrite("grayImageButterworth-high-" + str(n) + ".jpg", grayFiltered)
    H = ski.img_as_ubyte(H / H.max())
    cv2.imwrite("butter-" + str(n) + ".jpg", H)
    slice = H[int(H.shape[0]/2), :]
    plt.plot(xval, slice, colors[n-1], label='n='+str(n))
    plt.legend(loc='upper left')
    plt.show()
cv2.destroyAllWindows()