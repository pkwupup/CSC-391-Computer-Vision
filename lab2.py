import numpy as np
import cv2
cap = cv2.VideoCapture(0)
# ########################## cornerHarris
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret==True:
        img=cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        corner=cv2.cornerHarris(img,7,7,0.1)
        img2=img.copy()
        frame[corner > 0.1 * corner.max()] = [0, 0, 255]
        # out.write(frame)

        cv2.imshow('frame',frame)
        # use Esc as keyboard interrupt
        if cv2.waitKey(1) == 27:
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()

# ########################## canny edge detection
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret==True:
        frame = cv2.Canny(frame,50,100)
        cv2.imshow('frame',frame)
        # use Esc as keyboard interrupt
        if cv2.waitKey(1) == 27:
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()

# ########################## SIFT
sift=cv2.xfeatures2d.SIFT_create(0,3,0.05,5,1.6)
bfmatcher=cv2.BFMatcher_create(cv2.NORM_L2,crossCheck=True)

img=cv2.imread('test.JPG')
img=cv2.resize(img, (0, 0), fx=0.25, fy=0.25)
gray=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
kp=sift.detect(gray,None)
kp,dsc=sift.compute(gray,kp)

img2=cv2.drawKeypoints(gray,kp,img,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imshow('kp',img2)
cv2.waitKey(0)
cv2.imwrite('kp.jpg',img2)
cv2.destroyAllWindows()

# ########################## SIFT scaling
sift=cv2.xfeatures2d.SIFT_create(0,3,0.05,10,1.6)
bfmatcher=cv2.BFMatcher_create(cv2.NORM_L2,crossCheck=True)

img=cv2.imread('DSC_9259.JPG')
#img=cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
gray=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)

img2=cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
gray2=cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)

kp=sift.detect(gray,None)
kp,dsc=sift.compute(gray,kp)
kp2 = sift.detect(gray2, None)
kp2,dsc2=sift.compute(gray2,kp2)

matcher = bfmatcher.match(dsc,dsc2)
matcher = sorted(matcher, key= lambda  x: x.distance)
img3=cv2.drawMatches(img,kp,img2,kp2,matcher[:20],None,flags=2)
cv2.imshow('kpMatch',img3)
cv2.waitKey(0)
cv2.destroyAllWindows()

# ########################## SIFT rotation
sift=cv2.xfeatures2d.SIFT_create(0,3,0.05,10,1.6)
bfmatcher=cv2.BFMatcher_create(cv2.NORM_L2,crossCheck=True)

img=cv2.imread('DSC_9259.JPG')
img=cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
gray=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)

img2=cv2.flip(img,0)
gray2=cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)

kp=sift.detect(gray,None)
kp,dsc=sift.compute(gray,kp)
kp2 = sift.detect(gray2, None)
kp2,dsc2=sift.compute(gray2,kp2)

matcher = bfmatcher.match(dsc,dsc2)
matcher = sorted(matcher, key= lambda  x: x.distance)
img3=cv2.drawMatches(img,kp,img2,kp2,matcher[:20],None,flags=2)
cv2.imshow('kpMatch',img3)
cv2.waitKey(0)
cv2.destroyAllWindows()

# ########################## matching by corners
sift=cv2.xfeatures2d.SIFT_create(0,3,0.05,10,1.6)
bfmatcher=cv2.BFMatcher_create(cv2.NORM_L2,crossCheck=True)
img=cv2.imread('compare.JPG')
img=cv2.resize(img, (0, 0), fx=0.25, fy=0.25)
gray=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)

img2=cv2.imread('compare1.JPG')
img2=cv2.resize(img2, (0, 0), fx=0.25, fy=0.25)
gray2=cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
corners=cv2.cornerHarris(gray,7,7,0.1)
corners2=cv2.cornerHarris(gray2, 7, 7, 0.1)
kpsCorners=np.argwhere(corners > 0.1*corners.max())
kpsCorners = [cv2.KeyPoint(pt[1],pt[0],2) for pt in kpsCorners]
kpsCorners,dstCorners = sift.compute(gray,kpsCorners)
kpsCorners2=np.argwhere(corners > 0.1*corners2.max())
kpsCorners2 = [cv2.KeyPoint(pt[1],pt[0],2) for pt in kpsCorners2]
kpsCorners2,dstCorners2 = sift.compute(gray2,kpsCorners2)
matcher = bfmatcher.match(dstCorners,dstCorners2)
matcher = sorted(matcher, key= lambda  x: x.distance)
img4=cv2.drawMatches(img,kpsCorners,img2,kpsCorners2,matcher[:20],None,flags=2)
cv2.imshow('cornerMatch',img4)
cv2.waitKey(0)
cv2.destroyAllWindows()

# ########################## matching by SIFT
sift=cv2.xfeatures2d.SIFT_create(0,3,0.05,10,1.6)
bfmatcher=cv2.BFMatcher_create(cv2.NORM_L2,crossCheck=True)
img=cv2.imread('compare.JPG')
img=cv2.resize(img, (0, 0), fx=0.25, fy=0.25)
gray=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)

img2=cv2.imread('compare1.JPG')
img2=cv2.resize(img2, (0, 0), fx=0.25, fy=0.25)
gray2=cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
kp=sift.detect(gray,None)
kp,dsc=sift.compute(gray,kp)
kp2 = sift.detect(gray2, None)
kp2,dsc2=sift.compute(gray2,kp2)

matcher = bfmatcher.match(dsc,dsc2)
matcher = sorted(matcher, key= lambda  x: x.distance)
img3=cv2.drawMatches(img,kp,img2,kp2,matcher[:10],None,flags=2)
cv2.imshow('SIFTMatch',img3)
cv2.waitKey(0)
cv2.destroyAllWindows()

