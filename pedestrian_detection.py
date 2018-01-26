# import the necessary packages
from __future__ import print_function
from imutils.object_detection import non_max_suppression
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2
from decimal import(
    getcontext,
    Decimal)
 
# construct the argument parse and parse the arguments
focal=2.3
realw=889
height=[]
wide=[]
quest=False
# initialize the HOG descriptor/person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

cap=cv2.VideoCapture(0)
print (cap.isOpened())
# cap.set(3,320)
# cap.set(4,240)
print (cap.get(3))
print (cap.get(4))
while(cap.isOpened()):
    ret,image=cap.read()
    # load the image and resize it to (1) reduce detection time
    # and (2) improve detection accuracy
    # image = cv2.imread(frame)
    image = imutils.resize(image, width=min(400, image.shape[1]))
    orig = image.copy()
 
    # detect people in the image
    (rects, weights) = hog.detectMultiScale(image, winStride=(8, 8),
        padding=(32, 32), scale=1.06)
 
    # draw the original bounding boxes
    for (x, y, w, h) in rects:
        cv2.rectangle(orig, (x, y), (x + w, y + h), (0, 0, 255), 2)
 
    # apply non-maxima suppression to the bounding boxes using a
    # fairly large overlap threshold to try to maintain overlapping
    # boxes that are still people
    rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
    pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)
 
    # draw the final bounding boxes
    for (xA, yA, xB, yB) in pick:
        xA=int(xA*1.2)
        yA=int(yA*1.3)
        xB=int(xB*.9)
        yB=int(yB*.85)
        cv2.rectangle(image, (xA, yA), (xB, yB), (0, 255, 0), 2)
        # draw bounding box around feet
        cv2.rectangle(image,(xA,int(yB*.8)),(xB,yB),(0, 255, 0), 2)


        digw=(xB-xA)*0.004064
        distance=((focal/digw)*realw)/1000
        distance=round(distance,2)
        cv2.putText(image,str(distance)+'ft',(xA,yB+25),cv2.FONT_HERSHEY_SIMPLEX,1,(0, 255, 0))
        #  find the height in pixels
        if (len(height)<100 and quest==False):
            wide.append(xB-xA)
            height.append((yB-yA))
        if (len(height)==100 and quest==False):
            height=np.array(height)
            wide=np.array(wide)
            wide_avg=np.average(wide)
            avg=np.average(height)
            quest=True
            print('average width is:  ',wide_avg)
            print ('average height is :  ',avg)
            print('ya: ',yA)
            print('yb: ',yB)
            print('diff',yB-yA)
        
        # print ('distance is: ',distance,'meters   (',distance*3.28084,'  feet)')
 
    # show some information on the number of bounding boxes
    # filename = imagePath[imagePath.rfind("/") + 1:]
    # print("[INFO] {}: {} original boxes, {} after suppression".format(
    #   'person', len(rects), len(pick)))
 
    # show the output images
    # cv2.imshow("Before NMS", orig)
    cv2.imshow("After NMS", image)
    if cv2.waitKey(1) & 0xff==ord('q'):
        break

       
cap.release()
cv2.destroyAllWindows()