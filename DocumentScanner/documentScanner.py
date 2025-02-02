import numpy as np 
import cv2 
import matplotlib.pyplot as plt 
import imutils 
from imutils.perspective import four_point_transform


def resizer(image,width=500):
    h,w,c = image.shape
    height = int((h/w)*width)

    img = cv2.resize(image,(width,height))
    size = (width,height)
    return img,size


def document_scanner(img_orig):
    img_re,size = resizer(img_orig,width = 500)
    flag = 0


    detail = cv2.detailEnhance(img_re, sigma_s = 20, sigma_r = 0.15)#Enhance
    gray = cv2.cvtColor(detail,cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(5,5),0)

    edge_image = cv2.Canny(blur,75,200)

    kernel = np.ones((5,5), np.uint8)
    dilate = cv2.dilate(edge_image, kernel,iterations=1)
    closing = cv2.morphologyEx(dilate,cv2.MORPH_CLOSE,kernel)

    contours, hire = cv2.findContours(closing,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)

    contours = sorted(contours, key = cv2.contourArea, reverse = True)
    for contour in contours:
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02*peri, True)
        
        if len(approx)==4:
            four_points = np.squeeze(approx)
            break
    if flag == 0:    
        cv2.drawContours(img_re,[four_points],-1,(0,0,255),3)
        flag = 1 

    multiplier = img_orig.shape[1]/size[0]
    four_points_orig = four_points*multiplier
    four_points_orig = four_points_orig.astype(int)

    wrap_image = four_point_transform(img_orig,four_points_orig)
    
    return wrap_image,four_points_orig,img_re,closing


img = cv2.imread(r"C:\Users\Debajyoti\OneDrive\Desktop\Project 2\DocumentScanner\images\03.jpg")
wrapping,points,cnt_img,edging = document_scanner(img)

cv2.imshow("original",img)
cv2.imshow("wrapped img",wrapping)
cv2.imshow("edge",edging)
cv2.imshow("cnt image",cnt_img)
cv2.waitKey(0)
cv2.destroyAllWindows()





