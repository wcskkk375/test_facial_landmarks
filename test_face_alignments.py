import cv2
import numpy as np
import apply_regression
def test():
	img=cv2.imread("test_image/1.jpg")
	apply_regression.go(-1,0,img)
test()

