import cv2

import numpy as np
import matplotlib.pyplot as plt
plt.show()
import tensorflow as tf
from tensorflow.keras import layers
import keras

import imutils
import csv


def detect_digit():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train =x_train/255
    x_test =x_test/255
    plt.imshow(x_train[0],cmap='Greys')
    plt.show()
def  saveToCSV(data):
    print(data)
def file_read():
    # srcimg,otmp,itmp,csvname=cmdargpar()
    srcimg='./sample.jpg'
    # slash='\\'
    # srcimg=srcimg.replace(slash, slash+slash)
    # otmp=otmp.replace(slash, slash+slash)
    # itmp=itmp.replace(slash, slash+slash)
    # temp1= cv2.imread(otmp,0)
    # temp2= cv2.imread(itmp,0)
    img=cv2.imread(srcimg, cv2.COLOR_BGR2GRAY)

    return img

def detect_corners(img):
    gray = np.float32(img)
    dst = cv2.cornerHarris(gray, 2, 29, 0.04)

    # result is dilated for marking the corners, not important
    dst = cv2.dilate(dst, None)

    # Threshold for an optimal value, it may vary depending on the image.
    img[dst > 0.01 * dst.max()] = [0, 0, 255]

    cv2.imshow('corners', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def remove_noise(img):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (8, 8))
    # morph = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    morph = cv2.dilate(img, kernel, iterations = 2)
    out_gray = cv2.divide(img, morph, scale=255)
    out_binary = cv2.threshold(out_gray, 0, 255, cv2.THRESH_OTSU)[1]
    # cv2.imshow('remove noise', out_binary)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return out_binary

def invert_img_color(img):
    return np.invert(img)

def main():
    print(keras.__version__)
    loaded_img = file_read()
    # DRAW ROI
    # draw_rect = cv2.rectangle(loaded_img, (200, 368), (1400,  1404), (255, 0, 0), 1, thickness=2)
    ROI =  cv2.selectROI('Select ROI with mouse', loaded_img)
    Crop = loaded_img[int(ROI[1]):int(ROI[1]+ROI[3]),
                      int(ROI[0]):int(ROI[0]+ROI[2])]
    # ROI = [368:1400, 200:1404]
    # Crop = loaded_img[ROI]
    hsv = cv2.cvtColor(Crop, cv2.COLOR_BGR2HSV)

    lower_black = np.array([0, 0, 0])
    upper_black = np.array([50, 50, 50])
    lower_blue = np.array([110,50,50])
    upper_blue= np.array([130,255,255])


    mask = cv2.inRange(Crop, lower_black, upper_black)

    # Isolate hand writen digits by blue color filter
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

    img_without_noise=remove_noise(mask)
    inverted_img = invert_img_color(img_without_noise)
    digit_img_without_noise=remove_noise(mask_blue)

    inverted_blue_img =  invert_img_color(digit_img_without_noise)

    # cv2.imshow('blue colors', inverted_blue_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # TODO: save into a file
    cv2.imwrite('ROI.jpg', Crop);
    cv2.imwrite('no_noise_img.jpg', inverted_img);
    cv2.imwrite('digit_only_img.jpg', inverted_blue_img);

    # color = np.full_like(Crop, (255, 255, 255))
    # res = cv2.bitwise_and(ROI, Crop, mask=mask)
    # result = cv2.add(mask, res)
    #  detect corners

    # gray = cv2.cvtColor(Crop, cv2.COLOR_BGR2GRAY)
    # detect_corners(gray)

    # INFO: detect all corner points
    # corners = cv2.goodFeaturesToTrack(mask, 27, 0.01, 10)
    # corners = np.int0(corners)
    # print(corners)

    # cv2.threshold(mask, mask, 100, 255, cv2.THRESH_BINARY)

    # white=mask.setTo([255, 255, 255], res)

    cv2.namedWindow('detected', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('detected', 600, 600)
    cv2.imshow('detected', inverted_img)
    # with open('Fudicial.csv', 'w', newline='') as fp:
    #     a = csv.writer(fp, delimiter=',')
    #     data = [['X Axis', 'Y Axis'], ['value x', 'value y']]
    #     a.writerows(data)
    # fp.close()
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# The Program first starts executing from Here
if __name__ == "__main__":
    print(imutils.__version__)
    # main()
    detect_digit()
    # input("Press Enter To Exit...")

