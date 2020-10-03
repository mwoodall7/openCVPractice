import sys
import os
import numpy as np
import cv2 as cv
import ctypes
import math

user32 = ctypes.windll.user32

screenSize = (user32.GetSystemMetrics(0), user32.GetSystemMetrics(1))


def gammaCorrect(img, gamma=1.0):
    invG = 1.0/gamma
    gammaTable = np.array([((i/255.0)**invG) * 255 for i in np.arange(0, 256)]).astype('uint8')
    return cv.LUT(img, gammaTable)


def setupSobelImage(img, axis='x', kernel=1):
    xval = 0
    yval = 0
    if axis == 'x':
        xval = 1
    elif axis == 'y':
        yval = 1
    elif axis == 'both':
        xval = 1
        yval = 1
    sobel_64F = cv.Sobel(img, cv.CV_64F, xval, yval, ksize=kernel)
    absSobel = np.absolute(sobel_64F)
    return np.uint8(absSobel)


def showImages(imgDict):
    for imgName, img in imgDict.items():
        cv.imshow(imgName, img)


def initializeWindowPositions(imgNames):
    numImgs = len(imgNames)
    colCount = 0
    rowCount = 0
    winResX = 625
    winResY = 525
    for imgName in imgNames:
        cv.namedWindow(imgName)
        cv.resizeWindow(imgName, winResX, winResY)
        if (int(colCount*winResX) + winResX) > screenSize[0]:
            xval = 0
        else:
            xval = int(colCount*winResX)

        if (int(colCount*winResX) + winResX) > screenSize[0]:
            rowCount += 1

        yval = int(rowCount*winResY)

        cv.moveWindow(imgName, xval, yval)

        if xval == 0 and not int(colCount*winResX) == 0:
            colCount = 0
        else:
            colCount += 1


def main():
    vid = cv.VideoCapture(0)
    vid.set(cv.CAP_PROP_AUTO_EXPOSURE, 0.25)
    exposureTimeInMs = 60
    exposureTimeInExp = int(math.log2(exposureTimeInMs/1000))
    vid.set(cv.CAP_PROP_EXPOSURE, exposureTimeInMs)
    gamma = 1.0
    increment = 0.05
    imgThresh = 50
    font = cv.FONT_HERSHEY_PLAIN

    imgDict = {'Feed':None, 'Laplacian':None, 'Sobel X':None, 'Sobel Y':None, 'Sobel Both':None}

    initializeWindowPositions(imgDict.keys())

    while True:
        vid.set(cv.CAP_PROP_EXPOSURE, exposureTimeInExp)

        success, frame = vid.read()

        grayscale = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        if gamma != 1.0:
            grayscale = gammaCorrect(grayscale, gamma)

        avgVal = np.average(grayscale)

        cv.putText(grayscale, 'Gamma = {:.2f}'.format(gamma), (10, 50), font, 4, (0, 0, 255), 2, cv.LINE_AA)
        cv.putText(grayscale, 'Avg V = {:.2f}'.format(avgVal), (10, 150), font, 4, (255, 0, 0), 2, cv.LINE_AA)
        cv.putText(grayscale, 'Exposure Val = {:.2f}'.format(exposureTimeInMs), (10, 200), font, 3, (255, 0, 255), 2, cv.LINE_AA)

        imgDict['Feed'] = grayscale

        laplacian = cv.Laplacian(grayscale, cv.CV_64F)

        sobelx = setupSobelImage(grayscale, axis='x', kernel=3)
        sobely = setupSobelImage(grayscale, axis='y', kernel=3)
        sobelBoth = setupSobelImage(grayscale, axis='both', kernel=3)

        imgDict['Laplacian'] = np.uint8(np.multiply((laplacian>imgThresh), 255))
        imgDict['Sobel X'] = np.uint8(np.multiply((sobelx>imgThresh), 255))
        imgDict['Sobel Y'] = np.uint8(np.multiply((sobely>imgThresh), 255))
        imgDict['Sobel Both'] = np.uint8(np.multiply((sobelBoth>imgThresh), 255))

        showImages(imgDict)

        if not success:
            break

        keypress = cv.waitKey(1) & 0xFF

        if keypress == ord('q'):
            break

        if keypress == ord('w'):
            gamma += increment

        if keypress == ord('s'):
            gamma -= increment

        if keypress == ord('e'):
            exposureTimeInMs += 1

        if keypress == ord('d'):
            if exposureTimeInMs <= 1:
                inc = 0.05
            else:
                inc = 1
            exposureTimeInMs -= inc

        exposureTimeInExp = int(math.log2(exposureTimeInMs/1000))

    vid.release()
    cv.destroyAllWindows()


if __name__  == '__main__':
    main()
