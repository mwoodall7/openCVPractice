import numpy as np
import cv2 as cv
from matplotlib import pyplot as plot
from matplotlib import colors


def plotImage(img, imgTitle, isGray=False):
    if isGray:
        plot.imshow(img, cmap='gray')
    else:
        plot.imshow(img)
    plot.title(imgTitle)
    plot.xticks([])
    plot.yticks([])


def gammaCorrect(img, gamma=1.0):
    invG = 1.0/gamma
    gammaTable = np.array([((i/255.0)**invG) * 255 for i in np.arange(0, 256)]).astype('uint8')
    return cv.LUT(img, gammaTable)


def main():
    img = cv.imread('Lily.jpg')

    b, g, r = cv.split(img)
    img = cv.merge((r, g, b))

    main = plot.figure(1)
    main.suptitle("Lily Image")
    main.add_subplot(221)
    plotImage(img, "Main Image", isGray=False)

    main.add_subplot(222)
    plotImage(r, "Red Channel", isGray=True)

    main.add_subplot(223)
    plotImage(g, "Green Channel", isGray=True)

    main.add_subplot(224)
    plotImage(b, "Blue Channel", isGray=True)


    imghsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    h, s, v = cv.split(imghsv)

    hsvMain = plot.figure(2)
    hsvMain.suptitle("Lily HSV")
    hsvMain.add_subplot(221)
    plotImage(img, "Main Image", isGray=False)

    hsvMain.add_subplot(222)
    plotImage(h, "Hue Channel", isGray=True)

    hsvMain.add_subplot(223)
    plotImage(s, "Saturation Channel", isGray=True)

    hsvMain.add_subplot(224)
    plotImage(v, "Value/Intensity Channel", isGray=True)

    histogram = plot.figure(3)
    histogram.suptitle("Image Histrogram")
    histogram.add_subplot(111)

    hist, edges = np.histogram(v, bins=255)

    avgVal = np.average(hist)

    edges = np.delete(edges, 0, 0)
    plot.plot(edges, hist)


    gamma = plot.figure(4)
    gamma.suptitle("Gamma Correction Values")
    gamma.add_subplot(221)
    plotImage(gammaCorrect(v, 1.5), "Gamma = 1.5", isGray=True)

    gamma.add_subplot(222)
    plotImage(gammaCorrect(v, 0.5), "Gamma = 0.5", isGray=True)

    gamma.add_subplot(223)
    plotImage(gammaCorrect(v, 1.2), "Gamma = 1.2", isGray=True)

    gamma.add_subplot(224)
    plotImage(gammaCorrect(v, 0.75), "Gamma = 0.75", isGray=True)

    plot.show()



if __name__  == '__main__':
    main()
