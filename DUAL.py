import numpy as np
from skimage import exposure
from LIME import LIME
import cv2

class DUAL:
    def __init__(self, iterations, alpha, rho, gamma, limestrategy):
        self.limecore = LIME(iterations,alpha,rho,gamma,limestrategy)

    def load(self, imgPath):
        self.img = cv2.imread(imgPath) / 255
        self.imgrev = 1 - self.img


    def multi_exposureimageFushion(self):
        mergecore = cv2.createMergeMertens(1,1,1)
        img = mergecore.process([self.img,self.forwardimg,self.reverseimg])
        return img

    def run(self):
        print('Using LIME for forward illumination enhancement!')
        self.limecore.loadimage(self.img)
        self.forwardimg = self.limecore.run()

        print('Using LIME for reverse illumination enhancement!')
        self.limecore.loadimage(self.imgrev)
        self.reverseimg = 255 - self.limecore.run()

        print('Use multi-exposure image fusion to generate the result!')
        l = self.multi_exposureimageFushion()
        savePath = "./pics/enhanced_ll.jpg"
        cv2.imwrite(savePath, exposure.rescale_intensity(l, (0, 1)) * 255)
