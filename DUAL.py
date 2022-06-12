import os
from LIME import LIME
import cv2

class DUAL:
    def __init__(self, iterations, alpha, rho, gamma, limestrategy):
        self.limecore = LIME(iterations,alpha,rho,gamma,limestrategy)

    def load(self, imgPath):
        self.img = cv2.imread(imgPath) / 255
        self.imgrev = 1 - self.img
        self.imgname = os.path.split(imgPath)[-1]

    def multi_exposureimageFushion(self):
        mergecore = cv2.createMergeMertens(1,1,1)
        img = mergecore.process([self.img,self.forwardimg,self.reverseimg])
        return img

    def run(self):
        print('Using LIME for forward illumination enhancement!')
        self.limecore.loadimage(self.img)
        self.forwardimg = self.limecore.run()
        cv2.imwrite("./pics/DUAL_forward_{}".format(self.imgname),self.forwardimg)

        print('Using LIME for reverse illumination enhancement!')
        self.limecore.loadimage(self.imgrev)
        self.reverseimg = 255 - self.limecore.run()
        cv2.imwrite("./pics/DUAL_reverse_{}".format(self.imgname), self.reverseimg)

        print('Use multi-exposure image fusion to generate the result!')
        l = self.multi_exposureimageFushion()
        cv2.imwrite("./pics/DUAL_result_{}".format(self.imgname), l * 255)
