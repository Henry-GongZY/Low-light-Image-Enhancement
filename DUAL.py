import os
from LIME import LIME
import cv2

class DUAL:
    # 初始化DUAL中使用的LIME算子
    def __init__(self, iterations, alpha, rho, gamma, limestrategy):
        self.limecore = LIME(iterations,alpha,rho,gamma,limestrategy,exact=True)

    # 读取增强图像
    def load(self, imgPath):
        self.img = cv2.imread(imgPath) / 255
        self.imgrev = 1 - self.img
        self.imgname = os.path.split(imgPath)[-1]

    # 多曝光图像融合
    def multi_exposureimageFushion(self):
        mergecore = cv2.createMergeMertens(1,1,1)
        img = mergecore.process([self.img,self.forwardimg,self.reverseimg])
        return img

    def run(self):
        # 生成 forward (曝光增强图像)
        print('Using LIME for forward illumination enhancement!')
        self.limecore.loadimage(self.img)
        self.forwardimg = self.limecore.run()
        cv2.imwrite("./pics/DUAL_forward_{}".format(self.imgname),self.forwardimg)

        # 生成 reverse (过曝抑制图像)
        print('Using LIME for reverse illumination enhancement!')
        self.limecore.loadimage(self.imgrev)
        self.reverseimg = 255 - self.limecore.run()
        cv2.imwrite("./pics/DUAL_reverse_{}".format(self.imgname), self.reverseimg)

        # 使用 forward 图像，reverse 图像和原图进行图像融合，生成最终结果
        print('Use multi-exposure image fusion to generate the result!')
        l = self.multi_exposureimageFushion()
        cv2.imwrite("./pics/DUAL_result_{}".format(self.imgname), l * 255)
