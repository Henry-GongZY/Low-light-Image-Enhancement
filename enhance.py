import os
import cv2
from LIME import LIME
from DUAL import DUAL

# def main():
#     # load
#     filePath = "./pics/2.jpg"
#     # initiate LIME operator and load pictures
#     lime = LIME(iterations=30,alpha=0.15,rho=1.1,gamma=0.6,strategy=2,exact=True)
#     lime.load(filePath)
#     # run
#     R = lime.run()
#     # save results
#     filename = os.path.split(filePath)[-1]
#     savePath = f"./pics/LIME_{filename}"
#     cv2.imwrite(savePath, R)

def main():
    filePath = "./pics/over_exposed_1.png"
    dual = DUAL(iterations=30, alpha=0.15, rho=1.1, gamma=0.6, limestrategy=2)
    dual.load(filePath)
    dual.run()

if __name__ == "__main__":
    main()
