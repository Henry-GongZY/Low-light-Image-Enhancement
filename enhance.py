import os
import cv2
from LIME import LIME
from DUAL import DUAL

def main():
    filePath = "./pics/2.jpg"
    lime = LIME(iterations=30,alpha=0.15,rho=1.1,gamma=0.6,strategy=2,exact=True)
    lime.load(filePath)
    R = lime.run()
    filename = os.path.split(filePath)[-1]
    savePath = f"./pics/LIME_{filename}"
    cv2.imwrite(savePath, R)

# def main():
#     filePath = "./pics/2.jpg"
#     dual = DUAL(iterations=30, alpha=0.15, rho=1.1, gamma=0.6, limestrategy=2)
#     dual.load(filePath)
#     dual.run()

if __name__ == "__main__":
    main()
