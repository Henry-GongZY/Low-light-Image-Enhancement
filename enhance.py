import os
import cv2
from LIME import LIME

def main():
    filePath = "./demo/2.jpg"

    lime = LIME(iterations=30,alpha=0.15,rho=1.1,gamma=0.6,strategy=2)
    lime.load(filePath)
    lime.run()
    filename = os.path.split(filePath)[-1]
    savePath = f"./demo/enhanced_{filename}"
    cv2.imwrite(savePath, lime.R)

if __name__ == "__main__":
    main()
