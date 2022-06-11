import numpy as np
from scipy.fft import *
from skimage import exposure
import os
import cv2
from tqdm import trange

def firstOrderDerivative(n, k=1):
    return np.eye(n) * (-1) + np.eye(n, k=k)

class LIME:
    def __init__(self, iterations, alpha, rho, gamma, strategy, *args, **kwargs):
        self.iterations = iterations
        self.alpha = alpha
        self.rho = rho
        self.gamma = gamma
        self.strategy = strategy

    def load(self, imgPath):
        self.L = cv2.imread(imgPath) / 255
        self.row = self.L.shape[0]
        self.col = self.L.shape[1]

        self.T_hat = np.max(self.L, axis=2)
        self.dv = firstOrderDerivative(self.row)
        self.dh = firstOrderDerivative(self.col, -1)

        dxe = np.zeros((self.row, self.col))
        dye = np.zeros((self.row, self.col))

        dxe[1, 0] = 1
        dxe[1, 1] = -1
        dye[0, 1] = 1
        dye[1, 1] = -1

        dxf = fft2(dxe)
        dxc = np.conj(dxf)
        dx_mod = dxc * dxf

        dyf = fft2(dye)
        dyc = np.conj(dyf)
        dy_mod = dyc * dyf

        self.DD = dx_mod + dy_mod

        self.W = self.weightingStrategy()

    def weightingStrategy(self):
        if self.strategy == 2:
            dTv = self.dv @ self.T_hat
            dTh = self.T_hat @ self.dh
            Wv = 1 / (np.abs(dTv) + 1)
            Wh = 1 / (np.abs(dTh) + 1)
            return np.vstack((Wv, Wh))
        else:
            return np.ones((self.row * 2, self.col))

    def T_sub(self, G, Z, miu):
        X = G - Z / miu
        Xv = X[:self.row, :]
        Xh = X[self.row:, :]

        numerator = fft2(2 * self.T_hat + miu * (self.dv @ Xv + Xh @ self.dh))
        denominator = self.DD * miu + 2
        T = ifft2(numerator / denominator)
        T = np.real(T)

        return exposure.rescale_intensity(T, (0, 1), (0.001, 1))

    def G_sub(self, T, Z, miu, W):
        dT = np.vstack([self.dv @ T,T @ self.dh])
        epsilon = self.alpha * W / miu
        X = dT + Z / miu
        return np.sign(X) * np.maximum(np.abs(X) - epsilon, 0)

    def Z_sub(self, T, G, Z, miu):
        dT = np.vstack([self.dv @ T,T @ self.dh])
        return Z + miu * (dT - G)

    def miu_sub(self, miu):
        return miu * self.rho

    def run(self):
        T = np.zeros((self.row, self.col))
        G = np.zeros((self.row * 2, self.col))
        Z = np.zeros((self.row * 2, self.col))
        miu = 1

        for i in trange((0,self.iterations)):
            T = self.T_sub(G, Z, miu)
            G = self.G_sub(T, Z, miu, self.W)
            Z = self.Z_sub(T, G, Z, miu)
            miu = self.miu_sub(miu)

        self.T = T ** self.gamma
        self.R = self.L / np.repeat(self.T[..., None], 3, axis = -1)
        self.R = exposure.rescale_intensity(self.R, (0, 1)) * 255
        return self.R


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
