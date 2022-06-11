import numpy as np
from scipy.fft import *
from skimage import exposure
import cv2
from tqdm import trange

class LIME:
    def __init__(self, iterations, alpha, rho, gamma, strategy):
        self.iterations = iterations
        self.alpha = alpha
        self.rho = rho
        self.gamma = gamma
        self.strategy = strategy

    def load(self, imgPath):
        self.loadimage(cv2.imread(imgPath) / 255)

    def loadimage(self,L):
        self.L = L
        self.row = self.L.shape[0]
        self.col = self.L.shape[1]

        self.T_esti = np.max(self.L, axis=2)
        self.dv = -np.eye(self.row) + np.eye(self.row, k=1)
        self.dh = -np.eye(self.col) + np.eye(self.col, k=-1)

        dx = np.zeros((self.row, self.col))
        dy = np.zeros((self.row, self.col))
        dx[1, 0] = 1
        dx[1, 1] = -1
        dy[0, 1] = 1
        dy[1, 1] = -1

        dxf = fft2(dx)
        dyf = fft2(dy)

        self.DTD = np.conj(dxf) * dxf + np.conj(dyf) * dyf
        self.W = self.Strategy()

    def Strategy(self):
        if self.strategy == 2:
            Wv = 1 / (np.abs(self.dv @ self.T_esti) + 1)
            Wh = 1 / (np.abs(self.T_esti @ self.dh) + 1)
            return np.vstack((Wv, Wh))
        else:
            return np.ones((self.row * 2, self.col))

    def T_sub(self, G, Z, miu):
        X = G - Z / miu
        Xv = X[:self.row, :]
        Xh = X[self.row:, :]

        numerator = fft2(2 * self.T_esti + miu * (self.dv @ Xv + Xh @ self.dh))
        denominator = self.DTD * miu + 2
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

        for i in trange(0,self.iterations):
            T = self.T_sub(G, Z, miu)
            G = self.G_sub(T, Z, miu, self.W)
            Z = self.Z_sub(T, G, Z, miu)
            miu = self.miu_sub(miu)

        self.T = T ** self.gamma
        self.R = self.L / np.repeat(self.T[..., None], 3, axis = -1)
        return exposure.rescale_intensity(self.R, (0, 1)) * 255
