import numpy as np
from scipy.fft import *
from skimage import exposure
import cv2
from tqdm import trange

class LIME:
    # 初始化参数
    def __init__(self, iterations, alpha, rho, gamma, strategy, exact):
        self.iterations = iterations
        self.alpha = alpha
        self.rho = rho
        self.gamma = gamma
        self.strategy = strategy
        self.exact = exact

    # 加载图片，并归一化到 0~1 内
    def load(self, imgPath):
        self.loadimage(cv2.imread(imgPath) / 255)

    # 初始化Dx,Dy,DTD
    def loadimage(self,L):
        self.L = L
        self.row = self.L.shape[0]
        self.col = self.L.shape[1]

        self.T_esti = np.max(self.L, axis=2)
        self.Dv = -np.eye(self.row) + np.eye(self.row, k=1)
        self.Dh = -np.eye(self.col) + np.eye(self.col, k=-1)

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

    # 这里使用策略2
    def Strategy(self):
        if self.strategy == 2:
            self.Wv = 1 / (np.abs(self.Dv @ self.T_esti) + 1)
            self.Wh = 1 / (np.abs(self.T_esti @ self.Dh) + 1)
            print(self.Wv.shape,self.Wh.shape)
            exit()
            return np.vstack((self.Wv, self.Wh))
        else:
            return np.ones((self.row * 2, self.col))

    # T子问题
    def T_sub(self, G, Z, miu):
        X = G - Z / miu
        Xv = X[:self.row, :]
        Xh = X[self.row:, :]

        numerator = fft2(2 * self.T_esti + miu * (self.Dv @ Xv + Xh @ self.Dh))
        denominator = self.DTD * miu + 2
        T = np.real(ifft2(numerator / denominator))

        return exposure.rescale_intensity(T, (0, 1), (0.001, 1))

    # G子问题
    def G_sub(self, T, Z, miu, W):
        epsilon = self.alpha * W / miu
        temp = np.vstack((self.Dv @ T, T @ self.Dh)) + Z / miu
        return np.sign(temp) * np.maximum(np.abs(temp) - epsilon, 0)

    # Z子问题
    def Z_sub(self, T, G, Z, miu):
        return Z + miu * (np.vstack((self.Dv @ T, T @ self.Dh)) - G)

    # miu子问题
    def miu_sub(self, miu):
        return miu * self.rho

    def run(self):
        # 精确算法
        if self.exact:
            # exact algorithm
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
            return exposure.rescale_intensity(self.R,(0,1)) * 255
        # 快速算法
        else:
            pass
