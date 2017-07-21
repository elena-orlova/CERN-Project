import numpy as np
import scipy as sp

def osc(x, y, z, kx, ky, kz, cx, cy, cz):
    return 0.5 * (kx * (x - cx)**2 + ky * (y - cy)**2 + + kz * (z - cz)**2)

def gen_rhs(size):
    features = np.zeros((size, 25, 25, 25, 1))
    labels = np.zeros(size)
    x = np.linspace(-20, 20, 25)
    xx, yy = np.meshgrid(x, x)
    yy, zz = np.meshgrid(x, x)

    for i in range(size):
        cx = -8.0 + 16.0 * np.random.rand()
        cy = -8.0 + 16.0 * np.random.rand()
        cz = -8.0 + 16.0 * np.random.rand()
        kx = 0.16 * np.random.rand()
        ky = 0.16 * np.random.rand()
        kz = 0.16 * np.random.rand()
        features[i, :, :, :, 0] = osc(xx, yy, zz, kx, ky, kz, cx, cy, cz)
        #labels[i] = 0.5 * (np.sqrt(kx) + np.sqrt(ky) + np.sqrt(kz))

    return features.reshape((size, 25 * 25 * 25)).astype(np.float32), labels.astype(np.float32)
