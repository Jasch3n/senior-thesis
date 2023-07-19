import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from PIL import Image

def radial_avg(z, n_bins, x0=0, y0=0):
    m,n = z.shape 
    assert m==n, "ERROR: Input to radial_avg must be square."
    N = m

    ucirc_coords = np.linspace(-1,1,N)
    X,Y = np.meshgrid(ucirc_coords, ucirc_coords)
    X -= x0 
    Y -= y0 
    r = np.sqrt(X**2 + Y**2)

    dr = 1/(n_bins-1)
    R = np.arange(0,1,dr)
    print(len(R))

    bins = (np.floor(r/dr) + 1).astype(int)
    print(bins.shape)
    bins = bins.reshape(N*N)
    z = z.reshape(N*N)

    counts = np.bincount(bins)
    radial_sum = np.bincount(bins, weights=z)

    radial_average = radial_sum / (counts+1e-6)
    radial_average[np.where(counts==0)] = 0

    return radial_average[0:n_bins-1], N*R

def power_spec_1D(im, n_bins=50):
    m,n = im.shape
    f_im = np.fft.fft2(im) / (m*n)
    f_im = np.fft.fftshift(f_im)
    plt.figure()
    plt.imshow(np.real(f_im))

    rad_avg, R = radial_avg(np.abs(f_im), n_bins)

    return rad_avg,R

if __name__ == '__main__':
    # x = np.linspace(-1,1,50)
    # y = np.linspace(-1,1,50)
    # X, Y = np.meshgrid(x,y)
    # f = 1 + np.cos(10*np.pi*Y) + np.sin(10*np.pi*X)
    max_size=(128,128)
    im = Image.open("sample-image.jpg").convert("L").resize(max_size)
    f = np.array(im)
    print("Mean f", np.mean(f))

    plt.figure()
    plt.imshow(f, cmap='gray')

    rad_avg,R = power_spec_1D(f, n_bins=30)
    plt.figure()
    plt.plot(np.log(R[1:]), np.log(rad_avg[1:]))
    plt.show()