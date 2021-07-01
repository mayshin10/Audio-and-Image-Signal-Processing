import cv2
import numpy as np
from matplotlib import pyplot as plt
import math

#PSNR metric
def psnr(original, contrast):
    mse = np.mean((original - contrast) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    PSNR = 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
    print(PSNR)
    return PSNR

#convolution in time domain
def conv2(original, kernel, kernel_size):
    num_pad = kernel_size // 2;
    pd = ((num_pad, num_pad), (num_pad, num_pad))
    pad_img = np.pad(original, pd, 'constant', constant_values=(0))
    conv_img = []
    length = len(original)
    for i in range(0, length):
        temp = []
        for j in range(0, length):
            sum = 0
            for m in range(0, kernel_size):
                for n in range(0, kernel_size):
                    sum+=pad_img[i+m][j+n] * kernel[m][n]
            temp.append(sum)
        conv_img.append(temp)
    return conv_img

#expand filter's kernel by zero-padding in time domain
def expand(kernel, img_size, kernel_size):
    filter_expand = []
    for row in range(0, img_size):
        temp = []
        for col in range(0, img_size):
            if row < kernel_size and col < kernel_size:
                temp.append(kernel[row][col])
            else:
                temp.append(0)
        filter_expand.append(temp)
    return filter_expand

#Image load
img = cv2.imread('cameraman.jpg',0)

#FFT original image
f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)

#Plot image, magnitude spectrum and phase spectrum before filtering
magnitude_spectrum = 20*np.log(1+np.abs(fshift))
phase_spectrum = np.angle(fshift)
plt.subplot(131),plt.imshow(img, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(132),plt.imshow(magnitude_spectrum, cmap = 'gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
plt.subplot(133),plt.imshow(phase_spectrum, cmap = 'gray')
plt.title('Phase Spectrum'), plt.xticks([]), plt.yticks([])
plt.show()

#H_lpf(wx, wy)
H_lpf = []
H_hpf = []
origin_row = (fshift.shape[0]-1) // 2
origin_col = (fshift.shape[1]-1) // 2
w_row = np.linspace(-np.pi, +np.pi, fshift.shape[0])
w_col = np.linspace(-np.pi, +np.pi, fshift.shape[1])

#make Ideal LPF and HPF
for i in range(0, fshift.shape[0]):
    temp_lpf = []
    temp_hpf = []
    for j in range(0, fshift.shape[1]):
        if np.sqrt(np.square(w_row[i] - w_row[origin_row]) + np.square(w_col[j] - w_col[origin_col])) <= (np.pi/4) :
            temp_lpf.append(1)
            temp_hpf.append(0)
        else:
            temp_lpf.append(0)
            temp_hpf.append(1)
    H_lpf.append(temp_lpf)
    H_hpf.append(temp_hpf)

#Plot Ideal LPF and HPF
plt.subplot(121),plt.imshow(H_lpf, cmap = 'gray')
plt.title('Ideal low pass filter'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(H_hpf, cmap = 'gray')
plt.title('Ideal high pass filter'), plt.xticks([]), plt.yticks([])
plt.show()

#Filtering by Ideal LPF and HPF
Y_lpf = fshift * H_lpf
Y_hpf = fshift * H_hpf
y_lpf = np.fft.ifft2(np.fft.ifftshift(Y_lpf)).real
y_hpf = np.fft.ifft2(np.fft.ifftshift(Y_hpf)).real
y_hpf = np.abs(y_hpf) / np.max(y_hpf)

plt.subplot(121),plt.imshow(y_lpf, cmap = 'gray')
plt.title('Low pass filtered img'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(y_hpf, cmap = 'gray')
plt.title('High pass filter img'), plt.xticks([]), plt.yticks([])
plt.show()

#3x3 laplacian filter
laplacian = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])

#Gausian filter with std.dev = 0.8, kernel size =3
sigma = 0.8
kernel_size = 3
gaussian = []
for i in range(0, kernel_size):
    temp = []
    for j in range(0, kernel_size):
        x = np.abs(i - (kernel_size // 2))
        y = np.abs(j - (kernel_size // 2))
        temp.append(1 / 2 / np.pi / sigma / sigma * np.exp(-(np.square(x) + np.square(y)) / 2 / sigma / sigma))
    gaussian.append(temp)

#filtering by laplacian and gaussian filter
Y_laplacian = np.fft.fftshift(np.fft.fft2(expand(laplacian, len(img), 3)))
Y_gaussian = np.fft.fftshift(np.fft.fft2(expand(gaussian, len(img), 3)))

#plot images filtered by laplacian and gaussain filter
plt.subplot(121), plt.imshow(20*np.log(1+np.abs(Y_laplacian)), cmap='gray')
plt.title('Laplacian magnitude spectrum'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(20*np.log(1+np.abs(Y_gaussian)), cmap='gray')
plt.title('Gaussian magnitude spectrum'), plt.xticks([]), plt.yticks([])
plt.show()

#filtering noised image by gaussian filter
noised_img = cv2.imread('noised_img.jpg',0)
noised_fshift = np.fft.fftshift(np.fft.fft2(noised_img))
Y_lpf_noised = noised_fshift * H_lpf
y_lpf_noised = np.fft.ifft2(np.fft.ifftshift(Y_lpf_noised)).real
y_gaussian_noised = conv2(noised_img, gaussian, 3)

#print PSNR score
psnr(img, y_lpf_noised)
psnr(img, y_gaussian_noised)

#compare Ideal LPF and Gaussian filter
plt.subplot(121), plt.imshow(y_lpf_noised, cmap='gray')
plt.title('Ideal LPF'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(y_gaussian_noised, cmap='gray')
plt.title('Gaussian filter'), plt.xticks([]), plt.yticks([])
plt.show()

#filtering by laplacian filter
Y_laplacian = np.fft.fftshift(np.fft.fft2(expand(laplacian, len(img), 3))) * fshift
y_laplacian = np.fft.ifft2(np.fft.ifftshift(Y_laplacian)).real
y_laplacian = np.abs(y_laplacian) / np.max(y_laplacian)

#compare Ideal HPF and Laplacian filter
plt.subplot(121),plt.imshow(y_hpf, cmap = 'gray')
plt.title('Ideal HPF'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(y_laplacian, cmap ='gray')
plt.title('Laplacian Filter'), plt.xticks([]), plt.yticks([])
plt.show()