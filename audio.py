import numpy as np
import librosa, librosa.display

import matplotlib.pyplot as plt
import sounddevice as sd
import soundfile as sf

#extract x[n]
data, samplerate = librosa.load('x[n].wav')   #16000 samplerate
times = np.arange(len(data))/float(samplerate)

#Listen original sound
# sd.play(data, samplerate)
# sd.wait()

#DFT of original sound
X = np.fft.fftshift(np.fft.fft(data))
w = np.linspace(-np.pi, +np.pi, len(X))

#Gaussian Noise
v = np.random.normal(0, np.sqrt(0.02), size=data.shape)
V = np.fft.fftshift(np.fft.fft(v))

#Listen gaussian noised sound
# sd.play(data + v, samplerate)
# sd.wait()

#save sounds
sf.write('gaussian_noise_only.wav', v, samplerate, 'PCM_16')
sf.write('gaussian_noise_with_original_sound.wav', data+v, samplerate, 'PCM_16')

#plot x[n]
plt.figure('x[n], |X(w)| and v[n]')
plt.subplot(311)
plt.plot(times, data)
plt.xlabel('time (s)')
plt.ylabel('amplitude')

#plot |X(w)|
plt.subplot(312)
plt.plot(w, np.log(1+np.abs(X)))
plt.xlabel('w')
plt.ylabel('log(1+magnitude)')

#plot v[n]
plt.subplot(313)
plt.plot(times, v)
plt.xlabel('time (s)')
plt.ylabel('amplitude')
plt.show()

#truncated filter
N = 39
n = np.arange(0, N)
h_d = []
for i in n:
    if i==0:
        h_d.append(1/4)
    else:
        h_d.append(1 / 4 * np.sin(np.pi / 4 * i) / np.pi * 4 / i)
h_d = np.pad(h_d, (0, len(data)-N), 'constant', constant_values=0)

#DFT of h_d
H_d = np.fft.fftshift(np.fft.fft(h_d))

#plot Re(H_d(w))
plt.figure('Real part of H_d(w)')
plt.plot(w, np.real(H_d))
plt.xlabel('w')
plt.ylabel('Re(H_d)')
plt.show()

#Hanning windowing, wc = pi/2
h = []
for i in n:
    if i == (N-1)/2:
        h.append(1/2 * 0.5*(1- np.cos(2*np.pi/(N-1) * i)))
    else :
        h.append(1/2 * np.sin(np.pi / 2 * (i - (N-1)/2))/np.pi*2/(i - (N-1)/2) * 0.5*(1- np.cos(2*np.pi/(N-1) * i)))

h = np.pad(h, (0, len(data)-N), 'constant', constant_values=0)

#DFT of h
H = np.fft.fftshift(np.fft.fft(h))

#plot h[n]
plt.figure('h[n] and |H(w)|')
plt.subplot(211)
plt.plot(times, h)
plt.xlabel('time (s)')
plt.ylabel('amplitude')

#plot |H(w)|
plt.subplot(212)
plt.plot(w, np.log(1+np.abs(H)))
plt.xlabel('w')
plt.ylabel('log(1+magnitude)')
plt.show()

#Filtering gaussian noise
V_f = H * V

#IFFT to get filtered noise
v_f= np.fft.ifft(np.fft.ifftshift(V_f)).real

#x_d[n] and X_d(w)
x_d = data + v_f
X_d = np.fft.fftshift(np.fft.fft(x_d))

#generate x_d[n]
# sd.play(x_d, samplerate)
# sd.wait()
sf.write('sound filtered by Hanning Window.wav', x_d, samplerate, 'PCM_16')

#plot v_f[n]
plt.figure('v_f[n] and |V_f(w)|')
plt.subplot(211)
plt.plot(times, v_f)
plt.xlabel('time (s)')
plt.ylabel('amplitude')

#plot |V_f(w)|
plt.subplot(212)
plt.plot(w, np.log(1+np.abs(V_f)))
plt.xlabel('w')
plt.ylabel('log(1+magnitude)')
plt.show()

#My own filter design
b = 0.3069
r = 0.6
theta = 0.5
H2 = b / ((1- r * np.exp(1j*(theta - w)))*(1-r*np.exp(-1j*(theta+w))))

# Filtering signal
X_d2 = X_d * H2
x_d2 = np.fft.ifft(np.fft.ifftshift(X_d2)).real

#evalute
print(np.sqrt(np.sum(np.square(v_f)))) # 1-4 , hanning windowing
print(np.sqrt(np.sum(np.square(data - x_d2)))) # 1-5, my own filter

# sd.play(x_d2, samplerate)
# sd.wait()
sf.write('my_filter.wav', x_d2, samplerate, 'PCM_16')

#plot |H2(w)|
plt.figure('|H2(w)| and <H2(w)')
plt.subplot(211)
plt.plot(w, np.abs(H2))
plt.xlabel('w')
plt.ylabel('log(1+magnitude)')


#plot <H2(w)
plt.subplot(212)
plt.plot(w, np.angle(H2))
plt.xlabel('w')
plt.ylabel('phase')
plt.show()