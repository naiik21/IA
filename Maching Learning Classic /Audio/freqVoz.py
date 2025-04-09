import json
import numpy as np #aqui hi ha les estructures de dades que faig servir
import matplotlib.pyplot as plt #per dibuixar
import scipy.io as sio #el input output d'audio (wavfile.read)
from IPython.display import Audio #el reproductor d'audio
from numpy.fft import fft, ifft #podria fer numpy.fft en comptes de fer aquest import
from wav2vec import cutvowel, wav2vec #el nostre modul



with open("./vowels/jordi.json") as f:
    data = json.load(f)

start = data[30]["start"]
end = data[30]["end"]

select = 10
start = float(data[select]["start"])
end=float(data[select]["end"])

vowel= data[select]["vocal"]



Fs, audio = sio.wavfile.read("vowels/jordi.wav")
cut = audio[int(start*Fs):int(end*Fs)]

fourier = fft(cut)

Fsmall = fourier[0:300]

toprocess = np.sqrt((np.real(Fsmall)**2+np.imag(Fsmall)**2))
plt.figure()
plt.plot(toprocess)
plt.show()



