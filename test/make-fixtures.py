"""Builds a synthetic 71 s two-speaker test interview with fan noise, hum and bangs.

Usage: python3 test/make-fixtures.py <DeepFilterNet>/assets
(uses clean_freesound_33711.wav and noise_freesound_2530.wav from the DeepFilterNet repo)
"""
import numpy as np, wave, sys, subprocess
import os
A=sys.argv[1].rstrip('/')+'/'
OUT=os.path.join(os.path.dirname(os.path.abspath(__file__)),'fixtures','generated')+'/'
os.makedirs(OUT,exist_ok=True)
def rd(p):
    w=wave.open(p); n=w.getnframes(); ch=w.getnchannels(); fs=w.getframerate()
    x=np.frombuffer(w.readframes(n),dtype=np.int16).astype(np.float64)/32768
    return x.reshape(-1,ch).mean(axis=1), fs
sp,fs=rd(A+'clean_freesound_33711.wav')
nz,_=rd(A+'noise_freesound_2530.wav')
db=lambda d:10**(d/20)
rng=np.random.default_rng(1)
segs=[]
def pause(s): segs.append(np.zeros(int(s*fs)))
# host loud-ish, guest quiet, host again
for g in [0,-10,-2,-12,0,-6]:
    segs.append(sp*db(g)*0.5); pause(1.2)
x=np.concatenate(segs)
# noise bed (fan) at about -45 dBFS rms
bed=np.resize(nz,len(x)); bed=bed/np.sqrt(np.mean(bed**2))*db(-55)
x=x+bed
# hum 60 Hz + low rumble
t=np.arange(len(x))/fs
x+=db(-62)*np.sin(2*np.pi*60*t)
# spikes: a bang in a pause, a bump during quiet speech
def bang(at, amp, dur=0.08):
    i=int(at*fs); n=int(dur*fs)
    env=np.exp(-np.arange(n)/(0.015*fs))
    x[i:i+n]+=amp*env*rng.uniform(-1,1,n)
bang(11.0, 0.9)   # in the first pause
bang(14.0, 0.7)                # during the quiet guest
bang(40.0, 0.95, 0.15)
peak=np.max(np.abs(x)); print('peak',peak)
x=np.clip(x,-1,1)
def wr(p,y,ch=1,rate=fs,bits=24):
    y=np.repeat(y[:,None],ch,axis=1) if y.ndim==1 else y
    w=wave.open(p,'wb'); w.setnchannels(y.shape[1]); w.setframerate(rate); w.setsampwidth(bits//8)
    if bits==24:
        v=(np.clip(y,-1,1-1/8388608)*8388608).astype(np.int32).reshape(-1)
        b=np.stack([(v&255),(v>>8)&255,(v>>16)&255],axis=1).astype(np.uint8).tobytes()
    else:
        b=(np.clip(y,-1,1-1/32768)*32768).astype(np.int16).tobytes()
    w.writeframes(b); w.close()
wr(OUT+'interview-mono48-24.wav',x)
# stereo with host left, guest right-ish
st=np.stack([x, np.roll(x,37)*0.9],axis=1)
wr(OUT+'interview-stereo48-16.wav',st,bits=16)
print(len(x)/fs,'s')
