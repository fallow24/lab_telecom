import adi
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
import sys

import time

# Connect with the AdalmPluto

sdr = adi.Pluto('ip:192.168.2.1')

# Set the desired sampling rate

f_s_desired = 6000000 # 6ghz
sdr.sample_rate = f_s_desired
f_s = sdr.sample_rate
T_s = 1/f_s
print(f"Set the sample rate to {f_s/1e6} [MHz]")

# Set loopback mode of processing device to disable

sdr.loopback = 0 # 0: disable, 1: digital, 2: analog

# Local oscillator TX frequency

f_lo_tx_desired = 1009000000 # 1009 MHz
sdr.tx_lo = f_lo_tx_desired
f_lo_tx = sdr.tx_lo
print(f"Set the Tx LO frequency to {f_lo_tx/1e6} [MHz]")

# TX Bandwidth

bw_tx_desired = 5000000 # 5 MHz
sdr.tx_bandwidth = bw_tx_desired
bw_tx = sdr.tx_bandwidth
print(f"Set the Tx bandwidth to {bw_tx/1e6} [MHz]")

# Set to false since we only need one buffer 
sdr.tx_cyclic_buffer = False

# Set TX gain

gain_tx_desired = 0 #dB 
sdr.tx_hardwaregain_chan0 = gain_tx_desired
gain_tx = sdr.tx_hardwaregain_chan0
print(f"Set the Tx gain to {gain_tx} [dB]")

# Set RX frecuency

f_lo_rx_desired = 1200000000 # 1200 MHz
sdr.rx_lo = f_lo_rx_desired
f_lo_rx = sdr.rx_lo
print(f"Set the Rx LO frequency to {f_lo_rx/1e6} [MHz]")

# Set RX Bandwidth

bw_rx_desired = 5000000 # 5 MHz
sdr.rx_bandwidth = bw_rx_desired
bw_rx = sdr.rx_bandwidth
print(f"Set the Rx bandwidth to {bw_rx/1e6} [MHz]")

# Gain Controll Mode 

gain_control_mode = 'manual'
sdr.gain_control_mode_chan0 = gain_control_mode
gain_control_mode = sdr.gain_control_mode_chan0
sdr.rx_hardwaregain_chan0 = 60 # dB
print(f"Using Rx gain control mode {gain_control_mode}")

# Set Buffer size
# make the buffer extremly large since we want to
# capture the data with a minimum processing delay.
# Therefore, capture much data with (obviously) the same
# sampling time and measure the time difference between
# tx and rx using known sample time

buffer_size = 2 ** 9
sdr.rx_destroy_buffer()
sdr.rx_buffer_size = buffer_size
rxbuf = sdr.rx_buffer_size

# create cosine wave 

N_bits_tx = 12
f = 1000000 # 1 MHz 
print(f"Frequency {f / 1e6} MHz")
T = 2 # signal duration 2 seconds, minimum should be T = 1/f
t = np.arange(0,T,T_s)
A_max = (2**(N_bits_tx-1)-1)
A = 0.9 * A_max
def tx_signal_func(A, f, t):
    return A*np.cos(2*np.pi*f*t) + 1j*A*np.sin(2*np.pi*f*t)
tx_signal = tx_signal_func(A, f, t)

# Send signal wave 

sdr.tx_destroy_buffer()
sdr.rx_destroy_buffer()
buffer_data_list = []
n_buffers = 200
start_tx = time.time()
sdr.tx( tx_signal * 2**4 ) # maybe swap with destroy rx buffer if that is faster 
end_tx = time.time()
sdr.tx_destroy_buffer()


# get RX data, only one Buffer

ref = sdr.rx() # reference spectrum used for correlation
for k in range(n_buffers):
    buffer_data_list.append(sdr.rx())

end_rx = time.time()
step_duration = (end_rx - end_tx)/(n_buffers)

## Post processing of the buffer

i = 0
res = 0
# a signal has to be lost twice in order to be considered fully lost
lostonce = False
lost = False
# Compute the reference spectrum
ref_data = np.fft.fft(ref)/ref.size/2**11
ref_ampl = 10 * np.log10(np.fft.fftshift(np.abs(ref_data)**2))
print(f"REF AMPL: {ref_ampl[int(ref_ampl.size / 2)] }")

while i < n_buffers:
    fft_data = np.fft.fft(buffer_data_list[i])/buffer_data_list[i].size/2**11
    freqs = np.fft.fftfreq(fft_data.size, T_s)
    ampl = 10 * np.log10(np.fft.fftshift(np.abs(fft_data)**2))
    print(f"Amplitude at {i}: {ampl[int(ampl.size / 2)]}")
    plt.clf()
    plt.ylim([-100, 0])
    plt.plot(np.fft.fftshift(freqs), ampl)
    plt.draw()
    plt.pause(0.0001)
    print(f"Corr: {np.corrcoef(ampl, ref_ampl).item((0,1))}")
    
    # Amplitude method:
    # if the current amplitude is less than 10dB compared to the reference amplitude
    # we define to not see the signal anymore ( or it is too weak, so that means that 
    # the loopback device must be out of range )
    if ampl[int(ampl.size / 2)] < ref_ampl[int(ref_ampl.size / 2)] - 15 and not lost:
        if lostonce: 
            res = i
            lost = True
        lostonce = True
    else:
        lostonce = False
    # Correlation method:
    # if the correlation of the current spectrum correlated with the reference 
    # spectrum is smaller than 0.5 we define to have lost the signal.
    if np.corrcoef(ampl, ref_ampl).item((0,1)) < 0.5 and not lost:
        if lostonce: 
            res = i
            lost = True
        lostonce = True
    else:
        lostonce = False
    i += 1

if not lost: 
    print(f"The signal is stil there") 
else:
    print(f"After {res} buffers, the signal was lost. this is an overall duration of {step_duration*res}")

elapsed = step_duration * res

print(f"sdr.tx() time: {end_tx - start_tx}")
print(f"Start TX - End RX: {end_rx - start_tx}")
print(f"End TX - End RX: {end_rx - end_tx}")
print(f"Overall Signal Time: {end_tx - start_tx + elapsed}")
print(f"REAL Time: {end_tx - start_tx }")

del(sdr)





