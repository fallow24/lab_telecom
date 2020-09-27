## This script measures the time needed for 
## the sdr.tx( signal ) operation from the Python Pluto 
## SDR API and writes it to a file called "periods_tx.txt".
## A 1 MHz signal is used.

import adi
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
import sys

import time

sdr = adi.Pluto('ip:192.168.2.1')

f_s_desired = 6000000 # 6ghz
sdr.sample_rate = f_s_desired
f_s = sdr.sample_rate
T_s = 1/f_s
print(f"Set the sample rate to {f_s/1e6} [MHz]")

sdr.loopback = 0 # 0: disable, 1: digital, 2: analog

# Setting up LO TX and LO RX is not necessary since this script only
# measures the time needed for one .tx() operation

# create cosine wave signal (send it 2 seconds for reference)

N_bits_tx = 12
f = 1000000 # 1 MHz 
print(f"Signal Duration {1 / f} s")
T = 1/f # signal duration 2 seconds, minimum should be T = 1/f
t = np.arange(0,T,T_s)
A_max = (2**(N_bits_tx-1)-1)
A = 0.9 * A_max
def tx_signal_func(A, f, t):
    return A*np.cos(2*np.pi*f*t) + 1j*A*np.sin(2*np.pi*f*t)
tx_signal = tx_signal_func(A, f, t)

n = 1000 # how many measurements 
file = open("periods_tx.txt", "w")
for i in range(n):
    sdr.tx_destroy_buffer()
    start = time.time()
    sdr.tx( tx_signal * 2 **4 )
    end = time.time()
    file.write(str(end - start))
    file.write("\n")
file.close()

