from neuron import Neuron
import numpy as np
import matplotlib.pyplot as plt

# simple spiking
inj = np.zeros(1*10**5)
for i in range(1*10**3):
    inj[i+2000] = 20

n1 = Neuron(inj=inj)
n1.simulate(1*10**5)
f,a = n1.plot()
f.savefig('basic_spiking.pdf')

# sub-threshold
inj = np.zeros(1*10**5)
for i in range(1*10**3):
    inj[i+2000] = 1
n1 = Neuron(inj=inj)
n1.simulate(1*10**5)
f,a = n1.plot()
f.savefig('sub_thresh.pdf')

# repeated spiking
inj = np.zeros(1*10**5)
for i in range(8*10**4):
    inj[i+5000] = 20

n1 = Neuron(inj=inj)
n1.simulate(1*10**5)
f,a = n1.plot()
f.savefig('repeated_spiking.pdf')