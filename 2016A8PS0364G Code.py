#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Importing all necessary libraries
from brian2 import *

import keras
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from keras.datasets import mnist
import brian2tools
from PIL import Image
from brian2tools import *

get_ipython().run_line_magic('matplotlib', 'inline')


# # Extracting Spike frequencies from images

# In[3]:


def Make_pix_to_spike_dictionary():
    #array containing all pixel intensities from 0 to 255
    pix=range(0,256,1)
    #spike_freq list to store corresponding frequency of the spike train for that pixel intensity value
    spike_freq=[]
    #Initialize frequency=5.1 for Pixel intensity of value 0
    init_spike_freq=5.1
    count=0
    for i in pix:
        if(count==5):
            #Count is used for increasing value of freq after every group of 5 pixel intensities
            count=0
            init_spike_freq+=round(0.1,1)

        count+=1
        spike_freq.append(round(init_spike_freq,1))

    #Linear plot of
    #fig = plt.figure()
    #ax = fig.add_axes([1,1,1,1])
    keys = pix
    values = spike_freq
    pix_to_spk_dict = dict(zip(keys, values))
    #plt.plot(pix,spike_freq)

    #print(pix_to_spk_dict)
    return pix_to_spk_dict

    #plt.savefig('Spike frequencies Vs Pixel intensity values.png', bbox_inches='tight')


# In[4]:


pix_to_spk_dict=Make_pix_to_spike_dictionary()
print(pix_to_spk_dict)


# In[5]:


def load_mnist_dataset():
    (train_x, train_y) , (test_x, test_y) = mnist.load_data()
    return (train_x,train_y),((test_x, test_y))


# In[6]:


(train_x,train_y),((test_x, test_y))=load_mnist_dataset()
train_x.shape


# In[7]:


def get_image_pixels(index):
    load_mnist_dataset()
    img=test_x[index]
    test_img = img.reshape(1,784)
    return test_img,img


# In[8]:


test_img,img=get_image_pixels(120)
print(img.shape)
mat = np.reshape(img,(28,28))

# Creates PIL image
img = Image.fromarray( mat , 'L')
plt.imshow(img)
plt.title('my picture')
plt.show()
print(test_img.shape)


# In[9]:


arr,img=get_image_pixels(120)
print(arr)


# In[10]:


arr.shape


# In[11]:


def get_spike_freqs_from_image(index):
    arr,img=get_image_pixels(index)
    spike_freqs=[]
    pix_to_spk_dict=Make_pix_to_spike_dictionary()
    #print(pix_to_spk_dict)
    #print(type(pix_to_spk_dict))
    for i in arr[0]:
        if(pix_to_spk_dict.get(i)):
            spike_freqs.append(pix_to_spk_dict.get(i))
    return spike_freqs


# In[12]:


spike_freqs_img_120=get_spike_freqs_from_image(120)
print(spike_freqs_img_120)


# # LIF neuron simulation

# In[13]:


import numpy as np
import pandas as pd
import brian2 as b2


# In[14]:


import os
import re

from brian2 import *


# In[15]:


tau = 2 * msecond        # membrane time constant
Vt = -52 * mvolt          # spike threshold
Vr = -65 * mvolt          # reset value
El = -65 * mvolt          # resting potential (same as the reset)


# In[16]:


eqs = '''dv/dt = -(v-El)/tau : volt'''
group = NeuronGroup(28*28, eqs,
                    threshold='v > -50*mV',
                    reset='v = -70*mV')
M = StateMonitor(group, 'v', record=True)


# In[17]:


# #By default the time step is 0.1ms
 run(1 * second)


# In[18]:


plot(M.t/ms, M.v[0])
print(len(M.t/ms))
xlabel('Time (ms)')
ylabel('v');


# # Creating the Input layer

# In[19]:


def Plot_raster_plot(spikemonitor):
    plot(spikemonitor.t/ms, spikemonitor.i, '.k')
    xlabel('Time (ms)')
    ylabel('Neuron index');


# In[20]:


# plot(M.t/ms, M.v[0])


# In[21]:


start_scope()
a=spike_freqs_img_120*Hz
P = PoissonGroup(784,a)
W=5*siemens
eqs1 = '''dv/dt = -(v-El)/tau : volt'''

neuron=NeuronGroup(100,eqs1,threshold='v>Vt',reset='v=Vr',method='exact')
neuron.v=Vr
synapses=Synapses(P,neuron,'w: siemens')
synapses.connect()
synapses.w=W

M = SpikeMonitor(P)
N=SpikeMonitor(neuron)
run(1*second)
brian_plot(M)


# In[23]:


print(synapses.w)


# In[24]:


test_img,img=get_image_pixels(120)
mat = np.reshape(img,(28,28))

# Creates PIL image
img = Image.fromarray( mat , 'L')
plt.imshow(img)
plt.title('MNIST Image')
plt.show()


# In[25]:


spike_freqs_img_120=get_spike_freqs_from_image(120)
print(spike_freqs_img_120)


# In[ ]:


start_scope()
a=spike_freqs_img_120*Hz
P = PoissonGroup(784,20*Hz)
refrac_e = 1. * ms

eqs = '''
        dv/dt = ((v_rest_e - v) + (I_synE+I_synI) / nS) / (2*ms)  : volt (unless refractory)
        I_synE = ge * nS *         -v                           : amp
        I_synI = gi * nS * (-70.*mV-v)                          : amp
        dge/dt = -ge/(1.0*ms)                                   : 1
        dgi/dt = -gi/(2.0*ms)                                  : 1
    '''

#neuron=NeuronGroup(1,eqs,threshold='v>Vt',reset='v=Vr',method='exact')
neuron= NeuronGroup(1, eqs, threshold= 'v>Vt',refractory=refrac_e, reset= 'v=Vr', method='euler')
neuron.v=Vr
synapses=Synapses(P,neuron,'w: siemens')
synapses.connect(i=0,j=0)
synapses.w = 25*nS
M = SpikeMonitor(P)
N=SpikeMonitor(neuron)
run(5*second)
brian_plot(M)


# In[ ]:


synapses.w


# In[ ]:


print(synapses.i[:])
print(synapses.t[:])


# In[ ]:


brian_plot(N)


# In[27]:


v_rest_e = -65. * mV
v_rest_i = -60. * mV
v_reset_e = -65. * mV
v_reset_i = -45. * mV
v_thresh_e = -52. * mV
v_thresh_i = -40. * mV
refrac_e = 5. * ms
refrac_i = 2. * ms


# In[28]:


no_exci=400
no_inhi=no_exci

v_reset_e = 'v = v_reset_e; theta += theta_plus_e; timer = 0*ms'

v_thresh_e_eqn = '(v>(theta - offset + v_thresh_e)) and (timer>refrac_e)'
v_thresh_i_eqn = 'v>v_thresh_i'
v_reset_i_eqn = 'v=v_reset_i'


# In[29]:


exc_neurons = '''
        dv/dt = ((v_rest_e - v) + (I_synE+I_synI) / nS) / (100*ms)  : volt (unless refractory)
        I_synE = ge * nS *         -v                           : amp
        I_synI = gi * nS * (-100.*mV-v)                          : amp
        dge/dt = -ge/(1.0*ms)                                   : 1
        dgi/dt = -gi/(2.0*ms)                                  : 1
        '''


# In[30]:


inh_neurons = '''
        dv/dt = ((v_rest_i - v) + (I_synE+I_synI) / nS) / (10*ms)  : volt (unless refractory)
        I_synE = ge * nS *         -v                           : amp
        I_synI = gi * nS * (-85.*mV-v)                          : amp
        dge/dt = -ge/(1.0*ms)                                   : 1
        dgi/dt = -gi/(2.0*ms)                                  : 1
        '''


# In[ ]:


neuron_groups['e'] = NeuronGroup(no_exci*len(population_names), exc_neurons, threshold= v_thresh_e_eqn, refractory= refrac_e, reset= v_reset_e, method='euler')


# In[ ]:


neuron_groups['i'] = NeuronGroup(no_inhi*len(population_names), inh_neurons, threshold= v_thresh_i_eqn, refractory= refrac_i, reset= v_reset_i_eqn, method='euler')


# In[ ]:


pre = 'Apost += w'
model='''w:1
         dApre/dt=-Apre/taupre : 1 (event-driven)
         dApost/dt=-Apost/taupost : 1 (event-driven)'''



ei_connection=Synapses(neuron_groups['e'], neuron_groups['i'], model=model, on_pre=pre)


# In[ ]:


for i in range(no_exci):
    ei_connection.connect(neuron_groups[i][0],neuron_groups[i][0])


# In[ ]:




