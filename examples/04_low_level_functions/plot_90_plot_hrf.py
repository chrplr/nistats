"""
Example of hemodynamic reponse functions.
=========================================

We consider the hrf model in SPM together with the hrf shape proposed by
G.Glover, as well as their time and dispersion derivatives. 

Requires matplotlib

Author : Bertrand Thirion: 2009-2015
"""

import numpy as np
import matplotlib.pyplot as plt
from nistats import hemodynamic_models


#########################################################################
# We generate a single event ...
# 

onset, amplitude, duration = 0., 1., 1.

frame_times = np.linspace(0, 30, 61)
stim = np.zeros_like(frame_times)
stim[(frame_times > onset) * (frame_times <= onset + duration)] = amplitude
print(stim)

exp_condition = np.array((onset, duration, amplitude)).reshape(3, 1)
print(exp_condition)

#########################################################################
# ... and convolved it with the basis functions
#
#########################################################################
# Glover model 
# ------------

fig = plt.figure(figsize=(9, 4))

hrf_models = ['glover + derivative', 'glover + derivative + dispersion']

for i, hrf_model in enumerate(hrf_models):
    signal, name = hemodynamic_models.compute_regressor(
        exp_condition, hrf_model, frame_times, con_id='main',
        oversampling=16)

    plt.subplot(1, 2, i + 1)
    plt.fill(frame_times, stim, 'k', alpha=.5, label='stimulus')
    for j in range(signal.shape[1]):
        plt.plot(frame_times, signal.T[j], label=name[j])
    plt.xlabel('time (s)')
    plt.legend(loc=1)
    plt.title(hrf_model)

plt.subplots_adjust(bottom=.12)
plt.show()


#########################################################################
# SPM model 
# ------------

fig = plt.figure(figsize=(9, 4))

hrf_models = ['spm + derivative', 'spm + derivative + dispersion']

for i, hrf_model in enumerate(hrf_models):
    signal, name = hemodynamic_models.compute_regressor(
        exp_condition, hrf_model, frame_times, con_id='main',
        oversampling=16)

    plt.subplot(1, 2, i + 1)
    plt.fill(frame_times, stim, 'k', alpha=.5, label='stimulus')
    for j in range(signal.shape[1]):
        plt.plot(frame_times, signal.T[j], label=name[j])
    plt.xlabel('time (s)')
    plt.legend(loc=1)
    plt.title(hrf_model)

plt.subplots_adjust(bottom=.12)
plt.show()


#####################################################################
# FIR model
# ---------
# 

signal, name = hemodynamic_models.compute_regressor(
    exp_condition, 'fir', frame_times, con_id='main',
    fir_delays=[0, 2, 4, 6, 8, 10, 12, 14, 16, 18],
    oversampling=16)

plt.fill(frame_times, stim, 'k', alpha=.5, label='stimulus')
for j in range(signal.shape[1]):
    plt.plot(frame_times, signal.T[j], label=name[j])
plt.xlabel('time (s)')
plt.legend(loc=1)
plt.title(hrf_model)

plt.show()
