"""
Second level analysis: one group of subjects and a single contrast per subject
==============================================================================

16 participants had to press either the right or the left button in a fast-event paradigm. 
The individual effect maps contrasting between left button press and right button press were computed 
in first level analyses.

Here, these contrast maps are smoothed at 8mm and entered in a second level, group analysis --- which simply corresponds to a single sample t-test.
(The design matrix is just a constant vector of ones). 

Author : Martin Perez-Guevara: 2016
"""

from scipy.stats import norm


#########################################################################
# Fetch dataset
# --------------
# We download a list of left vs right button press contrasts from a
# localizer dataset.


n_subjects = 16
from nilearn.datasets import fetch_localizer_contrasts
data = fetch_localizer_contrasts(["left vs right button press"], n_subjects,
                                 get_tmaps=True)
print(data.keys())

individual_tmaps = data['tmaps']
individual_contrasts = data['cmaps']
sub_id = [subject_data[0] for subject_data in data['ext_vars']]


###########################################################################
# Display subject t_maps
# ----------------------
# We plot each subject's t-map thresholded at t = 2 for
# simple visualization purposes. The button press effect is visible among
# all subjects

import matplotlib.pyplot as plt
from nilearn import plotting

fig, axes = plt.subplots(nrows=4, ncols=4)
for cidx, tmap in enumerate(individual_tmaps):
    plotting.plot_glass_brain(tmap, colorbar=False, threshold=2.0,
                              title=sub_id[cidx],
                              axes=axes[int(cidx / 4), int(cidx % 4)],
                              plot_abs=False, display_mode='z')
fig.suptitle('subjects t_map left-right button press')
plt.show()

############################################################################
# Generation and estimation of the GLM 
# ------------------------------------
# We define the input maps and the design matrix for the second level model
# and fit it.

# The design matrix is simply a column of '1' with one row per subject

import pandas as pd
import numpy as np
design_matrix = pd.DataFrame(np.ones(n_subjects),
                             columns=['left-right'])


# We estimate the GLM
from nistats.second_level_model import SecondLevelModel

# note that we specify the spatial smoothing of contrasts here
# we could have specied an explicit mask. Here an implicit one is computed.
second_level_model = SecondLevelModel(smoothing_fwhm=8.0)
second_level_model = second_level_model.fit(individual_contrasts,
                                            design_matrix=design_matrix)

##########################################################################
# Now, we can assess the effect by the simple contrast '1'.
# We can  provide the column name of the design matrix.

z_map = second_level_model.compute_contrast('left-right',
                                            output_type='z_score')

###########################################################################
# We threshold the second level contrast at uncorrected p < 0.001 and plot
p_val = 0.001
z_th = norm.isf(p_val)
display = plotting.plot_glass_brain(z_map, threshold=z_th, colorbar=True,
                                    plot_abs=False, display_mode='z')

plotting.show()
