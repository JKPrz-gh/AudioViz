'''File containing preprocessing functions for audioViz project.'''

import numpy as np

def auto_thresh(input_matrix: np.array):
    # This is currently just a placeholder
    thresh_dB = -30

    # Preprocess for sound data 
    #threshold_curve = np.zeros(NF_DETECT_RESOLUTION)
    #for index, tsh_val in np.ndenumerate(np.linspace(mags_db.max(), mags_db.min(), num=NF_DETECT_RESOLUTION)):
    #    threshold_curve[index] = (mags_db > tsh_val).sum()

    #final_slope = np.abs(threshold_curve[-1] - threshold_curve[-2])    


if __name__ == "__main__":
    pass
