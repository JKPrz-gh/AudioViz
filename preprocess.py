'''File containing preprocessing functions for audioViz project.'''

import numpy as np


def auto_thresh(input_matrix):
    # This is currently just a placeholder

    print("Loading in input matrix with following dimentions:", input_matrix.shape)
    thresh_dB = -20

    # Preprocess for sound data 
    #threshold_curve = np.zeros(NF_DETECT_RESOLUTION)
    #for index, tsh_val in np.ndenumerate(np.linspace(mags_db.max(), mags_db.min(), num=NF_DETECT_RESOLUTION)):
    #    threshold_curve[index] = (mags_db > tsh_val).sum()

    #final_slope = np.abs(threshold_curve[-1] - threshold_curve[-2])    

    output_matrix = input_matrix[input_matrix[:, 2] >= thresh_dB]

    print("Dumping matrix with following dimentions:", output_matrix.shape)
    return output_matrix


if __name__ == "__main__":
    pass
