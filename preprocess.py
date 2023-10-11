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


def fundamental_hunt():
    '''Funtion for fundamental_hunt

    Hunt for fundamental frequency points in 25ms
    intervals up through thefrequency spectrum,
    and drop points that probably shouldn't be 
    in the dataset.

    Plan for this is to extract fundamental
    from dataset, and use fundamental freq to
    extract harmonics of that fundamental,

    repeat per voice present in sample
    '''
    pass

if __name__ == "__main__":
    pass
