'''File containing preprocessing functions for audioViz project.'''

import numpy as np
import scipy.spatial

from settings import FUNDAMENTAL_DETECT_INTERVAL_MS as FDI


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


def fundamental_hunt(sound_data):
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
    # Extract the final time from the array
    # (which should be time sorted)

    initial_search_zone = sound_data[sound_data[:, 1] <= FDI/1000]
    start_point = sound_data[np.argmin(initial_search_zone[:,0]), :]

    # Find the next nearest point to that in the array
    nearest = sound_data[scipy.spatial.KDTree(sound_data).query(start_point)[1]]
    print(start_point)
    

if __name__ == "__main__":
    pass
