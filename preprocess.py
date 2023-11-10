'''File containing preprocessing functions for audioViz project.'''

import numpy as np
import scipy.spatial
import sklearn.cluster as skcl

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


def cluster(sound_data):
    prescale_x = 100
    sound_data[:, 1] *= prescale_x
    db = skcl.DBSCAN(eps=20, min_samples=10).fit(sound_data)
    sound_data[:, 1] /= prescale_x
    labels = db.labels_
    
    
    sound_data = np.vstack((sound_data.transpose(), labels)).transpose()
    sound_data = sound_data[sound_data[:, 3] != -1]

    return sound_data


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
    
    # Front Search Zone
    initial_search_zone_front = sound_data[sound_data[:, 1] <= FDI/1000]
    start_point_front = sound_data[
        np.argmin(
            initial_search_zone_front[:, 0]
        ),
        :
    ]

    # Rear search Zone
    last_sample_time = sound_data[-1, 1]
    initial_search_zone_end = sound_data[sound_data[:, 1] >= (last_sample_time - (FDI/1000))]
    start_point_end = initial_search_zone_end[initial_search_zone_end[:, 0].argmin(), :]

    # Find the next nearest point to that in the array
    met_in_middle = False

    fundamental_array = np.empty((0, 3), int)

    
    while not met_in_middle:
        
        # Odd number case
        if (start_point_front == start_point_end).all():
            met_in_middle = True
        
        # Append start point to array
        fundamental_array = np.append(fundamental_array, [start_point_front], axis=0)
        fundamental_array = np.append(fundamental_array, [start_point_end], axis=0)

        # Remove original point from sound array
        sound_data = np.delete(sound_data, np.where(  (sound_data == start_point_front).all(axis=1)  ), 0)
        sound_data = np.delete(sound_data, np.where(  (sound_data == start_point_end).all(axis=1)  ), 0)

        # Find nearest point to start point
        nearest_forward = sound_data[scipy.spatial.KDTree(sound_data).query(start_point_front)[1]]
        nearest_rear = sound_data[scipy.spatial.KDTree(sound_data).query(start_point_end)[1]]

        # Even number case
        if (nearest_forward == start_point_end).all():
            met_in_middle = True

        # Swap point with nearest
        start_point_front = nearest_forward
        start_point_end = nearest_rear

    return fundamental_array


if __name__ == "__main__":
    pass
