#!/usr/bin/python3
'''Main file for AudioViz Project.

As this is intended to be released as a library,
this file exists for testing purposes and will be
removed from the final release.
'''

import numpy as np
import matplotlib.pyplot as plt
import pyqtgraph.exporters
import pyqtgraph as pg
import librosa

NF_DETECT_RESOLUTION = 512

def main() -> None:
    # Y is amplitude vector, sr is sample rate
    amplitudes, sample_rate = librosa.load(librosa.ex('trumpet'))

    # Want to use librosa's reassigned spectrograms instead
    freqs, times, mags = librosa.reassigned_spectrogram(
        amplitudes,
        sr=sample_rate,
        n_fft=4086
    )

    mags_db = librosa.amplitude_to_db(np.abs(mags), ref=np.max)

    # These are effectively x and y and z co-ordinates
    freqs = np.ravel(freqs)
    times = np.ravel(times)
    mags_db = np.ravel(mags_db)

    # Put this into a matrix instead
    sound_data = np.c_[freqs, times, mags_db]

    # Preprocess for sound data 
    threshold_curve = np.zeros(NF_DETECT_RESOLUTION)
    for index, tsh_val in np.ndenumerate(np.linspace(mags_db.max(), mags_db.min(), num=NF_DETECT_RESOLUTION)):
        threshold_curve[index] = (mags_db > tsh_val).sum()

    final_slope = np.abs(threshold_curve[-1] - threshold_curve[-2])
    

    plt.plot(threshold_curve)
    

    #plt.scatter(times, freqs, c=mags_db, s=0.05, cmap='Greys')
    plt.show()



if __name__ == "__main__":
    main()
