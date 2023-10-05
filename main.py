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

    freqs = np.ravel(freqs)
    times = np.ravel(times)
    mags_db = np.ravel(mags_db)

    #c=mags_db
    #plt.scatter(times, freqs, c=mags_db, s=0.05, cmap='Greys')
    plt = pg.plot(times, freqs, pen=None)
    pg.show



if __name__ == "__main__":
    main()
