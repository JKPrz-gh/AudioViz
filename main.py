#!/usr/bin/python3
'''Main file for AudioViz Project.

As this is intended to be released as a library,
this file exists for testing purposes and will be
removed from the final release.
'''

import numpy as np
import matplotlib.pyplot as plt
import pyqtgraph.exporters
import data_classes
import pyqtgraph as pg
import librosa

import preprocess


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

    # Make sure our data is time sorted
    sound_data = sound_data[sound_data[:, 1].argsort()]

    print("Sound data matrix shape:", sound_data.shape)
    
    sound_data = preprocess.auto_thresh(sound_data)

    # Make sure our first data point is at time zero
    sound_data[:, 1] -= sound_data[0, 1]

    #plt.scatter(sound_data[:, 0], sound_data[:, 1], c=sound_data[:, 2], s=0.05, cmap='Greys')
    #plt.scatter(times, freqs, c=mags_db, s=0.05, cmap='Greys')

    ### ------------------------------------------------------------------------------------
    # prescale_x, eps, min_samples
    # Lower prescale values increase horizontal bias
    dbs_params = (120, 20, 10) # this should really be a dict or a custom object with a constructor

    cluster_set = data_classes.ClusteredDataSet(sound_data, dbs_params)

    for cluster in cluster_set.clusters:
        # Make the type checker happy
        if cluster is None:
            continue

        plt.scatter(
            cluster.times,
            cluster.freqs,
            c=cluster.color,
            s=6
        )

        plt.annotate(
            f"{cluster.id}",
            (
                cluster.times[0],
                cluster.freqs[0]
            )
        )

    plt.grid()
    plt.show()

    test_cluster = cluster_set.clusters[5]
    assert test_cluster is not None
    plt.scatter(
        test_cluster.times,
        test_cluster.freqs,
        c=test_cluster.color,
        s=20
    )

    curve = test_cluster.fit_curve()
    print(curve)
    trange = np.linspace(test_cluster.start_time, test_cluster.end_time, num=100)
    tdata = np.polynomial.polynomial.polyval(trange, curve)
    plt.scatter(trange, tdata)

    plt.grid()
    plt.show()




if __name__ == "__main__":
    main()
