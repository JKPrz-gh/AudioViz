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
import cluster_grapher
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

    colorlist = ['#FF3333', '#3333FF', '#33FF33', '#33AAAA', '#AA33AA', '#AAAA33', '#3F3FAA']
    
    ### ------------------------------------------------------------------------------------
    # prescale_x, eps, min_samples
    # Lower prescale values increase horizontal bias
    dbs_params = (220, 20, 10) # this should really be a dict or a custom object with a constructor

    cluster_factory = data_classes.ClusterGroupFactory(sound_data, dbs_params)
    cluster_set = cluster_factory.get_cluster_group()
    grapher = cluster_grapher.ClusterGrapher()

    # Just the cluster data
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

    plt.title("All clusters")
    plt.grid()
    plt.show()

    # Maybe show a single fitting?
    plotlist = []

    # All curves
    grapher.plot_curves([cluster_set], 'Cluster Curves')

    index = 0
    while not cluster_set.is_empty():
        lowest_id = cluster_set.get_lowest_cluster_id()
        c_subset = cluster_set.get_coinciding_clusters(lowest_id)
        h_stack = c_subset.get_harmonic_stack(lowest_id)
        cluster_set = cluster_set ^ h_stack

        h_stack.set_color(colorlist[index % 7])

        plotlist.append(h_stack)
        index += 1

    # Xor Out curves
    grapher.plot_curves(plotlist, 'Harmonic Stacks')


if __name__ == "__main__":
    main()
