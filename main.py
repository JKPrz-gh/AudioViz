#!/usr/bin/python3
'''Main file for AudioViz Project.

As this is intended to be released as a library,
this file exists for testing purposes and will be
removed from the final release.
'''

import numpy as np
import matplotlib.pyplot as plt
import data_classes
import plot_module
import librosa

import preprocess


def main() -> None:
    # Y is amplitude vector, sr is sample rate
    amplitudes, sample_rate = librosa.load(librosa.ex('trumpet'))
    #amplitudes, sample_rate = librosa.load(
    #    '/Users/janprzybyszewski/Documents/AudioViz/LongTrumpet.wav'
    #)
    # amplitudes = librosa.effects.harmonic(amplitudes, margin=1)

    times = np.zeros(len(amplitudes))
    for index, _ in enumerate(amplitudes):
        times[index] = (1/sample_rate)*index
    plt.figure()
    plt.plot(times, amplitudes, linewidth=0.25, color='mediumorchid')
    plt.xlim([0, 3.5])
    plt.grid(color='gray')
    plt.xlabel('Time (s)')
    plt.ylabel('Signal Amplitude')
    plt.title('Time Domain Signal')

    grapher = plot_module.PlotWriter()
    grapher.plot_spectrogram(amplitudes, sample_rate)
    
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

    plt.figure()
    plt.scatter(times, freqs, c=mags_db, s=0.3, cmap='gist_gray')
    plt.xlim([0, 3.5])
    plt.ylim([200, 3400])
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.title("Reassigned Spectrogram")
    plt.grid(color='gray')
    
    # Put this into a matrix instead
    sound_data = np.c_[freqs, times, mags_db]

    # Make sure our data is time sorted
    sound_data = sound_data[sound_data[:, 1].argsort()]

    print("Sound data matrix shape:", sound_data.shape)
    
    sound_data = preprocess.auto_thresh(sound_data)

    # Make sure our first data point is at time zero
    sound_data[:, 1] -= sound_data[0, 1]

    #plt.scatter(sound_data[:, 0], sound_data[:, 1], c=sound_data[:, 2], s=0.05, cmap='Greys')
    

    colorlist = [ "#358BAC", "#809AAF", "#FB7CA8", "#6935B3", "#3EBF70", "#0CCE8A",
        "#2847E0", "#C24606", "#559BB5", "#D0D658", "#C42C34", "#F77C34"
        ]
    
    ### ------------------------------------------------------------------------------------
    # prescale_x, eps, min_samples
    # Lower prescale values increase horizontal bias
    dbs_params = (220, 20, 10) # this should really be a dict or a custom object with a constructor

    cluster_factory = data_classes.ClusterGroupFactory(sound_data, dbs_params)
    cluster_set = cluster_factory.get_cluster_group()

    grapher.plot_scatterplot([cluster_set])

    plotlist = []

    # All curves
    grapher.plot_curves([cluster_set], 'Cluster Curves')

    index = 0
    while not cluster_set.is_empty():
        lowest_id = cluster_set.get_lowest_cluster_id()
        c_subset = cluster_set.get_coinciding_clusters(lowest_id)
        h_stack = c_subset.get_harmonic_stack(lowest_id)
        cluster_set = cluster_set ^ h_stack

        h_stack.set_color(colorlist[index % 12])

        plotlist.append(h_stack)
        index += 1

    # Xor Out curves
    grapher.plot_curves(plotlist, 'Harmonic Stacks')


if __name__ == "__main__":
    main()
