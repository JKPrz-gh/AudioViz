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
from matplotlib.ticker import EngFormatter


def main() -> None:
    grapher = plot_module.MplPlotDisplay()

    audio_file = data_classes.AudioFile(
        filepath='/Users/janprzybyszewski/Documents/AudioViz/Audio/ViolinDuet.wav'
    )

    # Lower prescale values increase horizontal bias
    prescale_x = 60

    cluster_factory = data_classes.ClusterGroupFactory(audio_file.get_reassigned_points(), prescale_x)
    cluster_set = cluster_factory.get_cluster_group()
    
    grapher.plot_scatterplot([cluster_set])
    
    # Xor Out curves
    plotlist = cluster_set.get_harmonic_stack_groups()
    grapher.plot_curves(plotlist, 'Harmonic Stacks')


if __name__ == "__main__":
    main()
