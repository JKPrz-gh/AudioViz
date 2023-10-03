#!/usr/bin/python3
'''Main file for AudioViz Project.

As this is intended to be released as a library,
this file exists for testing purposes and will be
removed from the final release.
'''

import numpy as np
import matplotlib.pyplot as plt
import librosa


def main() -> None:
    # Y is amplitude vector, sr is sample rate
    y, sr = librosa.load(librosa.ex('trumpet'))

    D = librosa.stft(y)  # STFT of y
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

    plt.figure()
    librosa.display.specshow(S_db)
    plt.colorbar()
    plt.show()


if __name__ == "__main__":
    main()
