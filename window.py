#!/usr/bin/python3
"""
Demonstrates use of GLScatterPlotItem with rapidly-updating plots.
"""

import numpy as np

import pyqtgraph as pg
import pyqtgraph.opengl as gl
from pyqtgraph import functions as fn
from pyqtgraph.Qt import QtCore

# Create a qt app
app = pg.mkQApp("3D scatter plot")


def init():
    # Add an openGL view widget
    w = gl.GLViewWidget()
    w.show()
    w.setWindowTitle('pyqtgraph example: GLScatterPlotItem')
    w.setCameraPosition(distance=20)

    # Add an openGL grid
    g = gl.GLGridItem()
    w.addItem(g)

    # Create a 3d matrix of zeros to store the positions of particles
    # pos_matrix = np.zeros((no. matrices, no. rows per matrix, no. values per row()))
    pos_matrix = np.zeros((100, 100, 3))

    # Do some sort of magic??? TODO: Investigate me
    # This first part "targets" only the first 2 colums of each of our
    # 100 matrices and overwrites them with the output of our np.mgrid function

    # The np.mgrid.transpose creates 100 2 column 100 row arrays
    # With the first value fixed per array and second value incrementing through
    # each possible value: i.e
    # [0, 0]
    # [0, 1]
    # [0, ... ]
    # [0, 99]
    # Next array
    # [1, 0]
    # [1, 1] etc etc
    # This is used to create a flat grid on the "floor" of
    # the environment, which we scale and position with this [_______]
    # The third dimention is untouched
    pos_matrix[:, :, :2] = np.mgrid[:100, :100].transpose(1, 2, 0)

    # Turn data into a 10k x 3 matrix
    pos_matrix = pos_matrix.reshape(10000, 3) * [0.1, 0.1, 1]
    pos_matrix = np.concatenate((
        pos_matrix,
        pos_matrix * [-1, -1, 1],
        pos_matrix * [-1, 1, 1],
        pos_matrix * [1, -1, 1]
    ))

    # This is the distance of some point p from origin
    # (pythoagoras theorem)
    dist = (pos_matrix**2).sum(axis=1)**0.5

    # Create the scatter plot item
    scatter_plot = gl.GLScatterPlotItem(
        pos=pos_matrix,
        color=(1, 1, 1, .3),
        size=0.1,
        pxMode=False
    )

    # Add it to the view widget
    w.addItem(scatter_plot)
    return scatter_plot, pos_matrix, dist


scatter_plot, pos_matrix, dist = init()

# Declare phase as a float
phase = 0.


# Define this update function
def update():
    # update volume colors
    global phase
    phase -= 0.1

    # update surface positions and colors
    global scatter_plot, dist, pos_matrix
    z = -np.sin(dist*2+phase)
    pos_matrix[:, 2] = z
    color = np.empty((len(dist), 4), dtype=np.float32)
    color[:, 3] = 5
    color[:, 0] = np.clip(z * 3.0, 0, 1)
    color[:, 1] = np.clip(z * 1.0, 0, 1)
    color[:, 2] = np.clip(z ** 3, 0, 1)
    scatter_plot.setData(pos=pos_matrix, color=color)


# Set up a QTimer(part of qt)
t = QtCore.QTimer()

# Connect that timer to the update function
# This will a (duration)ms timer and run the update function every
# time that timer finishes. Since the timer is not in single
# shot mode, when the timer finishes up, it will start again
duration = 10
t.timeout.connect(update)
t.start(duration)


if __name__ == '__main__':
    pg.exec()
