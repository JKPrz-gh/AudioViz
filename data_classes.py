#!/usr/bin/python3
'''Classes for AudioViz.'''

from __future__ import annotations
import numpy as np
import numpy.typing as npt
import numpy.polynomial.polynomial as np_poly
import sklearn.cluster as skcl
import random
import typing


class ClusterGroupFactory:
    '''Class Defining a generic ClusteredDataSet'''
    def __init__(self,
                 sound_data: npt.NDArray,
                 dbs_params: typing.Tuple[float, float, float]
                 ) -> None:
        '''Constructor for ClusteredDataSet.

        This method loads in a dataset and breaks
        it out into clusters.'''
        # load in data and preprocess it
        self.sound_data = sound_data
        self.__preprocess(dbs_params)

        # Break out data into cluster objects
        self.__clusters: typing.List[typing.Union[None, PointCluster]] \
            = [None]*self.__labels.size
        self.__create_cluster_objects()

    def get_cluster_group(self) -> ClusterGroup:
        group = ClusterGroup(self.__clusters)
        return group

    # Private helper Methods
    def __preprocess(self,
                     dbs_params: typing.Tuple[float, float, float]
                     ) -> None:
        '''Threshold and cluster data'''
        # Unpack our input tuple
        prescale_x, eps, min_samples = dbs_params
        sd_copy = self.sound_data

        # Apply the prescale factor and cluster
        sd_copy[:, 1] *= prescale_x
        db = skcl.DBSCAN(eps=20, min_samples=10).fit(sd_copy)
        sd_copy[:, 1] /= prescale_x

        # Add labels as a mix-in
        label_array = db.labels_
        self.__labels = np.arange(0, label_array.max()+1, 1)

        sd_copy = np.vstack((sd_copy.transpose(), label_array)).transpose()
        self.sound_data = sd_copy[sd_copy[:, 3] != -1]

    def __create_cluster_objects(self) -> None:
        '''Populate the cluster array'''
        for index, _ in enumerate(self.__clusters):
            # Pull out the relevant points
            cluster_points = self.sound_data[self.sound_data[:, 3] == index]

            # When creating clusters, assign random colors initially
            cluster_color = f"#{random.randrange(0x1000000):06x}"

            # Create our objects
            point_cluster = PointCluster(
                points_data=cluster_points,
                cluster_id=index,
                color=cluster_color
            )

            self.__clusters[index] = point_cluster

    def __str__(self):
        outstring = f"Labels Array:{self.__labels}\n" \
            + f"with length {self.__labels.size}"

        return outstring


class ClusterGroup:
    '''Main Class for dealing with clusters.'''

    def __init__(self,
                 clusters: typing.List[typing.Union[None, PointCluster]]
                 ) -> None:
        self.clusters = clusters


class PointCluster:
    '''A Class defining a single data cluster.'''

    def __init__(self,
                 points_data: npt.NDArray,
                 cluster_id: int,
                 color: str,
                 ) -> None:
        '''Constructor class for a cluster of Points.'''
        self.times = points_data[:, 1]
        self.freqs = points_data[:, 0]
        self.mags_db = points_data[:, 2]

        self.id = cluster_id
        self.color = color
        self.no_points = self.times.size

        self.__populate_metadata()

    def interpolate_curve(
            self,
            points: int = 100,
            degree: int = 4
    ) -> typing.Tuple[
        npt.NDArray,
        npt.NDArray
    ]:
        '''Method to fit a curve to a data cluster.

        This method fits a polynomial of degree {degree}
        to the points in the cluster, and evaluates that
        at {points} points.

        Returns a Typle of evaluated x and y values.'''
        # create the initial curve
        curve = np_poly.Polynomial.fit(
            self.times,
            self.freqs,
            degree
        )
        # Eval that curve at desired number of points
        x, y = curve.linspace(
            n=points,
            domain=[
                self.start_time,
                self.end_time
            ]
        )
        return x, y

    def __populate_metadata(self) -> None:
        self.start_time = self.times.min()
        self.end_time = self.times.max()
        self.avg_freq = self.freqs.mean
