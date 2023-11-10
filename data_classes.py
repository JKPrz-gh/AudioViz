#!/usr/bin/python3
'''Classes for AudioViz.'''

from __future__ import annotations
import numpy as np
import numpy.typing as npt
import numpy.polynomial.polynomial as np_poly
import sklearn.cluster as skcl
import random
import math
import re
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
        '''Construct an instance of a Clustergroup.'''
        self.clusters = clusters

        # At this point, we can remove None Clusters
        for idx, cluster in enumerate(self.clusters):
            if cluster is None:
                del self.clusters[idx]

    def is_empty(self) -> bool:
        if len(self.clusters) == 0:
            return True

        else:
            return False

    def has_member(self, input: typing.Union[int, PointCluster]):
        '''Check if given cluster is member of group.

        This function can be provided a cluster ID or an
        actual cluster,'''
        if input is int:
            input_id = input
        else:
            if isinstance(input, PointCluster):
                input_id = input.id

            else:
                raise TypeError

        for cluster in self.clusters:
            if input_id == cluster.id:
                return True

        return False

    def get_cluster_with_id(self, id: int) -> typing.Union[None, PointCluster]:
        for cluster in self.clusters:
            assert cluster is not None
            if cluster.id == id:
                return cluster

        return None

    def get_lowest_cluster_id(self) -> int:
        '''Function to get the cluster with the lowest frequency.

        Returns the ID of that cluster as an int.'''
        closest_cluster = self.clusters[0]
        assert closest_cluster is not None
        for cluster in self.clusters:
            # Cluster should never be none here
            assert cluster is not None

            if cluster.avg_freq < closest_cluster.avg_freq:
                closest_cluster = cluster

        print(f"Lowest cluster with id {closest_cluster.id}")
        return closest_cluster.id

    def get_coinciding_clusters(self,
                                reference_in: typing.Union[int, PointCluster],
                                tolerance_ms: float = 50) -> ClusterGroup:
        '''Method to get time grouped subset cluster'''
        tolerance_ms /= 1000

        reference = self.__resolve_inputs(reference_in)

        outlist = []
        for cluster in self.clusters:
            assert cluster is not None
            assert reference is not None

            # Overlap to the left
            if cluster.end_time >= (reference.start_time - tolerance_ms) and \
               cluster.end_time <= (reference.end_time + tolerance_ms):
                outlist.append(cluster)
                # Contnue to avoid duplicate clusters
                continue

            # Overlap to the right
            if cluster.start_time <= (reference.end_time + tolerance_ms) and \
               cluster.start_time >= (reference.start_time - tolerance_ms):
                outlist.append(cluster)
                continue

            # Overlap both sides
            if cluster.start_time <= (reference.start_time + tolerance_ms) and \
               cluster.end_time >= (reference.end_time - tolerance_ms):
                outlist.append(cluster)
                continue

        outgroup = ClusterGroup(clusters=outlist)

        return outgroup

    def get_harmonic_stack(self, in_id: int) -> ClusterGroup:
        '''Method to associate signal harmonics with fundamental.

        Method currently presumes that the cluster id is currently
        a fundamental. Divides average frequencies and tries to
        see what results in near integer ratios.'''
        fundamental_cluster = self.get_cluster_with_id(in_id)
        assert fundamental_cluster is not None
        fund_freq = fundamental_cluster.avg_freq

        outlist = []
        for cluster in self.clusters:
            assert cluster is not None
            ratio = cluster.avg_freq/fund_freq
            if ratio < 1:
                continue

            if math.isclose(ratio, round(ratio), abs_tol=0.08):
                outlist.append(cluster)

        return ClusterGroup(outlist)

    def set_color(self, color: str) -> None:
        '''Set the cluster to a named or hex color.'''

        regex = re.compile("^#([A-Fa-f0-9]{6}|[A-Fa-f0-9]{3})$")

        if re.search(regex, color):
            for cluster in self.clusters:
                assert cluster is not None
                cluster.color = color
        else:
            pass

    def __resolve_inputs(self,
                         input_v: typing.Union[int, PointCluster]
                         ) -> PointCluster:
        if isinstance(input_v, int):
            output = self.get_cluster_with_id(input_v)
            if output is None:
                raise KeyError
        else:
            output = input_v

        assert output is not None
        return output

    def __xor__(self, other: ClusterGroup) -> ClusterGroup:
        '''Perform Exclusive-Or operation on two cluster groups.'''
        outlist = []
        o_id_list = []
        
        for o_cluster in other.clusters:
            assert o_cluster is not None
            o_id_list.append(o_cluster.id)

        print(f"o id list was {o_id_list}")
        
        for s_cluster in self.clusters:
            if s_cluster.id not in o_id_list:
                outlist.append(s_cluster)

        return ClusterGroup(outlist)

    def __and__(self, other: ClusterGroup) -> None:
        pass

    def __or__(self, other: ClusterGroup) -> None:
        pass


class PointCluster:
    '''A Class defining a single data cluster.'''
    # Maybe might be a good idea to
    # hash relevant data like points etc on cluster
    # and use that as the cluster ID instead of
    # just some number.
    #
    # This will allow us to convincingly say
    # if too clusters are identical or not
    # And then perform "bitwise" operations
    # on them (investigate bitwise operator overloading???)
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

    def freq_at_time(self, time: float) -> float:

        if (time < self.start_time) or time (time > self.end_time):
            return float('NaN')

        nearest_points = []
        
        # Evalutate closest 4 points at time
        

    def interpolate_curve( #NB: ADD 3D FITTING TO ME!
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
        self.cluster_center = \
            (self.start_time + self.end_time)/2
        self.avg_freq = self.freqs.mean()
        self.origin_distance = \
            np.sqrt(self.avg_freq**2 + self.cluster_center**2)
