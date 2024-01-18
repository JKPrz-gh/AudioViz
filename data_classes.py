#!/usr/bin/python3
'''Classes for AudioViz.'''

from __future__ import annotations
import numpy as np
import numpy.typing as npt
import numpy.polynomial.polynomial as np_poly
import sklearn.cluster as skcl
import librosa
import soundfile as sf
import random
import math
import binascii
import io
import re
import copy
import typing
from typing import Union, Dict, Any
import pandas as pd
import settings


class AudioFile:
    '''Class Defining an Audio File.

    Attributes:
        path (str):
        threshold (int)
        amplitudes:
        sample_rate
        length
    '''
    def __init__(self,
                 file_descriptor: Union[int, None], 
                 harmonic_filtering: Union[int, None] = None,
                 threshold: int = -21,
                 ) -> None:
        '''Create a new instance of an AudioFile.

        Args:
            file_descriptor (int): an open file descriptor
        '''
        self.threshold = threshold
        
        if file_descriptor is None:
            pass
        else:
            self.fd = file_descriptor
            self.__load_file(harmonic_filtering)

    def raw_data(self) -> tuple(npt.NDArray, int):
        '''Get the raw data returned by librosa.

        Returns:
             returntype
        '''
        return (self.amplitudes, self.sample_rate)

    def as_df(self) -> pd.DataFrame:
        '''Get the data as a Pandas Dataframe.'''
        df = pd.DataFrame()

        df['Amplitude'] = self.amplitudes
        df['Time'] = df.index*(1/self.sample_rate)

        return df

    def get_reassigned_points(self) -> TimeFrequencyData:
        '''Use a Reassigned Spectrogram to compute points.'''
        freqs, times, mags = librosa.reassigned_spectrogram(
            self.amplitudes,
            sr=self.sample_rate,
            n_fft=4086
        )

        mags_db = librosa.amplitude_to_db(np.abs(mags), ref=np.max)

        # These are effectively x and y and z co-ordinates
        freqs = np.ravel(freqs)
        times = np.ravel(times)
        mags_db = np.ravel(mags_db)

        sound_data = np.c_[freqs, times, mags_db]
        sound_data = sound_data[sound_data[:, 2] >= self.threshold]

        # Make sure our data is time sorted
        sound_data = sound_data[sound_data[:, 1].argsort()]

        # Make sure our first data point is at time zero
        sound_data[:, 1] -= sound_data[0, 1]

        outData = TimeFrequencyData(
            sound_data
        )

        return outData
        
    def __load_file(self, harmonic_filtering) -> None:
        amplitudes, sample_rate = librosa.load(self.fd, sr=22050)

        if harmonic_filtering is not None:
            amplitudes = librosa.effects.harmonic(amplitudes, margin=harmonic_filtering)

        self.amplitudes = amplitudes
        self.sample_rate = sample_rate
        self.length = (1/sample_rate)*len(amplitudes)


    @classmethod
    def from_pathstring(cls, input_path: str):
        return cls(open(input_path, 'rb'))

    @classmethod
    def from_bytes(cls, input_data: bytes):
        return cls(io.BytesIO(input_data))
        
        
class TimeFrequencyData:
    '''Class for holding Time Frequency data with Decibel magnitude.

    This class exists as a wrapper for the underlying numpy
    arrays.

    Attributes:
        data (NDArray): 3xN array holding data points.
    '''
    def __init__(self,
                 t_f_data: npt.NDArray,
                 ) -> None:
        '''Create a new instance of TimeFrequencyData.

        Args:
            t_f_data: A 3xN array holding Time-Frequency information.
        '''
        self.__check_input(t_f_data)
        self.data = t_f_data

    def get_freqs(self) -> npt.NDArray:
        return self.data[:, 0]

    def get_times(self) -> npt.NDArray:
        return self.data[:, 1]
     
    def get_mags_db(self) -> npt.NDArray:
        return self.data[:, 2]

    def as_df(self) -> pd.DataFrame:
        '''Return Contained data as a Pandas DataFrame.

        Returns:
            pd.DataFrame containing time-frequency data.
        '''
        df = pd.DataFrame()

        df['Frequency'] = self.data[:, 0]
        df['Time'] = self.data[:, 1]
        df['DecibelMagnitude'] = self.data[:, 2]

        return df

    def __check_input(self, t_f_data) -> None:
        '''Function for Sanitizing inputs.

        Raises:
            pass
        '''
        pass


class ClusterGroupFactory:
    '''Class Defining a generic ClusteredDataSet'''
    def __init__(self,
                 sound_data: TimeFrequencyData,
                 prescale_x: typing.Tuple[float]
                 ) -> None:
        '''Constructor for ClusteredDataSet.

        This method loads in a dataset and breaks
        it out into clusters.'''
        # load in data and preprocess it
        self.sound_data = sound_data.data
        self.__preprocess(prescale_x)

        # Break out data into cluster objects
        self.__clusters: typing.List[typing.Union[None, PointCluster]] \
            = [None]*self.__labels.size
        self.__create_cluster_objects()

    def get_cluster_group(self) -> ClusterGroup:
        group = ClusterGroup(self.__clusters)
        return group

    # Private helper Methods
    def __preprocess(self,
                     prescale_x,
                     ) -> None:
        '''Threshold and cluster data'''
        # Unpack our input tuple
        sd_copy = self.sound_data

        # Apply the prescale factor and cluster
        sd_copy[:, 1] *= prescale_x
        db = skcl.HDBSCAN().fit(sd_copy)
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

    def has_member(self, input_data: typing.Union[int, PointCluster]):
        '''Check if given cluster is member of group.

        This function can be provided a cluster ID or an
        actual cluster,'''
        if isinstance(input_data, int):
            input_id = input_data
            print(f"Got an integer! {input_id} btw!")
        else:
            if isinstance(input_data, PointCluster):
                input_id = input_data.id

            else:
                raise TypeError

        for cluster in self.clusters:
            if input_id == cluster.id:
                return True

        return False

    def query_member(self, input_id: int) -> Dict[Any]:
        '''Get information about a cluster in the group.

        Method wrraping around the PointCluster.info()
        method, returning none if the cluster is not present.
        '''
        cluster = self.get_cluster_with_id(input_id)

        if cluster is None:
            return None
        else:
            return cluster.info()
        

    def get_cluster_with_id(self, input_id: int) -> typing.Union[None, PointCluster]:
        for cluster in self.clusters:
            assert cluster is not None
            if cluster.id == input_id:
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

        outlist = []
        for cluster in self.clusters:
            harmonic_freq = cluster.freq_at_time(cluster.cluster_center)
            fundamental_freq = fundamental_cluster\
                .freq_at_time(cluster.cluster_center)

            ratio = harmonic_freq/fundamental_freq
            if ratio < 1:
                continue

            if math.isnan(ratio):
                continue
            
            if math.isclose(ratio, round(ratio), abs_tol=0.2):
                outlist.append(cluster)

        return ClusterGroup(outlist)

    def get_harmonic_stack_groups(self) -> list[ClusterGroup]:
        '''Get Multiple Harmonic Stack Groups.'''
        colorlist = [ "#358BAC", "#809AAF", "#FB7CA8", "#6935B3", "#3EBF70", "#0CCE8A",
        "#2847E0", "#C24606", "#559BB5", "#D0D658", "#C42C34", "#F77C34"]
        iters = 0
        outlist = []

        cs_copy = copy.deepcopy(self)
        
        while not cs_copy.is_empty() and (iters < 200):
            lowest_id = cs_copy.get_lowest_cluster_id()
            c_subset = cs_copy.get_coinciding_clusters(lowest_id)
            h_stack = c_subset.get_harmonic_stack(lowest_id)
            cs_copy = cs_copy ^ h_stack
            
            h_stack.set_color(colorlist[iters % 12])

            outlist.append(h_stack)
            iters += 1

        return outlist

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

    def __str__(self) -> None:

        outstring = f"ClusterGroup containing {len(self.clusters)} clusters. \n [ "

        count = 0
        for cluster in self.clusters:
            outstring += " {:3d} ".format(cluster.id)

            if (count % 11) == 10:
                outstring += " \n   "

            count += 1
    
        outstring += ']'
  
        return outstring


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
    #
    # Later note:
    #     Do I really need this?
    def __init__(self,
                 points_data: npt.NDArray,
                 cluster_id: int,
                 color: str,
                 ) -> None:
        '''Constructor class for a cluster of Points.'''
        # Read in data from call
        self.times = points_data[:, 1]
        self.freqs = points_data[:, 0]
        self.mags_db = points_data[:, 2]

        self.id = cluster_id
        self.color = color
        self.no_points = self.times.size

        # Run the metadata population function
        self.__populate_metadata()

    def freq_at_time(self, time: float) -> float:
        '''Method for returning a frequency at a Given time'''

        # Return a NaN if try to query a frequency outside
        # the expected frequency range
        if (time < self.start_time) or (time > self.end_time):
            return float('NaN')

        # Init the array of points used to compute
        # the average frequency
        nearest_points = []
        
        # Evalutate closest 4 points at time
        distances = np.abs(self.times - time)
        n = 4

        # Parse the closest points in order
        # to get the frequencies
        idx = np.argpartition(distances, n)
        freq = self.freqs[idx[:n]].mean()
        
        return freq
        

    def interpolate_curve(
            self,
            degree: int = 4
    ) -> typing.Tuple[
        npt.NDArray,
        npt.NDArray
    ]:
        '''Method to fit a curve to a data cluster.

        This method fits a polynomial of degree {degree}
        to the points in the cluster.

        Returns a Typle of evaluated x and y values.'''
        # create the initial curve by fitting a polynomial
        curve_freq = np_poly.Polynomial.fit(
            self.times,
            self.freqs,
            degree
        )

        # Do the same for the power curve of the data
        curve_power = np_poly.Polynomial.fit(
            self.times,
            self.mags_db,
            2
        )

        # Want the same point density for clusters
        # of all lengths, so use a constant sample rate
        # Note:
        #    Duration is a float, but I_S_R is a large number,
        #    so minimal information is lost from it cast
        points = int(settings.INTERPOLATION_SAMPLE_RATE*self.duration)

        # Evalutate frequency curve at at the number of points
        # requessted
        x, y = curve_freq.linspace(
            n=points,
            domain=[
                self.start_time,
                self.end_time
            ]
        )

        # Do the same for the power curve, which will have
        # the same x values as the frequency curve,
        # so those can be discarded
        _, z = curve_power.linspace(
            n=points,
            domain=[
                self.start_time,
                self.end_time
                ]
        )
        return x, y, z

    def info(self) -> Dict[Any]:
        '''Get information about a specific cluster.

        Function returns a dict of values with
        details of a cluster's contents.
        '''
        infodict = {
            'id': self.id,
            'numPoints': self.no_points,
            'color': self.color,
            'startTime': self.start_time,
            'endTime': self.end_time,
            'duration': self.duration,
            'clusterMidpoint': self.cluster_center,
            'meanFrequency': self.avg_freq,
            'originDistance': self.origin_distance,
        }

        return infodict

    def __populate_metadata(self) -> None:
        '''Function for creating cluster metadata.'''
        self.start_time = self.times.min()
        self.end_time = self.times.max()
        self.duration = self.end_time - self.start_time
        self.cluster_center = \
            (self.start_time + self.end_time)/2
        self.avg_freq = self.freqs.mean()
        self.origin_distance = \
            np.sqrt(self.avg_freq**2 + self.cluster_center**2)
