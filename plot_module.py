import data_classes
import typing
import matplotlib
import matplotlib.pyplot as plt
import librosa
from matplotlib.ticker import EngFormatter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.collections import PolyCollection
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly

plt.style.use('dark_background')

class MplPlotDisplay:
    '''Matplotlib Plot Creator for Plots'''

    def __init__(self):
         px = 1/plt.rcParams['figure.dpi']
         matplotlib.rcParams['figure.figsize'] = (1920*px, 1080*px)
         matplotlib.rc('font', size=14)
         self.px = px

    def plot_spectrogram(
            self, y, sr,
            title: str = 'Spectrogram',
            xlabel: str = 'Time (s)',
            ylabel: str = 'Frequency',
            xlim: float = 3.5,
            ylim: float = 3400.0
    ) -> None:

        #D = librosa.stft(y)  # STFT of y
        #S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

        fig, ax = plt.subplots()
        ax.yaxis.set_major_formatter(EngFormatter(unit='Hz'))
        ax.xaxis.set_major_formatter(EngFormatter(unit='s'))
        #img = librosa.display.specshow(S_db,  x_axis='time', y_axis='linear', ax=ax)
        ax.specgram(y,
                    Fs=sr, cmap='magma', NFFT=1024,
                    pad_to=4096, mode='psd',
                    detrend='mean')
        ax.grid(color='gray')
        #fig.colorbar(ax=ax, format="%+2.f dB")
        ax.set_xlim([0, 3.5])
        ax.set_ylim([200, 3400])
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel, loc='bottom', rotation=0, labelpad=40.0)
        ax.set_title(title, fontsize = 20)
        plt.show()
        
    def plot_reassigned_scatter(
            self,
            times,
            freqs,
            mags_db,
            xlabel: str = 'Time',
            ylabel: str = 'Frequency'
    ) -> None:

        fig, ax = plt.subplots()
        ax.yaxis.set_major_formatter(EngFormatter(unit='Hz'))
        ax.xaxis.set_major_formatter(EngFormatter(unit='s'))
        ax.scatter(times, freqs, c=mags_db, s=1, cmap='magma')
        ax.grid(color='gray')
        ax.set_xlim([0, 20])
        ax.set_ylim([200, 3400])
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel, loc='bottom', rotation=0, labelpad=20.0)
        plt.title("Reassigned Spectrogram", fontsize = 20)
        plt.show()

    def plot_scatterplot(
            self,
            input_data: typing.List[data_classes.ClusterGroup],
            title: str = 'Cluster Plot',
            xlabel: str = 'Time',
            ylabel: str = 'Frequency'
    ) -> None:

        fig, ax = plt.subplots()
        ax.yaxis.set_major_formatter(EngFormatter(unit='Hz'))
        ax.xaxis.set_major_formatter(EngFormatter(unit='s'))
        for group in input_data:
            for cluster in group.clusters:
                # Make the type checker happy
                if cluster is None:
                    continue
                
                ax.scatter(
                    cluster.times,
                    cluster.freqs,
                    c=cluster.color,
                    s=7
                )

                ax.annotate(
                    f"{cluster.id}",
                    (
                        cluster.times[0],
                        cluster.freqs[0]
                    ),
                    fontsize = 10
                )

        ax.set_title("All clusters", fontsize = 20)
        ax.grid(color='gray')
        ax.set_xlim([0,20])
        ax.set_ylim([200, 3400])
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel, loc='bottom', rotation=0, labelpad=20.0)

        plt.show()

    def plot_curves(
            self,
            input_data: typing.List[data_classes.ClusterGroup],
            title: str = 'Curve Plot',
            xlabel: str = 'Time',
            ylabel: str = 'Frequency',
            zlabel: str = 'Magnitude'
    ) -> None:


        fig = plt.figure()
        ax = plt.axes(projection='3d')

        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False

        
        ax.view_init(55, -160)
        ax.yaxis.set_major_formatter(EngFormatter(unit='Hz'))
        ax.xaxis.set_major_formatter(EngFormatter(unit='s'))
        for group in input_data:
            for cluster in group.clusters:
                # Make the type checker happy
                if cluster is None:
                    continue

                if cluster.avg_freq > 3400:
                    continue

                cluster_x, cluster_y, cluster_z = cluster.interpolate_curve()
                # (np.ptp(cluster_x), np.ptp(cluster_y), np.ptp(cluster_z))
                ax.set_box_aspect(aspect=(1.4,1,0.2))
                ax.plot(
                    cluster_x,
                    cluster_y,
                    cluster_z,
                    c=cluster.color,
                    #alpha=cluster_z,
                    linewidth=4,
                    solid_capstyle='round'
                )

                PlotWriter.__fill_between_3d(
                    ax=ax,
                    x1=cluster_x,
                    y1=cluster_y,
                    z1=np.full((cluster_z.shape), -25),
                    x2=cluster_x,
                    y2=cluster_y,
                    z2=cluster_z,
                    mode=1,
                    c=cluster.color
                ) 

                ax.annotate(
                    f"{cluster.id}",
                    (
                        cluster.times[0],
                        cluster.freqs[0]
                    ),
                    fontsize=10
                )

        # Draw the plots
        ax.set_title(title, fontsize = 20)
        ax.set_xlabel(xlabel)
        ax.set_zlabel(zlabel)
        ax.set_ylabel(ylabel, loc='bottom', rotation=0, labelpad=20.0)
        ax.set_xlim([0, 20])
        ax.set_ylim([200, 3400])
        ax.set_zlim([-25, 0])
        ax.grid(color='gray')
        plt.show()

    def __fill_between_3d(ax,x1,y1,z1,x2,y2,z2,mode=1,c='steelblue',alpha=0.3):
        if mode == 1:
            for i in range(len(x1)-1):
            
                verts = [(x1[i],y1[i],z1[i]), (x1[i+1],y1[i+1],z1[i+1])] + \
                    [(x2[i+1],y2[i+1],z2[i+1]), (x2[i],y2[i],z2[i])]
                
                ax.add_collection3d(Poly3DCollection([verts],
                                                     alpha=alpha,
                                                     linewidths=0,
                                                     color=c))
        if mode == 2:
            verts = [(x1[i],y1[i],z1[i]) for i in range(len(x1))] + \
                [(x2[i],y2[i],z2[i]) for i in range(len(x2))]
            
            ax.add_collection3d(Poly3DCollection([verts],alpha=alpha,color=c))


class PxFigureCreator:

    def __init__(self,
                 plot_width: int = 1300,
                 plot_height: int = 650,
                 template: str = 'plotly_dark'
                 ) -> None:
        self.plot_width = plot_width
        self.plot_height = plot_height
        self.template = template

    def plot_time_domain(
            self,
            input_data: data_classes.AudioFile,
            xlabel: str = 'Time',
            ylabel: str = 'Frequency'
    ) -> plotly.graph_objs._figure.Figure:

        fig = px.line(
            input_data.as_df(),
            x='Time',
            y='Amplitude',
            template=self.template,
            width=self.plot_width,
            height=self.plot_height
        )
        
        return fig

    def plot_standard_spectrogram(self) -> None:
        pass

    def plot_reassigned_scatter(
            self,
            input_data: data_classes.AudioFile,
            xlabel: str = 'Time',
            ylabel: str = 'Frequency'
    ) -> plotly.graph_objs._figure.Figure:

        points = input_data.get_reassigned_points()
        
        fig = px.scatter(
            points.as_df(),
            x='Time',
            y='Frequency',
            color='DecibelMagnitude',
            template=self.template,
            width=self.plot_width,
            height=self.plot_height
        )

        fig.update_layout(
            yaxis_tickformat = '.2s',
            yaxis_ticksuffix = 'Hz',
            xaxis_tickformat = '.1,',
            xaxis_ticksuffix = ' s',
            clickmode='event+select'
        )

        return fig
    
    def plot_clusters(
            self,
            input_data: data_classes.AudioFile,
            xlabel: str = 'Time',
            ylabel: str = 'Frequency',
            prescale_x: int = 120
    ) -> plotly.graph_objs._figure.Figure:

        points = input_data.get_reassigned_points()

        group_factory = data_classes.ClusterGroupFactory(points, prescale_x)

        cluster_group = group_factory.get_cluster_group()

        fig = go.Figure(
            layout=go.Layout(
                template=self.template,
                width=self.plot_width,
                height=self.plot_height
            )
        )

        for cluster in cluster_group.clusters:
            # Add each cluster
            fig.add_trace(
                go.Scatter(
                    x=cluster.times,
                    y=cluster.freqs,
                    mode='markers',
                    marker=dict(
                        size=3,
                        color=cluster.color,
                        colorscale='Viridis',   # choose a colorscale
                        opacity=0.8
                    )
                )
            )

            # Add annotation per cluster
            fig.add_annotation(
                x=cluster.times[0], y=cluster.freqs[0],
                text=cluster.id,
                showarrow=False,
                yshift=0
            )
        
        fig.update_layout(
            showlegend=False,
            yaxis_tickformat = '.2s',
            yaxis_ticksuffix = 'Hz',
            xaxis_tickformat = '.1,',
            xaxis_ticksuffix = ' s',
            clickmode='event+select'
        )
        fig.update_xaxes(title_text=xlabel)
        fig.update_yaxes(title_text=ylabel)

        return fig
        
    def plot_curves(
            self,
            input_data: data_classes.AudioFile,
            xlabel: str = 'Time',
            ylabel: str = 'Frequency',
            prescale_x = 120,
    ) -> plotly.graph_objs._figure.Figure:

        points = input_data.get_reassigned_points()

        group_factory = data_classes.ClusterGroupFactory(points, prescale_x)

        cluster_group = group_factory.get_cluster_group()
        stack_groups = cluster_group.get_harmonic_stack_groups()

        fig = go.Figure(
            layout=go.Layout(
                template=self.template,
                width=self.plot_width,
                height=self.plot_height
            )
        )

        for stack in stack_groups:
            for cluster in stack.clusters:
                # Add each cluster
                
                x_vals, y_vals, z_vals = cluster.interpolate_curve()
                fig.add_trace(
                    go.Scatter(
                        x=x_vals,
                        y=y_vals,
                        line=dict(
                            color=cluster.color
                        )       
                    )
                )

            # Add annotation per group
            fig.add_annotation(
                x=cluster.times[0], y=cluster.freqs[0],
                text=cluster.id,
                showarrow=False,
                yshift=0
            )
        
        fig.update_layout(
            showlegend=False,
            yaxis_tickformat = '.2s',
            yaxis_ticksuffix = 'Hz',
            xaxis_tickformat = '.1,',
            xaxis_ticksuffix = ' s',
            clickmode='event+select'
            
        )
        fig.update_xaxes(title_text=xlabel)
        fig.update_yaxes(title_text=ylabel)

        
        return fig
    

        
            
