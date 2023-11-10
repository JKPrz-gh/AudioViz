import data_classes
import typing
import matplotlib.pyplot as plt


class ClusterGrapher:
    '''Class for graphing clustered data'''

    def plot_curves(
            self,
            input_data: typing.List[data_classes.ClusterGroup],
            title: str = 'Cluster Plot',
    ) -> None:

        for group in input_data:
            for cluster in group.clusters:
                # Make the type checker happy
                if cluster is None:
                    continue

                cluster_x, cluster_y = cluster.interpolate_curve()
                plt.plot(
                    cluster_x,
                    cluster_y,
                    c=cluster.color,
                    linewidth=2
                )

                plt.annotate(
                    f"{cluster.id}",
                    (
                        cluster.times[0],
                        cluster.freqs[0]
                    )
                )

        # Draw the plots
        plt.title(title)
        plt.grid()
        plt.show()
