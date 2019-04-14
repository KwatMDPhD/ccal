from plotly.offline import iplot
from plotly.offline import plot as offline_plot
from plotly.plotly import plot as plotly_plot


def plot_and_save(figure, html_file_path, plotly_html_file_path=None):

    if "layout" in figure:

        figure["layout"].update({"automargin": True, "hovermode": "closest"})

    if html_file_path is not None:

        print(
            offline_plot(
                figure,
                filename=html_file_path,
                auto_open=False,
                config={"editable": True},
            )
        )

    if plotly_html_file_path is not None:

        print(
            plotly_plot(
                figure,
                filename=plotly_html_file_path,
                file_opt="overwrite",
                auto_open=False,
            )
        )

    iplot(figure, config={"editable": True})
