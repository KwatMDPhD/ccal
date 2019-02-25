from plotly.offline import iplot
from plotly.offline import plot as offline_plot
from plotly.plotly import plot as plotly_plot


def plot_and_save(figure, html_file_path, plotly_html_file_path):

    plot_keyword_argumenmts = dict(auto_open=False, config=dict(editable=True))

    if html_file_path is not None:

        print(offline_plot(figure, filename=html_file_path, **plot_keyword_argumenmts))

    if plotly_html_file_path is not None:

        print(
            plotly_plot(
                figure,
                filename=plotly_html_file_path,
                file_opt="overwrite",
                sharing="public",
                **plot_keyword_argumenmts
            )
        )

    iplot(figure, config=dict(editable=True))
