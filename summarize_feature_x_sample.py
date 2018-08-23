from numpy.random import choice

from .plot.plot.plot_distributions import plot_distributions
from .plot.plot.plot_heat_map import plot_heat_map


def summarize_feature_x_sample(
        feature_x_sample,
        plot=True,
        max_plot_n=int(1e6),
):

    print('Feature-x-Sample Shape: {}'.format(feature_x_sample.shape))

    if plot:

        feature_x_sample_values = feature_x_sample.unstack().dropna()

        str_ = 'Feature-x-Sample'

        if feature_x_sample_values.size < max_plot_n:

            plot_heat_map(
                feature_x_sample,
                title=str_,
                xaxis_title='Sample',
                yaxis_title='Feature',
            )

        else:

            feature_x_sample_values = choice(
                feature_x_sample_values,
                size=max_plot_n,
                replace=False,
            )

            str_ += ' (random {:,})'.format(max_plot_n)

        str_ = '{} Value'.format(str_)

        plot_distributions(
            (str_, ),
            (feature_x_sample_values, ),
            plot_rug=False,
            title='{} Distribution'.format(str_),
            xaxis_title='Value',
        )

    isna__feature_x_sample = feature_x_sample.isna()

    n_0 = isna__feature_x_sample.values.sum()

    percent_0 = n_0 / feature_x_sample.size * 100

    if n_0:

        print('N NA: {} ({:.2f}%)'.format(
            n_0,
            percent_0,
        ))

        if plot:

            if max(isna__feature_x_sample.shape) < max_plot_n:

                plot_distributions(
                    (
                        'Feature',
                        'Sample',
                    ),
                    (
                        isna__feature_x_sample.sum(axis=1),
                        isna__feature_x_sample.sum(),
                    ),
                    title='NA Distribution',
                    xaxis_title='N NA',
                )
