from pandas import DataFrame, Series, merge

from .support import print_log, establish_path
from .information import information_coefficient
from .visualize import DPI, plot_features_against_reference
from .analyze import compute_against_reference


def match(features, ref, feature_type='continuous', ref_type='continuous',
          feature_ascending=False, ref_ascending=False, ref_sort=True,
          function=information_coefficient, n_features=0.95, n_samplings=30, confidence=0.95,
          n_perms=30, title=None, title_size=16, annotation_label_size=9, plot_colname=False,
          result_filename=None, figure_filename=None, figure_size='auto', dpi=DPI):
    """
    Compute 'features' vs. `ref`.
    :param features: pandas DataFrame; (n_features, n_samples); must have indices and columns
    :param ref: pandas Series; (n_samples); must have name and columns, which must match `features`'s
    :param feature_type: str; {'continuous', 'categorical', 'binary'}
    :param ref_type: str; {'continuous', 'categorical', 'binary'}
    :param feature_ascending: bool; True if features score increase from top to bottom, and False otherwise
    :param ref_ascending: bool; True if ref values increase from left to right, and False otherwise
    :param ref_sort: bool; sort `ref` or not
    :param function: function; function to score
    :param n_features: int or float; number threshold if >= 1 and percentile threshold if < 1
    :param n_samplings: int; number of sampling for confidence interval bootstrapping; must be > 2 to compute CI
    :param confidence: float; confidence interval
    :param n_perms: int; number of permutations for permutation test
    :param title: str; plot title
    :param title_size: int; title text size
    :param annotation_label_size: int; annotation text size
    :param plot_colname: bool; plot column names or not
    :param result_filename: str; file path to the output result
    :param figure_filename: str; file path to the output figure
    :param figure_size: 'auto' or tuple;
    :param dpi: int; dots per square inch of pixel in the output figure
    :return: None
    """
    print_log('Computing features vs. {} using {} ...'.format(ref.name, function))

    # Convert features into pandas DataFrame
    if isinstance(features, Series):
        features = DataFrame(features).T

    # Use intersecting columns
    col_intersection = set(features.columns) & set(ref.index)
    if col_intersection:
        print_log('features ({} cols) and ref ({} cols) have {} intersecting columns.'.format(features.shape[1],
                                                                                              ref.size,
                                                                                              len(col_intersection)))
        features = features.ix[:, col_intersection]
        ref = ref.ix[col_intersection]
    else:
        raise ValueError('features and ref have 0 intersecting columns, having {} and {} columns respectively'.format(
            features.shape[1], ref.size))

    # Drop rows having all 0 values
    features = features.ix[(features != 0).any(axis=1)]

    # Sort reference
    if ref_sort:
        ref = ref.sort_values(ascending=ref_ascending)
        features = features.reindex_axis(ref.index, axis=1)

    # Compute scores
    scores = compute_against_reference(features, ref, function=function,
                                       n_features=n_features, ascending=feature_ascending,
                                       n_samplings=n_samplings, confidence=confidence, n_perms=n_perms)
    features = features.reindex(scores.index)

    if result_filename:
        establish_path(result_filename)
        merge(features, scores, left_index=True, right_index=True).to_csv(result_filename, sep='\t')

    # Make annotations
    annotations = DataFrame(index=features.index)
    for idx, s in features.iterrows():
        if '{} MoE'.format(confidence) in scores.columns:
            annotations.ix[idx, 'IC(\u0394)'] = '{0:.3f}'.format(scores.ix[idx, function]) \
                                                + '({0:.3f})'.format(scores.ix[idx, '{} MoE'.format(confidence)])
        else:
            annotations.ix[idx, 'IC(\u0394)'] = '{0:.3f}(x.xxx)'.format(scores.ix[idx, function])

    annotations['P-val'] = ['{0:.3f}'.format(x) for x in scores.ix[:, 'Global P-value']]
    annotations['FDR'] = ['{0:.3f}'.format(x) for x in scores.ix[:, 'FDR (BH)']]

    # Limit features to be plotted
    if n_features < 1:
        above_quantile = scores.ix[:, function] >= scores.ix[:, function].quantile(n_features)
        print_log('Plotting {} features vs. reference > {} quantile ...'.format(sum(above_quantile), n_features))
        below_quantile = scores.ix[:, function] <= scores.ix[:, function].quantile(1 - n_features)
        print_log('Plotting {} features vs. reference < {} quantile ...'.format(sum(below_quantile), 1 - n_features))
        indices_to_plot = features.index[above_quantile | below_quantile].tolist()
    else:
        indices_to_plot = features.index[:n_features].tolist() + features.index[-n_features:].tolist()
        print_log('Plotting top & bottom {} features vs. reference ...'.format(len(indices_to_plot)))

    plot_features_against_reference(features.ix[indices_to_plot, :], ref, annotations.ix[indices_to_plot, :],
                                    feature_type=feature_type, ref_type=ref_type,
                                    figure_size=figure_size, title=title, title_size=title_size,
                                    annotation_header=' ' * 7 + 'IC(\u0394)' + ' ' * 9 + 'P-val' + ' ' * 4 + 'FDR',
                                    annotation_label_size=annotation_label_size, plot_colname=plot_colname,
                                    output_filepath=figure_filename, dpi=dpi)
