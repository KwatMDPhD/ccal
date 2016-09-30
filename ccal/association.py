"""
Computational Cancer Analysis Library

Authors:
    Huwate (Kwat) Yeerna (Medetgul-Ernar)
        kwat.medetgul.ernar@gmail.com
        Computational Cancer Analysis Laboratory, UCSD Cancer Center

    Pablo Tamayo
        ptamayo@ucsd.edu
        Computational Cancer Analysis Laboratory, UCSD Cancer Center
"""

from numpy import unique
from pandas import DataFrame, Series, merge
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from seaborn import heatmap

from .support import print_log, establish_filepath, read_gct, untitle_string, information_coefficient, \
    get_unique_in_order, normalize_pandas_object, compare_matrices, match, DPI, CMAP_CONTINUOUS, \
    CMAP_CATEGORICAL, CMAP_BINARY, save_plot, plot_clustermap


# ======================================================================================================================
# Match features against target
# ======================================================================================================================
def catalogue(annotations,
              target=None, target_gct=None, target_df=None, target_name=None, target_axis=1, target_type='continuous',
              filepath_prefix=None):
    """
    Annotate target using multiple annotations.
    :param annotations: list of lists;
        [[name, .gct, data_type, reverse_match_or_not, [row_name, ...](optional), [feature_name, ...](optional)], ...]
    :param target: pandas Series; annotation target
    :param target_gct: str; filepath to a file whose row or column is the annotation target
    :param target_df: DataFrame; whose row or column is the annotation target
    :param target_name: str; row or column name in target_df, either directly passed or read from target_gct
    :param target_axis: int; axis on which target_name is found in target_gct or target_df
    :param target_type: str; {continuous, categorical, binary}
    :param filepath_prefix: str; filepath_prefix_vs_annotation_name.txt and filepath_prefix_vs_annotation_name.pdf
        will be saved
    :return: None
    """

    # Load target
    if not target:
        print_log('Loading the annotation target ...')

        # Load and check target_df
        if target_gct:
            target_df = read_gct(target_gct)
        if not isinstance(target_df, DataFrame):
            raise ValueError('No target_df {}.'.format(target_df))

        # Check target_name
        if not target_name:
            raise ValueError('No target_name {}.'.format(target_name))

        # Load target
        if target_axis == 0:
            target = target_df.ix[:, target_name]
        elif target_axis == 1:
            target = target_df.ix[target_name, :]
        else:
            raise ValueError('Unknown target_axis {}.'.format(target_axis))

    # Load annotations and make_match_panel target

    for a_name, a_dict in _read_annotations(annotations).items():
        if a_dict['data_type'] == 'continuous':
            min_n_feature_values = 3
        else:
            min_n_feature_values = 2

        make_match_panel(a_dict['dataframe'], target, feature_type=a_dict['data_type'], target_type=target_type,
                         feature_ascending=a_dict['reverse_match_or_not'], min_n_feature_values=min_n_feature_values,
                         n_features=0, filepath_prefix=filepath_prefix + '_vs_{}'.format(untitle_string(a_name)))


def _read_annotations(annotations):
    """
    Read annotations from .gct files.
    :param annotations: list of lists;
        [[name, .gct, data_type, reverse_match_or_not, [row_name, ...](optional), [feature_name, ...](optional)], ...]
    :return: dict; {name:{dataframe: DataFrame, data_type: str, reverse_match_or_not: bool}}
    """

    annotation_dict = {}

    # Read all annotations
    for a in annotations:
        print_log('Reading annotation: {} ...'.format(' ~ '.join([str(x) for x in a])))

        name, filepath, data_type, reverse_match_or_not, row_names, feature_names = a

        # Read data type
        annotation_dict[name]['data_type'] = data_type

        # Read whether reverse make_match_panel or not
        annotation_dict[name]['reverse_match_or_not'] = reverse_match_or_not

        # Read annotation DataFrame
        df = read_gct(filepath)
        # Limit to specified features
        if row_names:
            df = df.ix[row_names, :]
            # Update specified features' names
            if feature_names:
                df.set_index(feature_names)
        annotation_dict[name]['dataframe'] = df

        print_log('\t{} features & {} samples.'.format(*annotation_dict[name]['dataframe'].shape))

    return annotation_dict


def make_match_panel(features, target, feature_type='continuous', target_type='continuous',
                     feature_ascending=False, target_sort=True, min_n_feature_values=2,
                     n_features=0.95, n_jobs=1, min_n_per_job=100, n_samplings=30, n_permutations=30,
                     figure_size='auto', title=None, title_size=16, annotation_label_size=9, plot_colname=False,
                     dpi=DPI,
                     filepath_prefix=None):
    """
    Compute: ith score = function(ith feature, target). Compute confidence interval (CI) for n_features
    features. Compute p-val and FDR (BH) for all features. And plot the result.
    :param features: pandas DataFrame; (n_features, n_samples); must have index and column names
    :param target: pandas Series; (n_samples); must have name and index matching features's column names
    :param feature_type: str; {'continuous', 'categorical', 'binary'}
    :param target_type: str; {'continuous', 'categorical', 'binary'}
    :param feature_ascending: bool; True if features scores increase from top to bottom, and False otherwise
    :param target_sort: bool; sort target or not
    :param min_n_feature_values: int; minimum number of unique values in a feature for it to be matched (default 2)
    :param n_features: int or float; number threshold if >= 1, and percentile threshold if < 1
    :param n_jobs: int; number of jobs to parallelize
    :param min_n_per_job: int; minimum number of n per job for parallel computing
    :param n_samplings: int; number of bootstrap samplings to build distribution to get CI; must be > 2 to compute CI
    :param n_permutations: int; number of permutations for permutation test to compute P-val and FDR
    :param figure_size: 'auto' or tuple;
    :param title: str; plot title
    :param title_size: int; title text size
    :param annotation_label_size: int; annotation text size
    :param plot_colname: bool; plot column names below the plot or not
    :param dpi: int; dots per square inch of pixel in the output figure
    :param filepath_prefix: str; filepath_prefix.txt and filepath_prefix.pdf will be saved
    :return: pandas DataFrame; scores
    """

    if isinstance(features, Series):  # Convert Series-features into DataFrame-features with 1 row
        features = DataFrame(features).T

    # Use intersecting columns (samples)
    intersection = set(features.columns) & set(target.index)
    if intersection:
        print_log('features ({} cols) and target {} ({} cols) have {} shared columns.'.format(features.shape[1],
                                                                                              target.name,
                                                                                              target.size,
                                                                                              len(intersection)))
        features = features.ix[:, intersection]
        target = target.ix[intersection]
    else:
        raise ValueError(
            'features ({} cols) and target {} ({} cols) have 0 shared columns.'.format(features.shape[1],
                                                                                       target.name,
                                                                                       target.size))

    # Drop features having less than min_n_feature_values unique values
    print_log('Dropping features with less than {} unique values ...'.format(min_n_feature_values))
    features = features.ix[features.apply(lambda row: len(set(row)), axis=1) >= min_n_feature_values]
    if features.empty:
        raise ValueError('No feature has at least {} unique values.'.format(min_n_feature_values))
    else:
        print_log('\tKept {} features.'.format(features.shape[0]))

    # Sort target
    if target_sort:
        target.sort_values(inplace=True)
        features = features.reindex_axis(target.index, axis=1)

    # Score
    scores = match(features, target, n_features=n_features, ascending=feature_ascending,
                   n_jobs=n_jobs, min_n_per_job=min_n_per_job, n_samplings=n_samplings, n_permutations=n_permutations)
    features = features.reindex(scores.index)

    # Merge features and scores
    features_and_scores = merge(features, scores, left_index=True, right_index=True)

    # Save
    if filepath_prefix:
        establish_filepath(filepath_prefix)
        features_and_scores.to_csv(filepath_prefix + '.txt', sep='\t')

    if not (isinstance(n_features, int) or isinstance(n_features, float)):  # n_features = None
        print_log('Not plotting.')
        return features_and_scores

    #
    # Make annotations and plot
    #
    annotations = DataFrame(index=features.index)

    # Format Score and confidence interval
    for idx, s in features.iterrows():
        if '0.95 MoE' in scores.columns:
            annotations.ix[idx, 'IC(\u0394)'] = '{0:.3f}({1:.3f})'.format(*scores.ix[idx, ['Score', '0.95 MoE']])
        else:
            annotations.ix[idx, 'IC(\u0394)'] = '{:.3f}(_nmf_and_score.xxx)'.format(scores.ix[idx, 'Score'])

    # Format P-Value
    annotations['P-val'] = ['{:.3f}'.format(x) for x in scores.ix[:, 'P-value']]

    # Format FDR
    annotations['FDR'] = ['{:.3f}'.format(x) for x in scores.ix[:, 'FDR']]

    # Plot limited features
    if n_features < 1:  # Limit using percentile
        above_quantile = scores.ix[:, 'Score'] >= scores.ix[:, 'Score'].quantile(n_features)
        print_log('Plotting {} features (> {} percentile) ...'.format(sum(above_quantile), n_features))
        below_quantile = scores.ix[:, 'Score'] <= scores.ix[:, 'Score'].quantile(1 - n_features)
        print_log('Plotting {} features (< {} percentile) ...'.format(sum(below_quantile), 1 - n_features))
        indices_to_plot = scores.index[above_quantile | below_quantile].tolist()
    else:  # Limit using numbers
        if 2 * n_features >= scores.shape[0]:
            indices_to_plot = scores.index
            print_log('Plotting all {} features ...'.format(scores.shape[0]))
        else:
            indices_to_plot = scores.index[:n_features].tolist() + scores.index[-n_features:].tolist()
            print_log('Plotting top & bottom {} features ...'.format(n_features))

    # Plot
    _plot_match_panel(features.ix[indices_to_plot, :], target, annotations.ix[indices_to_plot, :],
                      feature_type=feature_type, target_type=target_type,
                      figure_size=figure_size, title=title, title_size=title_size,
                      annotation_header=' ' * 11 + 'IC(\u0394)' + ' ' * 5 + 'P-val' + ' ' * 4 + 'FDR',
                      annotation_label_size=annotation_label_size, plot_colname=plot_colname,
                      dpi=dpi, filepath=filepath_prefix + '.pdf')

    return features_and_scores


def _plot_match_panel(features, target, annotations, feature_type='continuous', target_type='continuous',
                      std_max=3, figure_size='auto', title=None, title_size=20,
                      annotation_header=None, annotation_label_size=9, plot_colname=False,
                      dpi=DPI, filepath=None):
    """
    Plot make_match_panel panel.
    :param features: pandas DataFrame; (n_features, n_elements); must have indices and columns
    :param target: pandas Series; (n_elements); must have indices, which must make_match_panel features's columns
    :param annotations:  pandas DataFrame; (n_features, n_annotations); must have indices, which must make_match_panel features's
    :param feature_type: str; {'continuous', 'categorical', 'binary'}
    :param target_type: str; {'continuous', 'categorical', 'binary'}
    :param std_max: number;
    :param figure_size: 'auto' or tuple;
    :param title: str;
    :param title_size: number;
    :param annotation_header: str; annotation header to be plotted
    :param annotation_label_size: number;
    :param plot_colname: bool; plot column names or not
    :param dpi: int;
    :param filepath: str;
    :return: None
    """

    # Set up features
    if feature_type == 'continuous':
        features_cmap = CMAP_CONTINUOUS
        features_min, features_max = -std_max, std_max
        print_log('Normalizing continuous features ...')
        features = normalize_pandas_object(features, method='-0-', axis=1)
    elif feature_type == 'categorical':
        features_cmap = CMAP_CATEGORICAL
        features_min, features_max = 0, len(unique(features))
    elif feature_type == 'binary':
        features_cmap = CMAP_BINARY
        features_min, features_max = 0, 1
    else:
        raise ValueError('Unknown feature_type {}.'.format(feature_type))

    # Set up target
    if target_type == 'continuous':
        target_cmap = CMAP_CONTINUOUS
        target_min, target_max = -std_max, std_max
        print_log('Normalizing continuous ref ...')
        target = normalize_pandas_object(target, method='-0-')
    elif target_type == 'categorical':
        target_cmap = CMAP_CATEGORICAL
        target_min, target_max = 0, len(unique(target))
    elif target_type == 'binary':
        target_cmap = CMAP_BINARY
        target_min, target_max = 0, 1
    else:
        raise ValueError('Unknown target_type {}.'.format(target_type))

    # Set up figure
    if figure_size == 'auto':
        figure_size = (min(pow(features.shape[1], 0.7), 7), pow(features.shape[0], 0.9))
    plt.figure(figsize=figure_size)

    # Set up axes
    gridspec = GridSpec(features.shape[0] + 1, features.shape[1] + 1)
    ax_target = plt.subplot(gridspec[:1, :features.shape[1]])
    ax_features = plt.subplot(gridspec[1:, :features.shape[1]])
    ax_annotation_header = plt.subplot(gridspec[:1, features.shape[1]:])
    ax_annotation_header.axis('off')
    horizontal_text_margin = pow(features.shape[1], 0.39)

    #
    # Plot target, target label, and title
    #
    # Plot target heat band
    heatmap(DataFrame(target).T, ax=ax_target, vmin=target_min, vmax=target_max, cmap=target_cmap, xticklabels=False,
            cbar=False)

    # Adjust target name
    for t in ax_target.get_yticklabels():
        t.set(rotation=0, weight='bold')

    if target_type in ('binary', 'categorical'):  # Add binary or categorical target labels
        boundaries = [0]

        # Get values
        prev_v = target.iloc[0]
        for i, v in enumerate(target.iloc[1:]):
            if prev_v != v:
                boundaries.append(i + 1)
            prev_v = v
        boundaries.append(features.shape[1])

        # Get positions
        label_horizontal_positions = []
        prev_b = 0
        for b in boundaries[1:]:
            label_horizontal_positions.append(b - (b - prev_b) / 2)
            prev_b = b
        unique_target_labels = get_unique_in_order(target.values)

        # Plot values to their corresponding positions
        for i, pos in enumerate(label_horizontal_positions):
            ax_target.text(pos, 1, unique_target_labels[i], horizontalalignment='center', weight='bold')

    if title:  # Plot title
        ax_target.text(features.shape[1] / 2, 1.9, title, horizontalalignment='center', size=title_size, weight='bold')

    #
    # Plot features
    #
    # Plot features heatmap
    heatmap(features, ax=ax_features, vmin=features_min, vmax=features_max, cmap=features_cmap,
            xticklabels=plot_colname, cbar=False)
    for t in ax_features.get_yticklabels():
        t.set(rotation=0, weight='bold')

    #
    # Plot annotations
    #
    # Plot annotation header
    if not annotation_header:
        annotation_header = '\t'.join(annotations.columns).expandtabs()
    ax_annotation_header.text(horizontal_text_margin, 0.5, annotation_header, horizontalalignment='left',
                              verticalalignment='center', size=annotation_label_size, weight='bold')

    # Plot annotations for each feature
    for i, (idx, s) in enumerate(annotations.iterrows()):
        ax = plt.subplot(gridspec[i + 1:i + 2, features.shape[1]:])
        ax.axis('off')
        a = '\t'.join(s.tolist()).expandtabs()
        ax.text(horizontal_text_margin, 0.5, a, horizontalalignment='left', verticalalignment='center',
                size=annotation_label_size, weight='bold')

    # Save
    if filepath:
        save_plot(filepath, dpi=dpi)


# ======================================================================================================================
# Compare matrices
# ======================================================================================================================
def make_comparison_matrix(matrix1, matrix2, function=information_coefficient, axis=0, is_distance=False, title=None,
                           filepath_prefix=None):
    """
    Compare matrix1 and matrix2 by row (axis=1) or by column (axis=0), and plot cluster map.
    :param matrix1: pandas DataFrame or numpy 2D arrays;
    :param matrix2: pandas DataFrame or numpy 2D arrays;
    :param function: function; association or distance function
    :param axis: int; 0 for row-wise and 1 for column-wise comparison
    :param is_distance: bool; if True, distances are computed from associations, as in 'distance = 1 - association'
    :param title: str; plot title
    :param filepath_prefix: str; filepath_prefix.txt and filepath_prefix.pdf will be saved
    :return: pandas DataFrame; association or distance matrix
    """

    # Compute association or distance matrix, which is returned at the end
    comparison_matrix = compare_matrices(matrix1, matrix2, function, axis=axis, is_distance=is_distance)

    # Save
    if filepath_prefix:
        comparison_matrix.to_csv(filepath_prefix + '.txt', sep='\t')

    # Plot cluster map of the compared matrix
    if filepath_prefix:
        filepath = filepath_prefix + '.pdf'
    else:
        filepath = None
    plot_clustermap(comparison_matrix, title=title, filepath=filepath)

    return comparison_matrix
