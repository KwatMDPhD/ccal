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
def make_association_panels(target_bundle, feature_bundle,
                            n_features=0.95, n_jobs=1, n_samplings=30, n_permutations=30, filepath_prefix=None):
    """
    Annotate each target in the target bundle with each feature in the feature bundle.
    :param target_bundle: list of lists;
        [
            [name
            df or filepath to .gct
            data_type,
            is_ascending,
            (optional) index_axis,
            (optional) index,
            (optional) index_alias],

            ...

        ]
    :param feature_bundle: list of lists;
        [
            [name
            df or filepath to .gct
            data_type,
            is_ascending,
            (optional) index_axis,
            (optional) index,
            (optional) index_alias],

            ...

        ]
    :param n_features: int or float; number threshold if >= 1, and percentile threshold if < 1
    :param n_jobs: int; number of jobs to parallelize
    :param n_samplings: int; number of bootstrap samplings to build distribution to get CI; must be > 2 to compute CI
    :param n_permutations: int; number of permutations for permutation test to compute P-val and FDR
    :param filepath_prefix: str; filepath_prefix_annotation_name.txt and filepath_prefix_annotation_name.pdf are saved
    :return: None
    """

    # Load target
    print_log('Loading targets bundle ...')
    target_dict = _read_data_bundle(target_bundle)

    # Load features
    print_log('Loading feature bundle ...')
    feature_dict = _read_data_bundle(feature_bundle)

    # For each target dataframe
    for t_name, t_dict in target_dict.items():

        # For each target (row) in this dataframe
        for t_i, t in t_dict['dataframe'].iterrows():

            # Annotate this target with each feature
            for f_name, f_dict in feature_dict.items():
                print_log('')
                print_log('$')
                print_log('$$')
                print_log('$$$')
                print_log('Annotating {} with {} ...'.format(t_i, f_name))

                make_association_panel(t, f_dict['dataframe'],
                                       target_type=t_dict['data_type'], feature_type=f_dict['data_type'],
                                       feature_ascending=f_dict['is_ascending'], n_features=n_features,
                                       n_jobs=n_jobs, n_samplings=n_samplings, n_permutations=n_permutations,
                                       filepath_prefix=filepath_prefix + '{}_vs_{}'.format(untitle_string(t_i),
                                                                                           untitle_string(f_name)))
                print_log('$$$')
                print_log('$$')
                print_log('$')
                print_log('')


def _read_data_bundle(data_bundle):
    """
    Read data bundle.
    :param data_bundle: list;
        [
            [name
            df or filepath to .gct
            data_type,
            is_ascending,
            (optional) index_axis,
            (optional) index,
            (optional) index_alias],

            ...

        ]
    :return: dict; {name: {dataframe: DataFrame,
                    data_type: str,
                    is_ascending: bool}}
    """

    data_dict = {}

    # Read all annotations
    for name, dataframe_or_filepath, data_type, is_ascending, index_axis, index, index_alias in data_bundle:
        print_log('Reading {} ...'.format(name))
        print_log('\tData: {}.'.format(type(dataframe_or_filepath)))
        print_log('\tData type: {}.'.format(data_type))
        print_log('\tIs ascending: {}.'.format(is_ascending))
        print_log('\tIndex axis: {}.'.format(index_axis))
        print_log('\tIndex: {}.'.format(index))
        print_log('\tIndex alias: {}.'.format(index_alias))

        data_dict[name] = {}

        # Read data type
        data_dict[name]['data_type'] = data_type

        # Read whether reverse make_association_panel or not
        data_dict[name]['is_ascending'] = is_ascending

        # Read DataFrame
        if isinstance(dataframe_or_filepath, DataFrame):
            df = dataframe_or_filepath
        elif isinstance(dataframe_or_filepath, str):
            df = read_gct(dataframe_or_filepath)
        else:
            raise ValueError('dataframe_or_filepath (2nd in the list) must be either a DataFrame or str (filepath).')

        # Limit to specified features
        if index:  # Extract

            if index_axis == 0:  # By row
                df = df.ix[index, :]
                if isinstance(df, Series):
                    df = DataFrame(df).T

            elif index_axis == 1:  # By column
                df = df.ix[:, index]
                if isinstance(df, Series):
                    df = DataFrame(df).T
                else:
                    df = df.T
            else:
                raise ValueError('index_axis (6th in the list) must be either 0 (row) or 1 (column).')

            if index_alias:  # Use aliases instead of index
                if isinstance(index_alias, str):  # Wrap string with list
                    index_alias = [index_alias]
                df.index = index_alias

        data_dict[name]['dataframe'] = df

        print_log('\tRead {} features & {} samples.'.format(*data_dict[name]['dataframe'].shape))

    return data_dict


def make_association_panel(target, features, target_type='continuous', feature_type='continuous',
                           target_sort=True, feature_ascending=False,
                           min_n_feature_values=None, n_features=0.95, n_jobs=1, min_n_per_job=30,
                           n_samplings=30, n_permutations=30,
                           figure_size='auto', title=None, title_size=16, annotation_label_size=9, plot_colname=False,
                           dpi=DPI, filepath_prefix=None):
    """
    Compute: ith score = function(ith feature, target). Compute confidence interval (CI) for n_features
    features. Compute p-val and FDR (BH) for all features. And plot the result.
    :param target: pandas Series; (n_samples); must have name and index matching features's column names
    :param features: pandas DataFrame; (n_features, n_samples); must have index and column names
    :param target_type: str; {'continuous', 'categorical', 'binary'}
    :param feature_type: str; {'continuous', 'categorical', 'binary'}
    :param target_sort: bool; sort target or not
    :param feature_ascending: bool; True if features scores increase from top to bottom, and False otherwise
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
    :return: pandas DataFrame; merged features and scores
    """

    if isinstance(features, Series):  # Convert Series-features into DataFrame-features with 1 row
        features = DataFrame(features).T

    # Use intersecting columns (samples)
    intersection = set(features.columns) & set(target.index)
    if intersection:
        target = target.ix[intersection]
        features = features.ix[:, intersection]
        print_log('Target {} ({} cols) and features ({} cols) have {} shared columns.'.format(target.name,
                                                                                              target.size,
                                                                                              features.shape[1],
                                                                                              len(intersection)))
    else:
        raise ValueError('Target {} ({} cols) and features ({} cols) have 0 shared columns.'.format(target.name,
                                                                                                    target.size,
                                                                                                    features.shape[1]))

    if target_sort:  # Sort target
        target.sort_values(ascending=False, inplace=True)
        features = features.reindex_axis(target.index, axis=1)

    # Drop features having less than min_n_feature_values unique values
    if not min_n_feature_values:
        if feature_type == 'continuous':
            min_n_feature_values = 3
        elif feature_type in ('categorical', 'binary'):
            min_n_feature_values = 2
        else:
            raise ValueError('feature_type must be one of {continuous, categorical, binary}.')
    print_log('Dropping features with less than {} unique values ...'.format(min_n_feature_values))
    features = features.ix[features.apply(lambda row: len(set(row)), axis=1) >= min_n_feature_values]
    if features.empty:
        raise ValueError('No feature has at least {} unique values.'.format(min_n_feature_values))
    else:
        print_log('\tKept {} features.'.format(features.shape[0]))

    # Score and sort
    scores = match(target, features, n_features=n_features,
                   n_jobs=n_jobs, min_n_per_job=min_n_per_job, n_samplings=n_samplings, n_permutations=n_permutations)
    # Merge features and scores, and sort by scores
    features = merge(features, scores, left_index=True, right_index=True)
    features.sort_values('score', ascending=feature_ascending, inplace=True)

    # Save
    if filepath_prefix:
        establish_filepath(filepath_prefix)
        features.to_csv(filepath_prefix + '.txt', sep='\t')

    if not (isinstance(n_features, int) or isinstance(n_features, float)):  # n_features = None
        print_log('Not plotting.')
        return features

    #
    # Make annotations
    #
    annotations = DataFrame(index=features.index)

    # Add IC(0.95 confidence interval)
    for i in annotations.index:
        if '0.95 moe' in features.columns:
            annotations.ix[i, 'IC(\u0394)'] = '{0:.3f}({1:.3f})'.format(*features.ix[i, ['score', '0.95 moe']])
        else:
            annotations.ix[i, 'IC(\u0394)'] = '{:.3f}(x.xxx)'.format(features.ix[i, 'score'])

    # Add P-val
    annotations['P-val'] = ['{:.2e}'.format(x) for x in features.ix[:, 'p-value']]

    # Add FDR
    annotations['FDR'] = ['{:.2e}'.format(x) for x in features.ix[:, 'fdr']]

    #
    # Plot
    #
    # Limited features to plot
    if n_features < 1:  # Limit using percentile
        # Limit top features
        above_quantile = features.ix[:, 'score'] >= features.ix[:, 'score'].quantile(n_features)
        print_log('Plotting {} features (> {:.03f} percentile) ...'.format(sum(above_quantile), n_features))

        # Limit bottom features
        below_quantile = features.ix[:, 'score'] <= features.ix[:, 'score'].quantile(1 - n_features)
        print_log('Plotting {} features (< {:.03f} percentile) ...'.format(sum(below_quantile), 1 - n_features))

        indices_to_plot = features.index[above_quantile | below_quantile].tolist()

    else:  # Limit using numbers
        if 2 * n_features >= features.shape[0]:
            indices_to_plot = features.index
            print_log('Plotting all {} features ...'.format(features.shape[0]))
        else:
            indices_to_plot = features.index[:n_features].tolist() + features.index[-n_features:].tolist()
            print_log('Plotting top & bottom {} features ...'.format(n_features))

    # Right alignment: ' ' * 11 + 'IC(\u0394)' + ' ' * 10 + 'P-val' + ' ' * 15 + 'FDR',
    _plot_association_panel(target,
                            features.ix[indices_to_plot, :len(scores.columns)],
                            annotations.ix[indices_to_plot, :],
                            feature_type=feature_type, target_type=target_type,
                            figure_size=figure_size, title=title, title_size=title_size,
                            annotation_header=' ' * 6 + 'IC(\u0394)' + ' ' * 12 + 'P-val' + ' ' * 14 + 'FDR',
                            annotation_label_size=annotation_label_size, plot_colname=plot_colname,
                            dpi=dpi, filepath=filepath_prefix + '.pdf')

    return features


def _plot_association_panel(target, features, annotations, target_type='continuous', feature_type='continuous',
                            std_max=3, figure_size='auto', title=None, title_size=20,
                            annotation_header=None, annotation_label_size=9, plot_colname=False,
                            dpi=DPI, filepath=None):
    """
    Plot make_association_panel panel.
    :param target: pandas Series; (n_elements); must have indices, which must make_association_panel features's columns
    :param features: pandas DataFrame; (n_features, n_elements); must have indices and columns
    :param annotations: pandas DataFrame; (n_features, n_annotations); must have indices matching features's index
    :param target_type: str; {'continuous', 'categorical', 'binary'}
    :param feature_type: str; {'continuous', 'categorical', 'binary'}
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

    # Set up target
    if target_type == 'continuous':
        target_cmap = CMAP_CONTINUOUS
        target_min, target_max = -std_max, std_max
        print_log('Normalizing continuous target ...')
        target = normalize_pandas_object(target, method='-0-')
    elif target_type == 'categorical':
        target_cmap = CMAP_CATEGORICAL
        target_min, target_max = 0, len(unique(target))
    elif target_type == 'binary':
        target_cmap = CMAP_BINARY
        target_min, target_max = 0, 1
    else:
        raise ValueError('target_type must be one of {continuous, categorical, binary}.')

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
        raise ValueError('feature_type must be one of {continuous, categorical, binary}.')

    print_log('Plotting ...')

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
