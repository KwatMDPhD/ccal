"""
Computational Cancer Analysis Library

Authors:
    Huwate (Kwat) Yeerna (Medetgul-Ernar)
        kwat.medetgul.ernar@gmail.com
        Computational Cancer Analysis Laboratory, UCSD Cancer Center

    Pablo Tamayo - the author of the version implemented in R
        ptamayo@ucsd.edu
        Computational Cancer Analysis Laboratory, UCSD Cancer Center
"""

from pandas import DataFrame, Series, merge

from .support import print_log, establish_path, read_gct, untitle_string, information_coefficient, compare_matrices, \
    compute_against_target
from .visualize import DPI, plot_clustermap, plot_features_against_target


# ======================================================================================================================
# Match features against target
# ======================================================================================================================
# TODO: implement
def catalogue(annotations, filepath_prefix,
              target_series=None, target_gct=None, target_df=None, target_name=None, target_axis=1,
              feature_type='continuous', target_type='continuous', feature_ascending=False):
    """
    Annotate target using multiple annotations.
    :param annotations: list of lists; [[name, filepath_to_gct, [row_name1, row_name2, ...](optional)], ...]
    :param filepath_prefix: str; filepath_prefix + '_vs_name.txt' and filepath_prefix + '.pdf' will be saved
    :param target_series: pandas Series; annotation target
    :param target_gct: str; filepath to a file whose row or column is the annotation target
    :param target_df: DataFrame; whose row or column is the annotation target
    :param target_name: str; row or column name in target_df, either directly passed or read from target_gct
    :param target_axis: int; axis on which target_name is found in target_df, either directly passed or read from
        target_gct
    :param feature_type: str; {continuous, categorical, binary}
    :param target_type: str; {continuous, categorical, binary}
    :param feature_ascending: bool; True if features score_dataframe_against_series increase from top to bottom, and
        False otherwise
    :return: None
    """
    # Load target
    if not target_series:
        print_log('Loading the annotation target ...')

        # Load and check target_df
        if target_gct:
            target_df = read_gct(target_gct)
        if not isinstance(target_df, DataFrame):
            raise ValueError('No target_df {} ({}).'.format(target_df, type(target_df)))

        # Check target_name
        if not target_name:
            raise ValueError('No target_name {} ({}).'.format(target_name, type(target_name)))

        # Load target_series
        if target_axis == 0:
            target_series = target_df.ix[:, target_name]
        elif target_axis == 1:
            target_series = target_df.ix[target_name, :]
        else:
            raise ValueError('Unknown target_axis {}.'.format(target_axis))

    # Load annotations
    annotation_dfs = read_annotations(annotations)

    # Match target will all annotations
    for a_name, a_df in annotation_dfs.items():
        match(a_df, target_series, filepath_prefix + '_vs_{}'.format(untitle_string(a_name)),
              feature_type=feature_type, target_type=target_type, feature_ascending=feature_ascending)


def read_annotations(annotations):
    """
    Read annotations from .gct files.
    :param annotations: list of lists; [[name, filepath_to_gct, [row_name1, row_name2, ...](optional)], ...]
    :return: list of pandas DataFrames;
    """
    annotation_dfs = {}
    for a in annotations:
        print_log('Reading annotation: {} ...'.format(' ~ '.join([str(x) for x in a])))
        try:  # Filter with features
            a_name, a_file, a_features = a
            annotation_dfs[a_name] = read_gct(a_file).ix[a_features, :]
        except ValueError:  # Use all features
            a_name, a_file = a
            annotation_dfs[a_name] = read_gct(a_file)
        print_log('\t{} features & {} samples.'.format(*annotation_dfs[a_name].shape))
    return annotation_dfs


def match(features, target, filepath_prefix, feature_type='continuous', target_type='continuous',
          min_n_feature_values=1, feature_ascending=False, target_sort=True,
          n_features=0.95, n_jobs=1, min_n_per_job=100, n_samplings=30, n_permutations=30,
          figure_size='auto', title=None, title_size=16, annotation_label_size=9, plot_colname=False, dpi=DPI):
    """
    Compute scores[i] = `features`[i] vs. `target` using `function`. Compute confidence interval (CI) for `n_features`
    features. Compute p-val and FDR (BH) for all features. And plot the result.
    :param features: pandas DataFrame; (n_features, n_samples); must have row and column indices
    :param target: pandas Series; (n_samples); must have name and indices, which must match `features`'s column index
    :param filepath_prefix: str; `filepath_prefix`.txt and `filepath_prefix`.pdf will be saved
    :param feature_type: str; {'continuous', 'categorical', 'binary'}
    :param target_type: str; {'continuous', 'categorical', 'binary'}
    :param min_n_feature_values: int; minimum number of non-0 values in a feature to be matched
    :param feature_ascending: bool; True if features score_dataframe_against_series increase from top to bottom, and
        False otherwise
    :param target_sort: bool; sort `target` or not
    :param n_features: int or float; number threshold if >= 1, and percentile threshold if < 1
    :param n_jobs: int; number of jobs to parallelize
    :param min_n_per_job: int; minimum number of n per job for parallel computing
    :param n_samplings: int; number of bootstrap samplings to build distribution to get CI; must be > 2 to compute CI
    :param n_permutations: int; number of permutations for permutation test to compute P-val and FDR
    :param figure_size: 'auto' or tuple;
    :param title: str; plot title
    :param title_size: int; title text size
    :param annotation_label_size: int; annotation text size
    :param plot_colname: bool; plot column names or not
    :param dpi: int; dots per square inch of pixel in the output figure
    :return: pandas DataFrame; scores
    """
    print_log('Matching {} against features ...'.format(target.name))

    if isinstance(features, Series):  # Convert Series into DataFrame
        features = DataFrame(features).T

    # Use intersecting columns
    col_intersection = set(features.columns) & set(target.index)
    if col_intersection:
        print_log('features ({} cols) and target ({} cols) have {} intersecting columns.'.format(features.shape[1],
                                                                                                 target.size,
                                                                                                 len(col_intersection)))
        features = features.ix[:, col_intersection]
        target = target.ix[col_intersection]
    else:
        raise ValueError(
            'features ({} cols) and target ({} cols) have 0 intersecting columns.'.format(features.shape[1],
                                                                                          target.size))

    # Drop features having less than `min_n_feature_values` unique values
    print_log('Dropping features with less than {} unique values ...'.format(min_n_feature_values))
    features = features.ix[features.apply(lambda row: len(set(row)), axis=1) >= min_n_feature_values]
    if features.empty:
        raise ValueError('No feature has at least {} unique values.'.format(min_n_feature_values))
    else:
        print_log('\tKept {} features.'.format(features.shape[0]))

    # Sort target
    if target_sort:
        target.sort_values(ascending=False, inplace=True)
        features = features.reindex_axis(target.index, axis=1)

    # Score
    scores = compute_against_target(features, target, n_features=n_features, ascending=feature_ascending,
                                    n_jobs=n_jobs, min_n_per_job=min_n_per_job,
                                    n_samplings=n_samplings, n_permutations=n_permutations)
    features = features.reindex(scores.index)

    # Save features merged with their scores
    establish_path(filepath_prefix)
    features_and_scores = merge(features, scores, left_index=True, right_index=True)
    features_and_scores.to_csv(filepath_prefix + '.txt', sep='\t')

    # Make annotations
    if not (isinstance(n_features, int) or isinstance(n_features, float)):
        print_log('Not plotting.')

    else:  # Make annotations and plot
        # Make annotations
        annotations = DataFrame(index=features.index)

        # Format Score and confidence interval
        for idx, s in features.iterrows():
            if '0.95 MoE' in scores.columns:
                annotations.ix[idx, 'IC(\u0394)'] = '{0:.3f}({1:.3f})'.format(*scores.ix[idx, ['Score', '0.95 MoE']])
            else:
                annotations.ix[idx, 'IC(\u0394)'] = '{:.3f}(x.xxx)'.format(scores.ix[idx, 'Score'])

        # Format P-Value
        annotations['P-val'] = ['{:.3f}'.format(x) for x in scores.ix[:, 'P-value']]

        # Format FDR
        annotations['FDR'] = ['{:.3f}'.format(x) for x in scores.ix[:, 'FDR']]

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
        plot_features_against_target(features.ix[indices_to_plot, :], target, annotations.ix[indices_to_plot, :],
                                     feature_type=feature_type, target_type=target_type,
                                     figure_size=figure_size, title=title, title_size=title_size,
                                     annotation_header=' ' * 11 + 'IC(\u0394)' + ' ' * 5 + 'P-val' + ' ' * 4 + 'FDR',
                                     annotation_label_size=annotation_label_size, plot_colname=plot_colname,
                                     filepath=filepath_prefix + '.pdf', dpi=dpi)
    return features_and_scores


# ======================================================================================================================
# Compare matrices
# ======================================================================================================================
def compare(matrix1, matrix2, function=information_coefficient, axis=0, is_distance=False,
            title=None, filepath_prefix=None):
    """
    Compare `matrix1` and `matrix2` row-wise (`axis=1`) or column-wise (`axis=0`), and plot hierarchical clustering.
    :param matrix1: pandas DataFrame or numpy 2D arrays;
    :param matrix2: pandas DataFrame or numpy 2D arrays;
    :param function: function; association function
    :param axis: int; 0 and 1 for row-wise and column-wise comparison respectively
    :param is_distance: bool; if True, distances are computed from associations, as in 'distance = 1 - association'
    :param title: str; plot title
    :param filepath_prefix: str;
    :return: pandas DataFrame; association or distance matrix
    """
    # Compute association or distance matrix, which is returned at the end
    compared_matrix = compare_matrices(matrix1, matrix2, function, axis=axis, is_distance=is_distance)

    # Save
    if filepath_prefix:
        compared_matrix.to_csv(filepath_prefix + '.txt', sep='\t')

    # Plot hierarchical clustering of the matrix
    if filepath_prefix:
        filepath = filepath_prefix + '.pdf'
    else:
        filepath = None
    plot_clustermap(compared_matrix, title=title, filepath=filepath)

    # Return the computed association or distance matrix
    return compared_matrix
