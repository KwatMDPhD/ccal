"""
Computational Cancer Analysis Library

Authors:
Pablo Tamayo
ptamayo@ucsd.edu
Computational Cancer Analysis Laboratory, UCSD Cancer Center

Huwate (Kwat) Yeerna (Medetgul-Ernar)
kwat.medetgul.ernar@gmail.com
Computational Cancer Analysis Laboratory, UCSD Cancer Center

James Jensen
jdjensen@eng.ucsd.edu
Laboratory of Jill Mesirov
"""
from pandas import DataFrame, Series, merge

from .support import print_log, establish_path, read_gct, untitle_string, information_coefficient, compare_matrices, \
    compute_against_target
from .visualize import DPI, plot_clustermap, plot_features_against_reference


def make_match_panel(annotations, filepath_prefix,
                     target_series=None,
                     target_gct=None, target_df=None, target_name=None, target_axis=1,
                     feature_type='continuous', ref_type='continuous', feature_ascending=False, ref_ascending=False):
    """

    :param annotations:
    :param filepath_prefix:
    :param target_series:
    :param target_gct:
    :param target_df:
    :param target_name:
    :param target_axis:
    :param feature_type:
    :param ref_type:
    :param feature_ascending: bool; True if features score increase from top to bottom, and False otherwise
    :param ref_ascending: bool; True if ref values increase from left to right, and False otherwise
    :return:
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

    # Make match panel
    for a_name, a_df in annotation_dfs.items():
        match(a_df, target_series, filepath_prefix + '_vs_{}'.format(untitle_string(a_name)),
              feature_type=feature_type, ref_type=ref_type,
              feature_ascending=feature_ascending, ref_ascending=ref_ascending)


def read_annotations(annotations):
    """

    :param annotations:
    :return:
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
    return annotation_dfs


def match(features, target, filepath_prefix, feature_type='continuous', ref_type='continuous', min_n_feature_values=3,
          feature_ascending=False, ref_ascending=False, ref_sort=True,
          function=information_coefficient, n_features=0.95, n_samplings=30, confidence=0.95, n_permutations=30,
          title=None, title_size=16, annotation_label_size=9, plot_colname=False,
          figure_size='auto', dpi=DPI):
    """
    Compute scores[i] = `features`[i] vs. `target` using `function`. Compute confidence interval (CI) for `n_features`
    features. Compute p-val and FDR (BH) for all features. And plot the result.
    :param features: pandas DataFrame; (n_features, n_samples); must have row and column indices
    :param target: pandas Series; (n_samples); must have name and indices, which must match `features`'s column index
    :param filepath_prefix: str; `filepath_prefix`.txt and `filepath_prefix`.pdf will be saved
    :param feature_type: str; {'continuous', 'categorical', 'binary'}
    :param ref_type: str; {'continuous', 'categorical', 'binary'}
    :param min_n_feature_values: int; minimum number of non-0 values in a feature to be matched
    :param feature_ascending: bool; True if features score increase from top to bottom, and False otherwise
    :param ref_ascending: bool; True if ref values increase from left to right, and False otherwise
    :param ref_sort: bool; sort `ref` or not
    :param function: function; scoring function
    :param n_features: int or float; number threshold if >= 1, and percentile threshold if < 1
    :param n_samplings: int; number of bootstrap samplings to build distribution to get CI; must be > 2 to compute CI
    :param confidence: float; fraction compute confidence interval
    :param n_permutations: int; number of permutations for permutation test to compute P-val and FDR
    :param title: str; plot title
    :param title_size: int; title text size
    :param annotation_label_size: int; annotation text size
    :param plot_colname: bool; plot column names or not
    :param figure_size: 'auto' or tuple;
    :param dpi: int; dots per square inch of pixel in the output figure
    :return: pandas DataFrame; scores
    """
    print_log('Matching {} against features ...'.format(target.name))

    if isinstance(features, Series):  # Convert Series into DataFrame
        features = DataFrame(features).T

    # Use intersecting columns
    col_intersection = set(features.columns) & set(target.index)
    if col_intersection:
        print_log('features ({} cols) and ref ({} cols) have {} intersecting columns.'.format(features.shape[1],
                                                                                              target.size,
                                                                                              len(col_intersection)))
        features = features.ix[:, col_intersection]
        target = target.ix[col_intersection]
    else:
        raise ValueError('features ({} cols) and ref ({} cols) have 0 intersecting columns.'.format(features.shape[1],
                                                                                                    target.size))

    # Drop features having less than `min_n_feature_values` unique values
    features = features.ix[features.apply(lambda row: len(set(row)), axis=1) >= min_n_feature_values]
    if features.empty:
        raise ValueError('No features with at least {} unique values.'.format(min_n_feature_values))

    # Sort target
    if ref_sort:
        target = target.sort_values(ascending=ref_ascending)
        features = features.reindex_axis(target.index, axis=1)

    # Compute score, P-val, FDR, and confidence interval for some features
    scores = compute_against_target(features, target, function=function,
                                    n_features=n_features, ascending=feature_ascending,
                                    n_samplings=n_samplings, confidence=confidence, n_permutations=n_permutations)
    features = features.reindex(scores.index)

    establish_path(filepath_prefix)
    merge(features, scores, left_index=True, right_index=True).to_csv(filepath_prefix + '.txt', sep='\t')

    # Make annotations
    annotations = DataFrame(index=features.index)
    for idx, s in features.iterrows():
        # Format Score and confidence interval
        if '{} MoE'.format(confidence) in scores.columns:
            annotations.ix[idx, 'IC(\u0394)'] = '{0:.3f}'.format(scores.ix[idx, 'score']) \
                                                + '({0:.3f})'.format(scores.ix[idx, '{} MoE'.format(confidence)])
        else:
            annotations.ix[idx, 'IC(\u0394)'] = '{0:.3f}(x.xxx)'.format(scores.ix[idx, 'score'])
    # Format P-Value
    annotations['P-val'] = ['{0:.3f}'.format(x) for x in scores.ix[:, 'Global P-value']]
    # Format FDR
    annotations['FDR'] = ['{0:.3f}'.format(x) for x in scores.ix[:, 'FDR']]

    if not (isinstance(n_features, int) or isinstance(n_features, float)):
        print_log('Not plotting.')

    else:  # Plot confidence interval for limited features
        if n_features < 1:  # Limit using percentile
            above_quantile = scores.ix[:, 'score'] >= scores.ix[:, 'score'].quantile(n_features)
            print_log('Plotting {} features (> {} percentile) ...'.format(sum(above_quantile), n_features))
            below_quantile = scores.ix[:, 'score'] <= scores.ix[:, 'score'].quantile(1 - n_features)
            print_log('Plotting {} features (< {} percentile) ...'.format(sum(below_quantile), 1 - n_features))
            indices_to_plot = scores.index[above_quantile | below_quantile].tolist()
        else:  # Limit using numbers
            if 2 * n_features >= scores.shape[0]:
                indices_to_plot = scores.index
                print_log('Plotting all {} features ...'.format(scores.shape[0]))
            else:
                indices_to_plot = scores.index[:n_features].tolist() + scores.index[-n_features:].tolist()
                print_log('Plotting top & bottom {} features ...'.format(n_features))

        plot_features_against_reference(features.ix[indices_to_plot, :], target, annotations.ix[indices_to_plot, :],
                                        feature_type=feature_type, ref_type=ref_type,
                                        figure_size=figure_size, title=title, title_size=title_size,
                                        annotation_header=' ' * 7 + 'IC(\u0394)' + ' ' * 9 + 'P-val' + ' ' * 4 + 'FDR',
                                        annotation_label_size=annotation_label_size, plot_colname=plot_colname,
                                        filepath=filepath_prefix + '.pdf', dpi=dpi)
    return scores


# ======================================================================================================================
# Compare 2 matrices
# ======================================================================================================================
def compare(matrix1, matrix2, function=information_coefficient, axis=0, is_distance=False, verbose=False, title=None):
    """
    Compare `matrix1` and `matrix2` row-wise (`axis=1`) or column-wise (`axis=0`), and plot hierarchical clustering.
    :param matrix1: pandas DataFrame or numpy 2D arrays;
    :param matrix2: pandas DataFrame or numpy 2D arrays;
    :param function: function; association function
    :param axis: int; 0 and 1 for row-wise and column-wise comparison respectively
    :param is_distance: bool; if True, then distances are computed from associations as in: distance = 1 - association
    :param verbose: bool; print computation progress or not
    :param title: str; plot title
    :return: pandas DataFrame; association or distance matrix
    """
    # Compute association or distance matrix, which is returned at the end
    compared_matrix = compare_matrices(matrix1, matrix2, function, axis=axis, is_distance=is_distance, verbose=verbose)

    # Plot hierarchical clustering of the matrix
    plot_clustermap(compared_matrix, title=title)

    # Return the computed association or distance matrix
    return compared_matrix
