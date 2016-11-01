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

from os.path import join
from math import ceil, sqrt

from numpy import array, sum, unique
from numpy.random import shuffle, choice
from pandas import Series, DataFrame, read_csv, concat
from scipy.stats import norm
from statsmodels.sandbox.stats.multicomp import multipletests
from matplotlib.pyplot import figure, subplot
from matplotlib.gridspec import GridSpec
from matplotlib.colorbar import make_axes, ColorbarBase
from seaborn import heatmap

from .support import print_log, establish_filepath, read_gct, title_string, untitle_string, information_coefficient, \
    parallelize, get_unique_in_order, normalize_pandas, compare_matrices, get_top_and_bottom_indices, \
    FIGURE_SIZE, SPACING, CMAP_ASSOCIATION, CMAP_CATEGORICAL, CMAP_BINARY, FONT, FONT_TITLE, FONT_SUBTITLE, save_plot, \
    plot_clustermap


# ======================================================================================================================
# Association panel
# ======================================================================================================================
def make_association_panels(target, features_bundle, target_ascending=False, target_name=None, target_type='continuous',
                            n_jobs=1, n_features=0.95, n_samplings=30, n_permutations=30, directory_path=None):
    """
    Annotate target with each features in the features bundle.
    :param target: Series; (n_elements); must have indices
    :param features_bundle: list of lists;
        [
            [
                data_name,
                dataframe_or_filepath (to .gct),
                data_type,
                is_ascending,
                (optional) index_axis,
                (optional) index,
                (optional) alias
            ],
            ...
        ]
    :param target_ascending: bool; target is ascending from left to right or not
    :param target_name: str;
    :param target_type: str;
    :param n_jobs: int; number of jobs to parallelize
    :param n_features: int or float; number threshold if >= 1, and percentile threshold if < 1
    :param n_samplings: int; number of bootstrap samplings to build distribution to get CI; must be > 2 to compute CI
    :param n_permutations: int; number of permutations for permutation test to compute P-val and FDR
    :param directory_path: str; directory_path/target_name_vs_features_name.{txt, pdf} will be saved.
    :return: None
    """

    # Load feature bundle
    print_log('Loading features bundle ...')
    feature_dicts = _read_bundle(features_bundle)

    # Annotate this target with each feature
    for features_name, features_dict in feature_dicts.items():
        title = title_string('{} vs {}'.format(target.name, features_name))
        print_log('{} ...'.format(title))
        if directory_path:
            filepath_prefix = join(directory_path, untitle_string(title))
        else:
            filepath_prefix = None
        make_association_panel(target, features_dict['dataframe'],
                               target_ascending=target_ascending,
                               n_jobs=n_jobs, features_ascending=features_dict['is_ascending'],
                               n_features=n_features, n_samplings=n_samplings, n_permutations=n_permutations,
                               target_name=target_name, target_type=target_type,
                               features_type=features_dict['data_type'],
                               title=title,
                               filepath_prefix=filepath_prefix)


def make_association_panel(target, features,
                           filepath_scores=None,
                           target_ascending=False,
                           n_jobs=1, features_ascending=False,
                           n_features=0.95, max_n_features=50, n_samplings=30, n_permutations=30,
                           target_name=None, target_type='continuous',
                           features_type='continuous',
                           title=None, plot_colname=False,
                           filepath_prefix=None):
    """
    Compute: score_i = function(target, feature_i) for all features.
    Compute confidence interval (CI) for n_features features.
    Compute P-value and FDR (BH) for all features.
    Finally plot the result.
    :param target: Series; (n_samples); must have name and index matching features's column names
    :param features: DataFrame; (n_features, n_samples); must have index and column names
    :param filepath_scores: str;
    :param target_ascending: bool;
    :param n_jobs: int; number of jobs to parallelize
    :param features_ascending: bool; True if features scores increase from top to bottom, and False otherwise
    :param n_features: int or float; number threshold if >= 1, and percentile threshold if < 1
    :param max_n_features: int;
    :param n_samplings: int; number of bootstrap samplings to build distribution to get CI; must be > 2 to compute CI
    :param n_permutations: int; number of permutations for permutation test to compute P-val and FDR
    :param target_name: str;
    :param target_type: str; {'continuous', 'categorical', 'binary'}
    :param features_type: str; {'continuous', 'categorical', 'binary'}
    :param title: str; plot title
    :param plot_colname: bool; plot column names below the plot or not
    :param filepath_prefix: str; filepath_prefix.txt and filepath_prefix.pdf will be saved
    :return: None
    """

    # Score
    if filepath_scores:  # Read already computed scores; might have been calculated with a different number of samples
        print_log('Using already computed scores ...')

        # Make sure target is a Series and features a DataFrame
        # Keep samples found in both target and features
        # Drop features with less than 2 unique values
        target, features = _preprocess_target_and_features(target, features, target_ascending=target_ascending)

        scores = read_csv(filepath_scores, sep='\t', index_col=0)

    else:  # Compute score

        if filepath_prefix:
            filepath = filepath_prefix + '.txt'
        else:
            filepath = None

        target, features, scores = compute_association(target, features,
                                                       target_ascending=target_ascending,
                                                       n_jobs=n_jobs, features_ascending=features_ascending,
                                                       n_features=n_features, n_samplings=n_samplings,
                                                       n_permutations=n_permutations,
                                                       filepath=filepath)

    # Keep only scores and features to plot
    indices_to_plot = get_top_and_bottom_indices(scores, 'score', n_features, max_n=max_n_features)
    scores = scores.ix[indices_to_plot, :]
    features = features.ix[indices_to_plot, :]

    print_log('Making annotations ...')
    annotations = DataFrame(index=scores.index, columns=['IC(\u0394)', 'P-val', 'FDR'])

    # Add IC (0.95 confidence interval), P-val, and FDR
    annotations.ix[:, 'IC(\u0394)'] = scores.ix[:, ['score', '0.95 moe']].apply(lambda s: '{0:.3f}({1:.3f})'.format(*s),
                                                                                axis=1)
    annotations.ix[:, 'P-val'] = scores.ix[:, 'p-value'].apply('{:.2e}'.format)
    annotations.ix[:, 'FDR'] = scores.ix[:, 'fdr'].apply('{:.2e}'.format)

    print_log('Plotting ...')
    if filepath_prefix:
        filepath = filepath_prefix + '.pdf'
    else:
        filepath = None
    _plot_association_panel(target, features, annotations,
                            target_name=target_name, target_type=target_type,
                            features_type=features_type,
                            title=title, plot_colname=plot_colname, filepath=filepath)


# TODO: make empty DataFrame to absorb the results instead of concatenation
def compute_association(target, features, function=information_coefficient,
                        target_ascending=False,
                        n_jobs=1, min_n_per_job=100, features_ascending=False,
                        n_features=0.95, n_samplings=30, confidence=0.95, n_permutations=30,
                        filepath=None):
    """
    Compute: score_i = function(target, feature_i) for all features.
    Compute confidence interval (CI) for n_features features.
    Compute P-value and FDR (BH) for all features.
    :param target: Series; (n_samples); must have name and indices, matching features's column index
    :param features: DataFrame; (n_features, n_samples); must have row and column indices
    :param function: function; scoring function
    :param target_ascending: bool; target is ascending or not
    :param n_jobs: int; number of jobs to parallelize
    :param min_n_per_job: int; minimum number of n per job for parallel computing
    :param features_ascending: bool; True if features scores increase from top to bottom, and False otherwise
    :param n_features: int or float; number of features to compute confidence interval and plot;
                        number threshold if >= 1, percentile threshold if < 1, and don't compute if None
    :param n_samplings: int; number of bootstrap samplings to build distribution to get CI; must be > 2 to compute CI
    :param confidence: float; fraction compute confidence interval
    :param n_permutations: int; number of permutations for permutation test to compute P-val and FDR
    :param filepath: str;
    :return: Series, DataFrame, DataFrame; (n_features, 8 ('score', '<confidence> moe',
                                            'p-value (forward)', 'p-value (reverse)', 'p-value',
                                            'fdr (forward)', 'fdr (reverse)', 'fdr'))
    """

    # Make sure target is a Series and features a DataFrame
    # Keep samples found in both target and features
    # Drop features with less than 2 unique values
    target, features = _preprocess_target_and_features(target, features, target_ascending=target_ascending)

    results = DataFrame(index=features.index, columns=['score', '{} moe'.format(confidence),
                                                       'p-value (forward)', 'p-value (reverse)', 'p-value',
                                                       'fdr (forward)', 'fdr (reverse)', 'fdr'])

    #
    # Compute: score_i = function(target, feature_i)
    #
    if n_jobs == 1:  # Non-parallel computing
        print_log('Scoring ...')
        scores = _score((target, features, function))

    else:  # Parallel computing

        # Compute n per job
        n_per_job = features.shape[0] // n_jobs
        if n_per_job < min_n_per_job:  # n is not enough for parallel computing
            print_log('Scoring (with n_jobs=1 because n_per_job ({}) < min_n_per_job ({})) ...'.format(n_per_job,
                                                                                                       min_n_per_job))
            scores = _score((target, features, function))

        else:  # n is enough for parallel computing
            print_log('Scoring (n_jobs={}) ...'.format(n_jobs))

            # Group args
            args = []
            leftovers = list(features.index)
            for i in range(n_jobs):
                split_features = features.iloc[i * n_per_job: (i + 1) * n_per_job, :]
                args.append((target, split_features, function))

                # Remove scored features
                for feature in split_features.index:
                    leftovers.remove(feature)

            # Parallelize
            scores = concat(parallelize(_score, args, n_jobs=n_jobs), verify_integrity=True)

            # Score leftovers
            if leftovers:
                print_log('Scoring leftovers: {} ...'.format(leftovers))
                scores = concat([scores, _score((target, features.ix[leftovers, :], function))], verify_integrity=True)

    # Load scores and sort results by scores
    results.ix[scores.index, 'score'] = scores
    results.sort_values('score', ascending=features_ascending, inplace=True)

    #
    #  Compute CI using bootstrapped distribution
    #
    if n_samplings < 2:
        print_log('Not computing CI because n_samplings < 2.')

    elif ceil(0.632 * features.shape[1]) < 3:
        print_log('Not computing CI because 0.632 * n_samples < 3.')

    else:
        print_log('Computing {} CI for using distributions built by {} bootstraps ...'.format(confidence, n_samplings))
        indices_to_bootstrap = get_top_and_bottom_indices(results, 'score', n_features)

        # Bootstrap: for n_sampling times, randomly choose 63.2% of the samples, score, and build score distribution
        sampled_scores = DataFrame(index=indices_to_bootstrap, columns=range(n_samplings))
        for c_i in sampled_scores:
            # Random sample
            ramdom_samples = choice(features.columns.tolist(), int(ceil(0.632 * features.shape[1]))).tolist()
            sampled_target = target.ix[ramdom_samples]
            sampled_features = features.ix[indices_to_bootstrap, ramdom_samples]

            # Score
            sampled_scores.ix[:, c_i] = sampled_features.apply(lambda f: function(sampled_target, f), axis=1)

        # Compute scores' confidence intervals using bootstrapped score distributions
        # TODO: improve confidence interval calculation
        z_critical = norm.ppf(q=confidence)

        # Load confidence interval
        results.ix[sampled_scores.index, '{} moe'.format(confidence)] = sampled_scores.apply(
            lambda f: z_critical * (f.std() / sqrt(n_samplings)), axis=1)

    #
    # Compute P-values and FDRs by sores against permuted target
    #
    if n_permutations < 1:
        print_log('Not computing P-value and FDR because n_perm < 1.')
    else:
        if n_jobs == 1:  # Non-parallel computing
            print_log('Computing P-value & FDR by scoring against {} permuted targets ...'.format(n_permutations))
            permutation_scores = _permute_and_score((target, features, function, n_permutations))

        else:  # Parallel computing

            # Compute n for a job
            n_per_job = features.shape[0] // n_jobs

            if n_per_job < min_n_per_job:  # n is not enough for parallel computing
                print_log('Computing P-value & FDR by scoring against {} permuted targets'
                          '(with n_jobs=1 because n_per_jobs ({}) < min_n_jobs ({})) ...'.format(n_permutations,
                                                                                                 n_per_job,
                                                                                                 min_n_per_job))
                permutation_scores = _permute_and_score((target, features, function, n_permutations))

            else:  # n is enough for parallel computing
                print_log('Computing P-value & FDR by scoring against {} permuted targets (n_jobs={}) ...'.format(
                    n_permutations, n_jobs))

                # Group args
                args = []
                leftovers = list(features.index)
                for i in range(n_jobs):
                    split_features = features.iloc[i * n_per_job: (i + 1) * n_per_job, :]
                    args.append((target, split_features, function, n_permutations))

                    # Remove scored features
                    for feature in split_features.index:
                        leftovers.remove(feature)

                # Parallelize
                permutation_scores = concat(parallelize(_permute_and_score, args, n_jobs=n_jobs), verify_integrity=True)

                # Handle leftovers
                if leftovers:
                    print_log('Scoring against permuted target using leftovers: {} ...'.format(leftovers))
                    permutation_scores = concat([permutation_scores, _permute_and_score(
                        (target, features.ix[leftovers, :], function, n_permutations))], verify_integrity=True)

        print_log('\tComputing P-value and FDR ...')
        # All scores
        all_permutation_scores = permutation_scores.values.flatten()
        for i, (r_i, r) in enumerate(results.iterrows()):
            # This feature's score
            s = r.ix['score']

            # Compute forward P-value
            p_value_forward = sum(all_permutation_scores >= s) / len(all_permutation_scores)
            if not p_value_forward:
                p_value_forward = float(1 / len(all_permutation_scores))
            results.ix[r_i, 'p-value (forward)'] = p_value_forward

            # Compute reverse P-value
            p_value_reverse = sum(all_permutation_scores <= s) / len(all_permutation_scores)
            if not p_value_reverse:
                p_value_reverse = float(1 / len(all_permutation_scores))
            results.ix[r_i, 'p-value (reverse)'] = p_value_reverse

        # Compute forward FDR
        results.ix[:, 'fdr (forward)'] = multipletests(results.ix[:, 'p-value (forward)'], method='fdr_bh')[1]

        # Compute reverse FDR
        results.ix[:, 'fdr (reverse)'] = multipletests(results.ix[:, 'p-value (reverse)'], method='fdr_bh')[1]

        # Creating the summary P-value and FDR
        forward = results.ix[:, 'score'] >= 0
        results.ix[:, 'p-value'] = concat([results.ix[forward, 'p-value (forward)'], results.ix[~forward,
                                                                                                'p-value (reverse)']])
        results.ix[:, 'fdr'] = concat([results.ix[forward, 'fdr (forward)'], results.ix[~forward,
                                                                                        'fdr (reverse)']])

    # Save
    if filepath:
        establish_filepath(filepath)
        results.to_csv(filepath, sep='\t')

    return target, features, results


def _preprocess_target_and_features(target, features, target_ascending=False, min_n_unique_values=2):
    """
    Make sure target is a Series and features a DataFrame.
    Keep samples found in both target and features.
    Drop features with less than 2 unique values.
    :param target: Series or iterable;
    :param features: DataFrame or Series;
    :param target_ascending: bool;
    :param min_n_unique_values: int;
    :return: Series and DataFrame;
    """

    if isinstance(features, Series):  # Convert Series-features into DataFrame-features with 1 row
        features = DataFrame(features).T

    if not isinstance(target, Series):  # Convert target into a Series
        target = Series(target, index=features.columns)

    # Keep only columns shared by target and features
    shared = target.index & features.columns
    if any(shared):
        print_log('Target {} ({} cols) and features ({} cols) have {} shared columns.'.format(target.name,
                                                                                              target.size,
                                                                                              features.shape[1],
                                                                                              len(shared)))
        target = target.ix[shared].sort_values(ascending=target_ascending)
        features = features.ix[:, target.index]
    else:
        raise ValueError('Target {} ({} cols) and features ({} cols) have 0 shared columns.'.format(target.name,
                                                                                                    target.size,
                                                                                                    features.shape[1]))

    # Drop features having less than 2 unique values
    print_log('Dropping features with less than {} unique values ...'.format(min_n_unique_values))
    features = features.ix[features.apply(lambda f: len(set(f)), axis=1) >= min_n_unique_values]
    if features.empty:
        raise ValueError('No feature has at least {} unique values.'.format(min_n_unique_values))
    else:
        print_log('\tKept {} features.'.format(features.shape[0]))

    return target, features


def _score(args):
    """
    Compute: score_i = function(target, feature_i)
    :param args: list-like; [DataFrame (n_features, m_samples); features, Series (m_samples); target, function]
    :return: Series; (n_features)
    """

    t, f, func = args
    return f.apply(lambda a_f: func(t, a_f), axis=1)


def _permute_and_score(args):
    """
    Compute: ith score = function(target, ith feature) for n_permutations times.
    :param args: list-like;
        (Series (m_samples); target,
         DataFrame (n_features, m_samples); features,
         function,
         int; n_permutations)
    :return: DataFrame; (n_features, n_permutations)
    """

    t, f, func, n_perms = args

    scores = DataFrame(index=f.index, columns=range(n_perms))
    shuffled_target = array(t)
    for p in range(n_perms):
        print_log('\tScoring against permuted target ({}/{}) ...'.format(p, n_perms))
        shuffle(shuffled_target)
        scores.iloc[:, p] = f.apply(lambda r: func(shuffled_target, r), axis=1)
    return scores


def _plot_association_panel(target, features, annotations,
                            target_name=None, target_type='continuous',
                            features_type='continuous',
                            title=None, plot_colname=False,
                            filepath=None):
    """
    Plot association panel.
    :param target: Series; (n_elements); must have indices matching features's columns
    :param features: DataFrame; (n_features, n_elements); must have indices and columns
    :param annotations: DataFrame; (n_features, n_annotations); must have indices matching features's index
    :param target_name: str;
    :param target_type: str; {'continuous', 'categorical', 'binary'}
    :param features_type: str; {'continuous', 'categorical', 'binary'}
    :param title: str;
    :param plot_colname: bool; plot column names or not
    :param filepath: str;
    :return: None
    """

    # Prepare target for plotting
    target, target_min, target_max, target_cmap = _prepare_data_for_plotting(target, target_type)
    if target_name:
        target.name = target_name

    # Prepare features for plotting
    features, features_min, features_max, features_cmap = _prepare_data_for_plotting(features, features_type)

    # Set up figure
    figure(figsize=(min(pow(features.shape[1], 0.7), 7), pow(features.shape[0], 0.9)))

    # Set up axis grids
    gridspec = GridSpec(features.shape[0] + 1, 1)
    # Set up axes
    target_ax = subplot(gridspec[:1, 0])
    features_ax = subplot(gridspec[1:, 0])

    #
    # Plot target, target label, and title
    #
    # Plot target
    heatmap(DataFrame(target).T, ax=target_ax, vmin=target_min, vmax=target_max, cmap=target_cmap,
            xticklabels=False, cbar=False)

    # Adjust target name
    for t in target_ax.get_yticklabels():
        t.set(rotation=0, **FONT)

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
        for i, x in enumerate(label_horizontal_positions):
            target_ax.text(x, target_ax.axis()[3] * (1 + SPACING), unique_target_labels[i],
                           horizontalalignment='center', **FONT)

    if title:  # Plot title
        target_ax.text(target_ax.axis()[1] * 0.5, target_ax.axis()[3] * 1.9, title, horizontalalignment='center',
                       **FONT_TITLE)

    # Plot annotation header
    target_ax.text(target_ax.axis()[1] + target_ax.axis()[1] * SPACING, target_ax.axis()[3] * 0.5,
                   ' ' * 6 + 'IC(\u0394)' + ' ' * 12 + 'P-val' + ' ' * 14 + 'FDR', verticalalignment='center', **FONT)

    # Plot features
    heatmap(features, ax=features_ax, vmin=features_min, vmax=features_max, cmap=features_cmap,
            xticklabels=plot_colname, cbar=False)
    for t in features_ax.get_yticklabels():
        t.set(rotation=0, **FONT)

    # Plot annotations
    for i, (a_i, a) in enumerate(annotations.iterrows()):
        features_ax.text(features_ax.axis()[1] + features_ax.axis()[1] * SPACING, features_ax.axis()[3] - i - 0.5,
                         '\t'.join(a.tolist()).expandtabs(), verticalalignment='center', **FONT)

    # Save
    if filepath:
        save_plot(filepath)


def plot_association_summary_panel(target, features_bundle, annotations_bundle, target_type='continuous', title=None,
                                   filepath=None):
    """
    Plot summary association panel.
    :param target: Series; (n_elements); must have indices
    :param features_bundle: list;
        [
            [
                data_name,
                dataframe_or_filepath (to .gct),
                data_type,
                is_ascending,
                (optional) index_axis,
                (optional) index,
                (optional) alias
            ],
            ...
        ]
    :param annotations_bundle:
        [
            [
                data_name,
                dataframe_or_filepath (to annotation .gct)
            ],
            ...
        ]
    :param target_type: str;
    :param title; str;
    :param filepath: str;
    :return: None
    """

    # Read features
    features_dicts = _read_bundle(features_bundle)

    # Prepare target for plotting
    target, target_min, target_max, target_cmap = _prepare_data_for_plotting(target, target_type)

    #
    # Set up figure
    #
    # Compute the number of row-grids for setting up a figure
    n = 0
    for features_name, features_dict in features_dicts.items():
        n += features_dict['dataframe'].shape[0] + 3
    # Add row for color bar
    n += 1

    # Set up figure
    figure(figsize=FIGURE_SIZE)

    # Set up axis grids
    gridspec = GridSpec(n, 1)

    #
    # Annotate this target with each feature
    #
    r_i = 0
    if not title:
        title = 'Association Summary Panel for {}'.format(target.name)
    plot_figure_title = True
    plot_header = True
    for features_name in [b[0] for b in features_bundle]:

        # Read feature dict
        features_dict = features_dicts[features_name]

        # Read features
        features = features_dict['dataframe']

        # Prepare features for plotting
        features, features_min, features_max, features_cmap = _prepare_data_for_plotting(features,
                                                                                         features_dict['data_type'])

        # Keep only columns shared by target and features
        shared = target.index & features.columns
        if any(shared):
            # Target is always descending from right to left
            a_target = target.ix[shared].sort_values(ascending=False)
            features = features.ix[:, a_target.index]
            print_log('Target {} ({} cols) and features ({} cols) have {} shared columns.'.format(target.name,
                                                                                                  target.size,
                                                                                                  features.shape[1],
                                                                                                  len(shared)))
        else:
            raise ValueError('Target {} ({} cols) and features ({} cols) have 0 shared columns.'.format(target.name,
                                                                                                        target.size,
                                                                                                        features.shape[
                                                                                                            1]))

        # Read corresponding annotations file
        annotations = read_csv([a[1] for a in annotations_bundle if a[0] == features_name][0], sep='\t', index_col=0)
        # Keep only features in the features dataframe and sort by score
        annotations = annotations.ix[features.index, :].sort_values('score', ascending=features_dict['is_ascending'])
        # Apply the sorted index to featuers
        features = features.ix[annotations.index, :]

        if any(features_dict['alias']):  # Use alias as index
            features.index = [features_dict['alias'][i] for i in features.index]
            annotations.index = [features_dict['alias'][i] for i in annotations.index]

        # Set up axes
        r_i += 1
        title_ax = subplot(gridspec[r_i: r_i + 1, 0])
        title_ax.axis('off')
        r_i += 1
        target_ax = subplot(gridspec[r_i: r_i + 1, 0])
        r_i += 1
        features_ax = subplot(gridspec[r_i: r_i + features.shape[0], 0])
        r_i += features.shape[0]

        if plot_figure_title:  # Plot figure title
            title_ax.text(title_ax.axis()[1] * 0.5, title_ax.axis()[3] * 2, title, horizontalalignment='center',
                          **FONT_TITLE)
            plot_figure_title = False

        # Plot title
        title_ax.text(title_ax.axis()[1] * 0.5, title_ax.axis()[3] * 0.3,
                      '{} (n={})'.format(features_name, len(shared)), horizontalalignment='center', **FONT_SUBTITLE)

        # Plot target
        heatmap(DataFrame(a_target).T, ax=target_ax, vmin=target_min, vmax=target_max, cmap=target_cmap,
                xticklabels=False, yticklabels=True, cbar=False)
        for t in target_ax.get_yticklabels():
            t.set(rotation=0, **FONT)

        if plot_header:  # Plot header only for the 1st target axis
            target_ax.text(target_ax.axis()[1] + target_ax.axis()[1] * SPACING, target_ax.axis()[3] * 0.5,
                           ' ' * 1 + 'IC(\u0394)' + ' ' * 6 + 'P-val' + ' ' * 15 + 'FDR', verticalalignment='center',
                           **FONT)
            plot_header = False

        # Plot features
        heatmap(features, ax=features_ax, vmin=features_min, vmax=features_max, cmap=features_cmap, xticklabels=False,
                cbar=False)
        for t in features_ax.get_yticklabels():
            t.set(rotation=0, **FONT)

        # Plot annotations
        for i, (a_i, a) in enumerate(annotations.iterrows()):
            # TODO: add confidence interval
            features_ax.text(features_ax.axis()[1] + features_ax.axis()[1] * SPACING,
                             features_ax.axis()[3] - i * (features_ax.axis()[3] / features.shape[0]) - 0.5,
                             '{0:.3f}\t{1:.2e}\t{2:.2e}'.format(*a.ix[['score', 'p-value', 'fdr']]).expandtabs(),
                             verticalalignment='center', **FONT)

        # Plot colorbar
        if r_i == n - 1:
            colorbar_ax = subplot(gridspec[r_i:r_i + 1, 0])
            colorbar_ax.axis('off')
            cax, kw = make_axes(colorbar_ax, location='bottom', pad=0.026, fraction=0.26, shrink=2.6, aspect=26,
                                cmap=target_cmap, ticks=[])
            ColorbarBase(cax, **kw)
            cax.text(cax.axis()[1] * 0.5, cax.axis()[3] * -2.6, 'Standardized Profile for Target and Features',
                     horizontalalignment='center', **FONT)
    # Save
    if filepath:
        save_plot(filepath)


def _prepare_data_for_plotting(dataframe, data_type, max_std=3):
    if data_type == 'continuous':
        return normalize_pandas(dataframe, method='-0-', axis=1), -max_std, max_std, CMAP_ASSOCIATION
    elif data_type == 'categorical':
        return dataframe.copy(), 0, len(unique(dataframe)), CMAP_CATEGORICAL
    elif data_type == 'binary':
        return dataframe.copy(), 0, 1, CMAP_BINARY
    else:
        raise ValueError('Target data type must be one of {continuous, categorical, binary}.')


def _read_bundle(data_bundle):
    """
    Read data bundle.
    :param data_bundle: list;
        [
            [
                data_name,
                dataframe_or_filepath (to .gct),
                data_type,
                is_ascending,
                (optional) index_axis,
                (optional) index,
                (optional) alias
            ],
            ...
        ]
    :return: dict;
        {
            name: {
                dataframe: DataFrame,
                data_type: str,
                is_ascending: bool,
                alias: dict {index: alias, ... }
            }
            ...
        }
    """

    dicts = {}

    # Read all annotations
    for data_name, dataframe_or_filepath, data_type, is_ascending, index_axis, index, alias in data_bundle:
        print_log('Reading {} ...'.format(data_name))
        print_log('\tData: {}.'.format(type(dataframe_or_filepath)))
        print_log('\tData type: {}.'.format(data_type))
        print_log('\tIs ascending: {}.'.format(is_ascending))
        print_log('\tIndex axis: {}.'.format(index_axis))
        print_log('\tIndex: {}.'.format(index))
        print_log('\tAlias: {}.'.format(alias))

        dicts[data_name] = {}

        # Read data type
        dicts[data_name]['data_type'] = data_type

        # Read whether reverse make_association_panel or not
        dicts[data_name]['is_ascending'] = is_ascending

        # Read alias for index
        dicts[data_name]['alias'] = {}
        for i, a in zip(index, alias):
            dicts[data_name]['alias'][i] = a

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

        dicts[data_name]['dataframe'] = df

        print_log('\tRead {} features & {} samples.'.format(*dicts[data_name]['dataframe'].shape))

    return dicts


# ======================================================================================================================
# Comparison panel
# ======================================================================================================================
def make_comparison_matrix(matrix1, matrix2, matrix1_label='Matrix 1', matrix2_label='Matrix 2',
                           function=information_coefficient, axis=0, is_distance=False, title=None,
                           filepath_prefix=None):
    """
    Compare matrix1 and matrix2 by row (axis=1) or by column (axis=0), and plot cluster map.
    :param matrix1: DataFrame or numpy 2D arrays;
    :param matrix2: DataFrame or numpy 2D arrays;
    :param matrix1_label: str;
    :param matrix2_label: str;
    :param function: function; association or distance function
    :param axis: int; 0 for row-wise and 1 for column-wise comparison
    :param is_distance: bool; if True, distances are computed from associations, as in 'distance = 1 - association'
    :param title: str; plot title
    :param filepath_prefix: str; filepath_prefix.txt and filepath_prefix.pdf will be saved
    :return: DataFrame; association or distance matrix
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
    plot_clustermap(comparison_matrix, title=title, xlabel=matrix2_label, ylabel=matrix1_label, filepath=filepath)

    return comparison_matrix
