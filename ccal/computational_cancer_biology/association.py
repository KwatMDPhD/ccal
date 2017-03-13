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

from math import ceil, sqrt
from os.path import join

from matplotlib.colorbar import make_axes, ColorbarBase
from matplotlib.gridspec import GridSpec
from matplotlib.pyplot import figure, subplot
from numpy import array, unique
from numpy.random import seed, set_state, shuffle, choice, get_state
from pandas import Series, DataFrame, read_csv, concat
from scipy.stats import norm
from seaborn import heatmap
from statsmodels.sandbox.stats.multicomp import multipletests

from .. import RANDOM_SEED
from ..machine_learning.score import compute_similarity_matrix
from ..mathematics.information import information_coefficient
from ..support.d1 import get_unique_in_order
from ..support.d2 import get_top_and_bottom_indices, split_dataframe, normalize_2d_or_1d
from ..support.file import establish_filepath
from ..support.log import print_log
from ..support.parallel_computing import parallelize
from ..support.plot import FIGURE_SIZE, SPACING, CMAP_ASSOCIATION, CMAP_CATEGORICAL, CMAP_BINARY, FONT, FONT_TITLE, \
    FONT_SUBTITLE, save_plot, plot_clustermap
from ..support.str_ import title_str, untitle_str


# ======================================================================================================================
# Association summary panel
# ======================================================================================================================
def make_association_summary_panel(target, data_bundle, annotation_files, order=(), target_ascending=False,
                                   target_type='continuous', title=None, filepath=None):
    """
    Plot summary association panel.
    :param target: Series; (n_elements);
    :param data_bundle: dict;
    :param annotation_files: dict;
    :param order: iterable;
    :param target_ascending: bool;
    :param target_type: str;
    :param title; str;
    :param filepath: str;
    :return: None
    """

    # Prepare target for plotting
    target, target_min, target_max, target_cmap = _prepare_data_for_plotting(target, target_type)

    #
    # Set up figure
    #
    # Compute the number of row-grids for setting up a figure
    n = 0
    for features_name, features_dict in data_bundle.items():
        n += features_dict['dataframe'].shape[0] + 3
    # Add a row for color bar
    n += 1
    # Set up figure
    fig = figure(figsize=FIGURE_SIZE)
    # Set up axis grids
    gridspec = GridSpec(n, 1)

    #
    # Annotate target with features
    #
    r_i = 0
    if not title:
        title = 'Association Summary Panel for {}'.format(title_str(target.name))
    fig.suptitle(title, horizontalalignment='center', **FONT_TITLE)
    plot_annotation_header = True

    if not any(order):  # Sort alphabetically if order is not given
        order = sorted(data_bundle.keys())
    for features_name, features_dict in [(k, data_bundle[k]) for k in order]:

        # Read features
        features = features_dict['dataframe']

        # Prepare features for plotting
        features, features_min, features_max, features_cmap = _prepare_data_for_plotting(features,
                                                                                         features_dict['data_type'])

        # Keep only columns shared by target and features
        shared = target.index & features.columns
        if any(shared):
            a_target = target.ix[shared].sort_values(ascending=target_ascending)
            features = features.ix[:, a_target.index]
            print_log('Target {} ({} cols) and features ({} cols) have {} shared columns.'.format(target.name,
                                                                                                  target.size,
                                                                                                  features.shape[1],
                                                                                                  len(shared)))
        else:
            raise ValueError('Target {} ({} cols) and features ({} cols) have 0 shared column.'.format(target.name,
                                                                                                       target.size,
                                                                                                       features.shape[
                                                                                                           1]))

        # Read corresponding annotations file
        annotations = read_csv(annotation_files[features_name], sep='\t', index_col=0)
        # Keep only features in the features dataframe and sort by score
        annotations = annotations.ix[features_dict['original_index'], :]
        annotations.index = features.index
        annotations.sort_values('score', ascending=features_dict['emphasis'] == 'low')

        # Apply the sorted index to featuers
        features = features.ix[annotations.index, :]

        # TODO: update logic and consider removing this
        # if any(features_dict['alias']):  # Use alias as index
        #     features.index = features_dict['alias']
        #     annotations.index = features.index

        # Set up axes
        r_i += 1
        title_ax = subplot(gridspec[r_i: r_i + 1, 0])
        title_ax.axis('off')
        r_i += 1
        target_ax = subplot(gridspec[r_i: r_i + 1, 0])
        r_i += 1
        features_ax = subplot(gridspec[r_i: r_i + features.shape[0], 0])
        r_i += features.shape[0]

        # Plot title
        title_ax.text(title_ax.axis()[1] * 0.5, title_ax.axis()[3] * 0.3,
                      '{} (n={})'.format(title_str(features_name), len(shared)),
                      horizontalalignment='center', **FONT_SUBTITLE)

        # Plot target
        heatmap(DataFrame(a_target).T, ax=target_ax, vmin=target_min, vmax=target_max, cmap=target_cmap,
                xticklabels=False, yticklabels=True, cbar=False)
        for t in target_ax.get_yticklabels():
            t.set(rotation=0, **FONT)

        if plot_annotation_header:  # Plot header only for the 1st target axis
            target_ax.text(target_ax.axis()[1] + target_ax.axis()[1] * SPACING, target_ax.axis()[3] * 0.5,
                           ' ' * 1 + 'IC(\u0394)' + ' ' * 6 + 'P-val' + ' ' * 15 + 'FDR', verticalalignment='center',
                           **FONT)
            plot_annotation_header = False

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
    save_plot(filepath)


# ======================================================================================================================
# Association panel
# ======================================================================================================================
def make_association_panels(target, data_bundle, dropna='all', target_ascending=False,
                            target_prefix='', data_prefix='',
                            target_type='continuous',
                            n_jobs=1, n_features=0.95, n_samplings=30, n_permutations=30, random_seed=RANDOM_SEED,
                            directory_path=None):
    """
    Annotate target with each features in the features bundle.
    :param target: DataFrame or Series; (n_targets, n_elements) or (n_elements)
    :param data_bundle: dict;
    :param dropna: str; 'any' or 'all'
    :param target_ascending: bool; target is ascending from left to right or not
    :param target_prefix: str; prefix added before the target name
    :param data_prefix: str; prefix added before the data name
    :param target_type: str;
    :param n_jobs: int; number of jobs to parallelize
    :param n_features: int or float; number threshold if >= 1, and percentile threshold if < 1
    :param n_samplings: int; number of bootstrap samplings to build distribution to get CI; must be > 2 to compute CI
    :param n_permutations: int; number of permutations for permutation test to compute P-val and FDR
    :param random_seed: int;
    :param directory_path: str; directory_path/target_name_vs_features_name.{txt, pdf} will be saved.
    :return: None
    """

    if isinstance(target, Series):
        target = DataFrame(target).T

    for t_i, t in target.iterrows():

        # Annotate this target with each data (feature)
        for data_name, data_dict in data_bundle.items():

            if target_prefix and not target_prefix.endswith(' '):
                target_prefix += ' '
            if data_prefix and not data_prefix.endswith(' '):
                data_prefix += ' '
            title = title_str('{}{} vs {}{}'.format(target_prefix, t_i, data_prefix, data_name))
            print('{} ...'.format(title))

            if directory_path:
                filepath_prefix = join(directory_path, untitle_str(title))
            else:
                filepath_prefix = None

            make_association_panel(t, data_dict['dataframe'], dropna=dropna,
                                   target_ascending=target_ascending,
                                   n_jobs=n_jobs, features_ascending=data_dict['emphasis'] == 'low',
                                   n_features=n_features, n_samplings=n_samplings, n_permutations=n_permutations,
                                   random_seed=random_seed,
                                   target_name=t_i, target_type=target_type,
                                   features_type=data_dict['data_type'],
                                   title=title,
                                   filepath_prefix=filepath_prefix)


def make_association_panel(target, features, dropna='all', filepath_scores=None,
                           target_ascending=False, features_ascending=False,
                           n_jobs=1, n_features=0.95, max_n_features=100, n_samplings=30, n_permutations=30,
                           random_seed=RANDOM_SEED,
                           target_name=None, target_type='continuous', features_type='continuous',
                           title=None, plot_colname=False, filepath_prefix=None):
    """
    Compute: score_i = function(target, feature_i) for all features.
    Compute confidence interval (CI) for n_features features.
    Compute P-value and FDR (BH) for all features.
    Finally plot the result.
    :param target: Series; (n_samples); must have name and index matching features's column names
    :param features: DataFrame; (n_features, n_samples); must have index and column names
    :param dropna: str; 'any' or 'all'
    :param filepath_scores: str;
    :param target_ascending: bool;
    :param n_jobs: int; number of jobs to parallelize
    :param features_ascending: bool; True if features scores increase from top to bottom, and False otherwise
    :param n_features: int or float; number threshold if >= 1, and percentile threshold if < 1
    :param max_n_features: int;
    :param n_samplings: int; number of bootstrap samplings to build distribution to get CI; must be > 2 to compute CI
    :param n_permutations: int; number of permutations for permutation test to compute P-val and FDR
    :param random_seed: int;
    :param target_name: str;
    :param target_type: str; {'continuous', 'categorical', 'binary'}
    :param features_type: str; {'continuous', 'categorical', 'binary'}
    :param title: str; plot title
    :param plot_colname: bool; plot column names below the plot or not
    :param filepath_prefix: str; filepath_prefix.txt and filepath_prefix.pdf will be saved
    :return: DataFrame; (n_features, 8 ('score', '<confidence> moe',
                                        'p-value (forward)', 'p-value (reverse)', 'p-value',
                                        'fdr (forward)', 'fdr (reverse)', 'fdr'))
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

        target, features, scores = compute_association(target, features, dropna=dropna,
                                                       target_ascending=target_ascending,
                                                       n_jobs=n_jobs, features_ascending=features_ascending,
                                                       n_features=n_features, n_samplings=n_samplings,
                                                       n_permutations=n_permutations,
                                                       random_seed=random_seed,
                                                       filepath=filepath)

    # Keep only scores and features to plot
    indices_to_plot = get_top_and_bottom_indices(scores, 'score', n_features, max_n=max_n_features)
    scores_to_plot = scores.ix[indices_to_plot, :]
    features_to_plot = features.ix[indices_to_plot, :]

    print_log('Making annotations ...')
    annotations = DataFrame(index=scores_to_plot.index, columns=['IC(\u0394)', 'P-val', 'FDR'])

    # Add IC (0.95 confidence interval), P-val, and FDR
    annotations.ix[:, 'IC(\u0394)'] = scores_to_plot.ix[:, ['score', '0.95 moe']].apply(
        lambda s: '{0:.3f}({1:.3f})'.format(*s), axis=1)
    annotations.ix[:, 'P-val'] = scores_to_plot.ix[:, 'p-value'].apply('{:.2e}'.format)
    annotations.ix[:, 'FDR'] = scores_to_plot.ix[:, 'fdr'].apply('{:.2e}'.format)

    print_log('Plotting ...')
    if filepath_prefix:
        filepath = filepath_prefix + '.pdf'
    else:
        filepath = None
    _plot_association_panel(target, features_to_plot, annotations,
                            target_name=target_name, target_type=target_type,
                            features_type=features_type,
                            title=title, plot_colname=plot_colname, filepath=filepath)

    return scores


def compute_association(target, features, function=information_coefficient, dropna='all',
                        target_ascending=False, features_ascending=False,
                        n_jobs=1, min_n_per_job=100, n_features=0.95, n_samplings=30, confidence=0.95,
                        n_permutations=30,
                        random_seed=RANDOM_SEED,
                        filepath=None):
    """
    Compute: score_i = function(target, feature_i) for all features.
    Compute confidence interval (CI) for n_features features.
    Compute P-value and FDR (BH) for all features.
    :param target: Series; (n_samples); must have name and indices, matching features's column index
    :param features: DataFrame; (n_features, n_samples); must have row and column indices
    :param function: function; scoring function
    :param dropna: str; 'any' or 'all'
    :param target_ascending: bool; target is ascending or not
    :param n_jobs: int; number of jobs to parallelize
    :param min_n_per_job: int; minimum number of n per job
    :param features_ascending: bool; True if features scores increase from top to bottom, and False otherwise
    :param n_features: int or float; number of features to compute confidence interval and plot;
                        number threshold if >= 1, percentile threshold if < 1, and don't compute if None
    :param n_samplings: int; number of bootstrap samplings to build distribution to get CI; must be > 2 to compute CI
    :param confidence: float; fraction compute confidence interval
    :param n_permutations: int; number of permutations for permutation test to compute P-val and FDR
    :param random_seed: int;
    :param filepath: str;
    :return: Series, DataFrame, DataFrame; (n_features, 8 ('score', '<confidence> moe',
                                            'p-value (forward)', 'p-value (reverse)', 'p-value',
                                            'fdr (forward)', 'fdr (reverse)', 'fdr'))
    """

    # TODO: make empty DataFrame to absorb the results instead of concatenation

    # Make sure target is a Series and features a DataFrame
    # Keep samples found in both target and features
    # Drop features with less than 2 unique values
    target, features = _preprocess_target_and_features(target, features, dropna=dropna,
                                                       target_ascending=target_ascending)

    results = DataFrame(index=features.index, columns=['score', '{} moe'.format(confidence),
                                                       'p-value (forward)', 'p-value (reverse)', 'p-value',
                                                       'fdr (forward)', 'fdr (reverse)', 'fdr'])

    #
    # Compute: score_i = function(target, feature_i)
    #
    print_log('Scoring (n_jobs={}) ...'.format(n_jobs))

    # Split features for parallel computing
    if features.shape[0] < n_jobs * min_n_per_job:
        n_jobs = 1
    split_features = split_dataframe(features, n_jobs)

    # Score
    scores = concat(parallelize(_score, [(target, f, function) for f in split_features], n_jobs), verify_integrity=True)

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
        seed(random_seed)
        for c_i in sampled_scores:
            # Random sample
            ramdom_samples = choice(features.columns.tolist(), int(ceil(0.632 * features.shape[1]))).tolist()
            sampled_target = target.ix[ramdom_samples]
            sampled_features = features.ix[indices_to_bootstrap, ramdom_samples]
            rs = get_state()

            # Score
            sampled_scores.ix[:, c_i] = sampled_features.apply(lambda f: function(sampled_target, f), axis=1)

            set_state(rs)

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
        print_log(
            'Computing P-value & FDR by scoring against {} permuted targets (n_jobs={}) ...'.format(n_permutations,
                                                                                                    n_jobs))

        # Permute and score
        permutation_scores = concat(parallelize(_permute_and_score,
                                                [(target, f, function, n_permutations, random_seed) for f in
                                                 split_features], n_jobs), verify_integrity=True)

        print_log('\tComputing P-value and FDR ...')
        # All scores
        all_permutation_scores = permutation_scores.values.flatten()
        for i, (r_i, r) in enumerate(results.iterrows()):
            # This feature's score
            s = r.ix['score']

            # Compute forward P-value
            p_value_forward = (all_permutation_scores >= s).sum() / len(all_permutation_scores)
            if not p_value_forward:
                p_value_forward = float(1 / len(all_permutation_scores))
            results.ix[r_i, 'p-value (forward)'] = p_value_forward

            # Compute reverse P-value
            p_value_reverse = (all_permutation_scores <= s).sum() / len(all_permutation_scores)
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


def _preprocess_target_and_features(target, features, dropna='all', target_ascending=False, min_n_unique_values=2):
    """
    Make sure target is a Series and features a DataFrame.
    Keep samples found in both target and features.
    Drop features with less than 2 unique values.
    :param target: Series or iterable;
    :param features: DataFrame or Series;
    :param dropna: 'any' or 'all'
    :param target_ascending: bool;
    :param min_n_unique_values: int;
    :return: Series and DataFrame;
    """

    if isinstance(features, Series):  # Convert Series-features into DataFrame-features with 1 row
        features = DataFrame(features).T

    features.dropna(axis=1, how=dropna, inplace=True)

    if not isinstance(target, Series):  # Convert target into a Series
        if isinstance(target, DataFrame) and target.shape[0] == 1:
            target = target.iloc[0, :]
        else:
            target = Series(target, index=features.columns)

    # Keep only columns shared by target and features
    shared = target.index & features.columns
    if any(shared):
        print_log('Target ({} cols) and features ({} cols) have {} shared columns.'.format(target.size,
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
         int; n_permutations,
         array; random_state)
    :return: DataFrame; (n_features, n_permutations)
    """

    if len(args) != 5:
        raise ValueError('args is not length of 5 (target, features, function, n_perms, and random_state).')
    else:
        t, f, func, n_perms, random_seed = args

    scores = DataFrame(index=f.index, columns=range(n_perms))

    # Target array to be permuted during each permutation
    permuted_t = array(t)

    seed(random_seed)
    for p in range(n_perms):
        print_log('\tScoring against permuted target ({}/{}) ...'.format(p, n_perms), print_process=True)

        shuffle(permuted_t)
        rs = get_state()

        scores.iloc[:, p] = f.apply(lambda r: func(permuted_t, r), axis=1)

        set_state(rs)

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
    save_plot(filepath)


def _prepare_data_for_plotting(dataframe, data_type, max_std=3):
    if data_type == 'continuous':
        return normalize_2d_or_1d(dataframe, method='-0-', axis=1), -max_std, max_std, CMAP_ASSOCIATION
    elif data_type == 'categorical':
        return dataframe.copy(), 0, len(unique(dataframe)), CMAP_CATEGORICAL
    elif data_type == 'binary':
        return dataframe.copy(), 0, 1, CMAP_BINARY
    else:
        raise ValueError('Target data type must be one of {continuous, categorical, binary}.')


# ======================================================================================================================
# Comparison panel
# ======================================================================================================================
def make_comparison_panel(matrix1, matrix2, matrix1_label='Matrix 1', matrix2_label='Matrix 2',
                          function=information_coefficient, axis=0, is_distance=False, annotate=True, title=None,
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
    :param annotate: bool; show values in the matrix or not
    :param title: str; plot title
    :param filepath_prefix: str; filepath_prefix.txt and filepath_prefix.pdf will be saved
    :return: DataFrame; association or distance matrix
    """

    # Compute association or distance matrix, which is returned at the end
    comparison_matrix = compute_similarity_matrix(matrix2, matrix1, function, axis=axis, is_distance=is_distance)

    if filepath_prefix:  # Save
        comparison_matrix.to_csv(filepath_prefix + '.txt', sep='\t')

    # Plot cluster map of the compared matrix
    if filepath_prefix:
        filepath = filepath_prefix + '.pdf'
    else:
        filepath = None
    plot_clustermap(comparison_matrix, title=title, xlabel=matrix1_label, ylabel=matrix2_label, annotate=annotate,
                    filepath=filepath)

    return comparison_matrix


# ======================================================================================================================
# Modalities
# ======================================================================================================================
def differential_gene_expression(phenotypes, gene_expression, output_filename, max_number_of_genes_to_show=20,
                                 number_of_permutations=10, title=None, random_seed=RANDOM_SEED):
    """
    Sort genes according to their association with a binary phenotype or class vector.
    :param phenotypes: Series; input binary phenotype/class distinction
    :param gene_expression: Dataframe; data matrix with input gene expression profiles
    :param output_filename: str; output files will have this name plus extensions .txt and .pdf
    :param max_number_of_genes_to_show: int; maximum number of genes to show in the heatmap
    :param number_of_permutations: int; number of random permutations to estimate statistical significance (p-values and FDRs)
    :param title: str;
    :param random_seed: int; random number generator seed (can be set to a user supplied integer for reproducibility)
    :return: Dataframe; table of genes ranked by Information Coeff vs. phenotype
    """
    gene_scores = make_association_panel(target=phenotypes,
                                         features=gene_expression,
                                         n_jobs=1,
                                         max_n_features=max_number_of_genes_to_show,
                                         n_permutations=number_of_permutations,
                                         target_type='binary',
                                         title=title,
                                         filepath_prefix=output_filename,
                                         random_seed=random_seed)
    return gene_scores
