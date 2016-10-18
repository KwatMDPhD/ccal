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

from pip import get_installed_distributions, main
from os import mkdir, environ
from os.path import abspath, split, isdir, isfile, islink
from subprocess import Popen, PIPE
from csv import reader, writer, excel, excel_tab
from datetime import datetime
from operator import add, sub
from multiprocessing import Pool

from numpy import finfo, array, asarray, empty, zeros, ones, unique, sign, sum, sqrt, exp, log, dot, isnan, sort, \
    argmax, average
from numpy.linalg import pinv
from numpy.random import random_sample, random_integers, shuffle
from pandas import Series, DataFrame, read_csv
from scipy.stats import pearsonr
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, fcluster, cophenet
from scipy.optimize import curve_fit, nnls
from sklearn.manifold import MDS
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import NMF
from sklearn.cross_validation import KFold
from networkx import Graph, DiGraph
import rpy2.robjects as ro
from rpy2.robjects.numpy2ri import numpy2ri
from rpy2.robjects.packages import importr
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import ListedColormap, LinearSegmentedColormap, Normalize
from matplotlib.cm import Paired, bwr
from matplotlib.colorbar import make_axes, ColorbarBase
from matplotlib.backends.backend_pdf import PdfPages
from seaborn import light_palette, heatmap, clustermap, pointplot

from . import VERBOSE, SEED

# ======================================================================================================================
# Parameter
# ======================================================================================================================
EPS = finfo(float).eps

CODON_TO_AMINO_ACID = {'GUC': 'V', 'ACC': 'T', 'GUA': 'V', 'GUG': 'V', 'GUU': 'V', 'AAC': 'N', 'CCU': 'P', 'UGG': 'W',
                       'AGC': 'S', 'auc': 'I', 'CAU': 'H', 'AAU': 'N', 'AGU': 'S', 'ACU': 'T', 'CAC': 'H', 'ACG': 'T',
                       'CCG': 'P', 'CCA': 'P', 'ACA': 'T', 'CCC': 'P', 'GGU': 'G', 'UCU': 'S', 'GCG': 'A', 'UGC': 'C',
                       'CAG': 'Q', 'GAU': 'D', 'UAU': 'Y', 'CGG': 'R', 'UCG': 'S', 'AGG': 'R', 'GGG': 'G', 'UCC': 'S',
                       'UCA': 'S', 'GAG': 'E', 'GGA': 'G', 'UAC': 'Y', 'GAC': 'D', 'GAA': 'E', 'AUA': 'I', 'GCA': 'A',
                       'CUU': 'L', 'GGC': 'G', 'AUG': 'M', 'CUG': 'L', 'CUC': 'L', 'AGA': 'R', 'CUA': 'L', 'GCC': 'A',
                       'AAA': 'K', 'AAG': 'K', 'CAA': 'Q', 'UUU': 'F', 'CGU': 'R', 'CGA': 'R', 'GCU': 'A', 'UGU': 'C',
                       'AUU': 'I', 'UUG': 'L', 'UUA': 'L', 'CGC': 'R', 'UUC': 'F'}

ro.conversion.py2ri = numpy2ri

mass = importr('MASS')
bcv = mass.bcv
kde2d = mass.kde2d


# ======================================================================================================================
# System
# ======================================================================================================================
def install_libraries(libraries_needed):
    """
    Check if libraries_needed are installed; if not, install using pip.
    :param libraries_needed: iterable; library names
    :return: None
    """

    print_log('Checking library dependencies ...')

    # Get currently installed libraries
    libraries_installed = [lib.key for lib in get_installed_distributions()]

    # If any of the libraries_needed is not in the currently installed libraries, then install it using pip
    for lib in libraries_needed:
        if lib not in libraries_installed:
            print_log('{} not found; installing it using pip ...'.format(lib))
            main(['install', lib])


def source_environment(filepath):
    """
    Update environment using source_environment.
    :param filepath:
    :return:
    """

    print_log('Sourcing {} ...'.format(filepath))

    for line in Popen('./{}; env'.format(filepath), stdout=PIPE, universal_newlines=True, shell=True).stdout:
        key, _, value = line.partition('=')
        key, value = key.strip(), value.strip()
        environ[key] = value
        print_log('\t{} = {}'.format(key, value))


# ======================================================================================================================
# Parallel computing
# ======================================================================================================================
def parallelize(function, list_of_args, n_jobs=1):
    """
    Apply function on args with parallel computing using n_jobs jobs.
    :param function: function;
    :param list_of_args: list-like; function's args
    :param n_jobs: int;
    :return: list; list of outputs returned by all jobs
    """

    # Parallelize
    with Pool(n_jobs) as p:
        return p.map(function, list_of_args)


# ======================================================================================================================
# File
# ======================================================================================================================
def establish_filepath(filepath):
    """
    If the path up to the deepest directory in filepath doesn't exist, make the path up to the deepest directory.
    :param filepath: str;
    :return: None
    """

    prefix, suffix = split(filepath)
    prefix = abspath(prefix)

    # Get missing directories
    missing_directories = []
    while not (isdir(prefix) or isfile(prefix) or islink(prefix)):
        missing_directories.append(prefix)
        prefix, suffix = split(prefix)

    # Make missing directories
    for d in reversed(missing_directories):
        mkdir(d)
        print_log('Created directory {}.'.format(d))


def split_file_extention(filepath):
    """
    Return the base filepath and extension as a tuple.
    :param filepath: str;
    :return: str and str;
    """

    split_filepath = filepath.split('.')
    base = ''.join(split_filepath[:-1])
    extension = split_filepath[-1]
    return base, extension


def count_n_lines_in_file(filepath):
    """
    Count the number of lines in filepath.
    :param filepath: str;
    :return: int;
    """

    with open(filepath) as f:
        i = -1
        for i, x in enumerate(f):
            pass
    return i + 1


def convert_csv_to_tsv(filepath):
    """
    Convert .csv file to .tsv file.
    :param filepath: str;
    :return: None
    """

    with open(filepath, 'rU') as infile:
        r = reader(infile, dialect=excel)
    with open(filepath.strip('.csv') + '.tsv', 'w') as outfile:
        w = writer(outfile, dialect=excel_tab)
    for line in r:
        w.writerow(line)


def read_gct(filepath, fill_na=None, drop_description=True, row_name=None, column_name=None):
    """
    Read a .gct (filepath) and convert it into a pandas DataFrame.
    :param filepath: str;
    :param fill_na: *; value to replace NaN in the DataFrame
    :param drop_description: bool; drop the Description column (column 2 in the .gct) or not
    :param row_name: str;
    :param column_name: str;
    :return: pandas DataFrame; [n_samples, n_features (or n_features + 1 if not dropping the Description column)]
    """

    # Read .gct
    df = read_csv(filepath, skiprows=2, sep='\t')

    # Fix missing values
    if fill_na:
        df.fillna(fill_na, inplace=True)

    # Get 'Name' and 'Description' columns
    c1, c2 = df.columns[:2]

    # Check if the 1st column is 'Name'; if so set it as the index
    if c1 != 'Name':
        if c1.strip() != 'Name':
            raise ValueError('Column 1 != \'Name\'.')
        else:
            raise ValueError('Column 1 has more than 1 extra space around \'Name\'. Please strip it.')
    df.set_index('Name', inplace=True)

    # Check if the 2nd column is 'Description'; is so drop it as necessary
    if c2 != 'Description':
        if c2.strip() != 'Description':
            raise ValueError('Column 2 != \'Description\'')
        else:
            raise ValueError('Column 2 has more than 1 extra space around \'Description\'. Please strip it.')
    if drop_description:
        df.drop('Description', axis=1, inplace=True)

    # Set row and column name
    df.index.name = row_name
    df.columns.name = column_name

    return df


def write_gct(pandas_object, filepath, descriptions=None):
    """
    Write a pandas_object to a filepath as a .gct file.
    :param pandas_object: pandas DataFrame or Serires; (n_samples, m_features)
    :param filepath: str;
    :param descriptions: iterable; (n_rows of pandas_object); description column for the .gct
    :return: None
    """

    # Copy
    obj = pandas_object.copy()

    # Work with only DataFrame
    if isinstance(obj, Series):
        obj = DataFrame(obj).T

    # Add description column if missing
    if obj.columns[0] != 'Description':
        if descriptions:
            obj.insert(0, 'Description', descriptions)
        else:
            obj.insert(0, 'Description', obj.index)

    # Set row and column name
    obj.index.name = 'Name'
    obj.columns.name = None

    # Save as .gct
    if not filepath.endswith('.gct'):
        filepath += '.gct'
    with open(filepath, 'w') as f:
        f.writelines('#1.2\n{}\t{}\n'.format(obj.shape[0], obj.shape[1] - 1))
        obj.to_csv(f, sep='\t')


def read_gmt(filepath, drop_description=True):
    """
    Read filepath, a .gmt file.
    :param filepath:
    :param drop_description: bool; drop the Description column (column 2 in the .gct) or not
    :return: pandas DataFrame; (n_gene_sets, n_genes_in_the_largest_gene_set)
    """

    # Read .gct
    df = read_csv(filepath, sep='\t')

    # Get 'Name' and 'Description' columns
    c1, c2 = df.columns[:2]

    # Check if the 1st column is 'Name'; if so set it as the index
    if c1 != 'Name':
        if c1.strip() != 'Name':
            raise ValueError('Column 1 != \'Name\'.')
        else:
            raise ValueError('Column 1 has more than 1 extra space around \'Name\'. Please strip it.')
    df.set_index('Name', inplace=True)

    # Check if the 2nd column is 'Description'; is so drop it as necessary
    if c2 != 'Description':
        if c2.strip() != 'Description':
            raise ValueError('Column 2 != \'Description\'')
        else:
            raise ValueError('Column 2 has more than 1 extra space around \'Description\'. Please strip it.')
    if drop_description:
        df.drop('Description', axis=1, inplace=True)

    # Set row name (column name is None when read)
    df.index.name = 'Gene Set'

    return df


def write_gmt(pandas_object, filepath, descriptions=None):
    """
    Write a pandas_object to a filepath as a .gmt file.
    :param pandas_object: pandas DataFrame or Serires; (n_samples, m_features)
    :param filepath: str;
    :param descriptions: iterable; (n_rows of pandas_object); description column for the .gmt
    :return: None
    """

    obj = pandas_object.copy()

    # Add description column if missing
    if obj.columns[0] != 'Description':
        if descriptions:
            obj.insert(0, 'Description', descriptions)
        else:
            obj.insert(0, 'Description', obj.index)

    # Set row and column name
    obj.index.name = 'Name'
    obj.columns.name = None

    # Save as .gmt
    if not filepath.endswith('.gmt'):
        filepath += '.gmt'
    obj.to_csv(filepath, sep='\t')


def read_dictionary(filepath, sep='\t', switch=False):
    """
    Make a dictionary from mapping_file: key<sep>value.
    By default, 1st column is the key and the 2nd value, and use tab delimeter.
    :param filepath:
    :param sep:
    :param switch:
    :return:
    """

    # Set column for key and value
    if switch:
        column_names = ['value', 'key']
    else:
        column_names = ['key', 'value']

    # Load mapping info; drop rows with NaN and duplicates
    mapping = read_csv(filepath, sep=sep, names=column_names).dropna().drop_duplicates()

    # Sort by key
    mapping.sort_values('key', inplace=True)

    # Loop to make dictionary
    dictionary = dict()
    prev = None
    temp = set()

    for i, s in mapping.iterrows():

        # Key and value
        k = s.ix['key']
        v = s.ix['value']

        # Add to dictionary when seeing new key
        if k != prev and prev:
            dictionary[prev] = temp
            temp = set()

        # Keep accumulating value for consecutive key
        temp.add(v)
        prev = k

    # Last addition to the dictionary for the edge case
    dictionary[prev] = temp

    return dictionary


def write_dictionary(dictionary, filepath, key_name, value_name, sep='\t'):
    """
    Write a dictionary as a 2-column-tab-separated file.
    :param dictionary: dict;
    :param filepath: str;
    :param key_name; str;
    :param value_name; str;
    :param sep: str; separator
    :return: None
    """

    with open(filepath, 'w') as f:
        f.write('{}\t{}\n'.format(key_name, value_name))
        for k, v in sorted(dictionary.items()):
            f.write('{}{}{}\n'.format(k, sep, v))


def write_colormap(cmap, filepath):
    """

    :param cmap:
    :param filepath:
    :return:
    """

    with open(filepath, 'w') as f:
        for i in range(cmap.N):
            f.writelines('\t'.join([str(c) for c in cmap(i)]) + '\n')


def read_colormap(filepath):
    """

    :param filepath:
    :return:
    """

    rgbas = []
    with open(filepath, 'r') as f:
        for l in f:
            rgbas.append([float(v) for v in l.split()])
    return ListedColormap(rgbas)


# ======================================================================================================================
# Log
# ======================================================================================================================
# TODO: use logging (https://docs.python.org/3.5/howto/logging.html)
def print_log(string):
    """
    Print string together with logging information.
    :param string: str; message to printed
    :return: None
    """

    if VERBOSE:
        print('<{}> {}'.format(timestamp(time_only=True), string))


def timestamp(time_only=False):
    """
    Get the current time.
    :param time_only: bool; exclude year, month, and date or not
    :return: str; the current time
    """

    if time_only:
        formatter = '%H%M%S'
    else:
        formatter = '%Y%m%d-%H%M%S'
    return datetime.now().strftime(formatter)


# ======================================================================================================================
# String
# ======================================================================================================================
def title_string(string):
    """
    Title a string.
    :param string: str;
    :return: str;
    """

    # Remember indices of original uppercase letters
    uppers = []
    start = end = None
    is_upper = False
    for i, c in enumerate(string):
        if c.isupper():
            # print('{} is UPPER.'.format(c))
            if is_upper:
                end += 1
            else:
                is_upper = True
                start = i
                end = start + 1
                # print('Start is {}.'.format(i))
        else:
            if is_upper:
                is_upper = False
                uppers.append((start, end))
                start = None
                end = start
    else:
        if start:
            uppers.append((start, end))

    # Title
    string = string.title().replace('_', ' ')

    # Upper all original uppercase letters
    for start, end in uppers:
        string = string[:start] + string[start: end].upper() + string[end:]

    # Lower some words
    for lowercase in ['a', 'an', 'the', 'and', 'but', 'or', 'for', 'nor', 'on', 'at', 'to', 'from', 'of', 'vs', 'vs']:
        string = string.replace(' ' + lowercase.title() + ' ', ' ' + lowercase + ' ')

    return string


def untitle_string(string):
    """
    Untitle a string.
    :param string: str;
    :return: str;
    """

    string = str(string)
    return string.lower().replace(' ', '_').replace('-', '_')


def clean_string(string, illegal_chars=(' ', '\t', ',', ';', '|'), replacement_char='_'):
    """
    Return a copy of string that has all non-allowed characters replaced by a new character (default: underscore).
    :param string:
    :param illegal_chars:
    :param replacement_char:
    :return:
    """

    new_string = str(string)
    for illegal_char in illegal_chars:
        new_string = new_string.replace(illegal_char, replacement_char)
    return new_string


# TODO: consider deleting
def cast_string_to_int_float_bool_or_str(string):
    """
    Convert string into the following data types (return the first successful): int, float, bool, or str.
    :param string:
    :return:
    """

    value = string.strip()

    for var_type in [int, float]:
        try:
            converted_var = var_type(value)
            return converted_var
        except ValueError:
            pass

    if value == 'True':
        return True
    elif value == 'False':
        return False

    return str(value)


# TODO: consider deleting
def indent_string(string, n_tabs=1):
    """
    Indent block of text by adding a n_tabs number of tabs (default 1) to the beginning of each line.
    :param string:
    :param n_tabs:
    :return:
    """

    return '\n'.join(['\t' * n_tabs + line for line in string.split('\n')])


# ======================================================================================================================
# Equation
# ======================================================================================================================
def exponential_function(x, a, k, c):
    """
    Apply exponential function on x.
    :param x: array-like; independent variables
    :param a: number; parameter a
    :param k: number; parameter k
    :param c: number; parameter c
    :return: numpy array; (n_independent_variables)
    """

    return a * exp(k * x) + c


# ======================================================================================================================
# Dictionary
# ======================================================================================================================
def dict_merge_with_function(function, dict_1, dict_2):
    """
    Apply function to values keyed by the same key in dict_1 and dict_2.
    :param function:
    :param dict_1:
    :param dict_2:
    :return:
    """

    new_dict = {}
    all_keys = set(dict_1.keys()).union(dict_2.keys())
    for k in all_keys:
        if k in dict_1 and k in dict_2:
            new_dict[k] = function(dict_1[k], dict_2[k])
        elif k in dict_1:
            new_dict[k] = dict_1[k]
        else:
            new_dict[k] = dict_2[k]
    return new_dict


def dict_add(dict_1, dict_2):
    """
    Add dict_1 and dict_2.
    :param dict_1:
    :param dict_2:
    :return:
    """

    return dict_merge_with_function(add, dict_1, dict_2)


def dict_subtract(dict_1, dict_2):
    """
    Subtract dict_2 from dict_1.
    :param dict_1:
    :param dict_2:
    :return:
    """

    return dict_merge_with_function(sub, dict_1, dict_2)


# ======================================================================================================================
# Array-like
# ======================================================================================================================
def compute_sliding_mean(vector, window_size=1):
    """
    Return a vector of means for each window_size in vector.
    :param vector:
    :param window_size:
    :return:
    """

    m = zeros(len(vector))
    for i in range(len(vector)):
        m[i] = sum(vector[max(0, i - window_size):min(len(vector), i + window_size + 1)]) / float(window_size * 2 + 1)
    return m


def compute_geometric_mean(vector):
    """
    Return the geometric mean (the n-th root of the product of n terms) of an vector.
    :param vector:
    :return:
    """
    product = vector[0]
    for n in vector[1:]:
        product *= n
    return product ** (1 / len(vector))


def quantize(vector, precision_factor):
    """
    Return a copy of vector that is scaled by precision_factor and then rounded to the nearest integer.
    To re-scale, simply divide by precision_factor.
    Note that because of rounding, an open interval from (x, y) will give rise to up to
    (x - y) * precision_factor + 1 bins.
    :param vector:
    :param precision_factor:
    :return:
    """

    return (asarray(vector) * precision_factor).round(0)


def group_iterable(iterable, n=2, partial_final_item=False):
    """
    Given iterable, return sub-lists made of n items.
    :param iterable:
    :param n:
    :param partial_final_item:
    :return:
    """

    accumulator = []
    for item in iterable:
        accumulator.append(item)
        if len(accumulator) == n:
            yield accumulator
            accumulator = []
    if len(accumulator) != 0 and (len(accumulator) == n or partial_final_item):
        yield accumulator


def get_unique_in_order(iterable):
    """
    Get unique elements in order or appearance in iterable.
    :param iterable: iterable;
    :return: list;
    """

    unique_in_order = []
    for x in iterable:
        if x not in unique_in_order:
            unique_in_order.append(x)
    return unique_in_order


def explode(series):
    """
    Make a label-x-sample binary matrix from a Series.
    :param series: pandas Series;
    :return: pandas DataFrame; (n_labels, n_samples)
    """

    # Make an empty DataFrame (n_unique_labels, n_samples)
    label_x_sample = DataFrame(index=sorted(set(series)), columns=series.index)

    # Binarize each unique label
    for i in label_x_sample.index:
        label_x_sample.ix[i, :] = (series == i).astype(int)

    return label_x_sample


# ======================================================================================================================
# Matrix-like
# ======================================================================================================================
def flatten_nested_iterables(nested_iterable, list_type=(list, tuple)):
    """
    Flatten an arbitrarily-deep nested_list.
    :param nested_iterable: a list to flatten_nested_iterables
    :param list_type: valid variable types to flatten_nested_iterables
    :return: list; a flattened list
    """

    type_ = type(nested_iterable)
    nested_iterable = list(nested_iterable)
    i = 0
    while i < len(nested_iterable):
        while isinstance(nested_iterable[i], list_type):
            if not nested_iterable[i]:
                nested_iterable.pop(i)
                i -= 1
                break
            else:
                nested_iterable[i:i + 1] = nested_iterable[i]
        i += 1
    return type_(nested_iterable)


def drop_nan_columns(arrays):
    """
    Keep only not-NaN column positions in all arrays.
    :param arrays: iterable of numpy arrays; must have the same length
    :return: list of numpy arrays; none of the arrays contains NaN
    """

    # Keep all column indices
    not_nan_filter = ones(len(arrays[0]), dtype=bool)

    # Keep column indices without missing value in all arrays
    for a in arrays:
        not_nan_filter &= ~isnan(a)

    return [a[not_nan_filter] for a in arrays]


def get_consensus(sample_x_clustering):
    """
    Count number of co-clusterings.
    :param sample_x_clustering: pandas DataFrame; (n_samples, n_clusterings)
    :return: pandas DataFrame; (n_samples, n_samples)
    """
    sample_x_clustering_array = asarray(sample_x_clustering)

    n_samples, n_clusterings = sample_x_clustering_array.shape

    # Make sample x sample matrix
    coclusterings = zeros((n_samples, n_samples))

    # Count the number of co-clusterings
    for i in range(n_samples):
        for j in range(n_samples):
            for c_i in range(n_clusterings):
                v1 = sample_x_clustering_array[i, c_i]
                v2 = sample_x_clustering_array[j, c_i]
                if v1 and v2 and (v1 == v2):
                    coclusterings[i, j] += 1

    # Normalize by the number of clusterings and return
    coclusterings /= n_clusterings
    return DataFrame(coclusterings, index=sample_x_clustering.index, columns=sample_x_clustering.index)


def mds(dataframe, distance_function=None, mds_seed=SEED, n_init=1000, max_iter=1000, standardize=True):
    """
    Multidimentional scale rows of pandas_object from <n_cols>D into 2D.
    :param dataframe: pandas DataFrame; (n_points, n_dimentions)
    :param distance_function: function; capable of computing the distance between 2 vectors
    :param mds_seed: int; random seed for setting the coordinates of the multidimensional scaling
    :param n_init: int;
    :param max_iter: int;
    :param standardize: bool;
    :return: pandas DataFrame; (n_points, 2 ('x', 'y'))
    """

    if distance_function:  # Use precomputed distances
        mds_obj = MDS(dissimilarity='precomputed', random_state=mds_seed, n_init=n_init, max_iter=max_iter)
        coordinates = mds_obj.fit_transform(compare_matrices(dataframe, dataframe, distance_function,
                                                             is_distance=True, axis=1))
    else:  # Use Euclidean distances
        mds_obj = MDS(random_state=mds_seed, n_init=n_init, max_iter=max_iter)
        coordinates = mds_obj.fit_transform(dataframe)

    # Convert to DataFrame
    coordinates = DataFrame(coordinates, index=dataframe.index, columns=['x', 'y'])

    if standardize:  # Rescale coordinates between 0 and 1
        coordinates = normalize_pandas_object(coordinates, method='0-1', axis=0)

    return coordinates


# ======================================================================================================================
# Normalization
# ======================================================================================================================
# TODO: make sure the normalization when size == 0 or range == 0 is correct
def normalize_pandas_object(pandas_object, method, axis=None, n_ranks=10000):
    """
    Normalize a pandas object.
    :param pandas_object: pandas DataFrame or Series;
    :param method: str; normalization type; {'-0-', '0-1', 'rank'}
    :param n_ranks: number; normalization factor for rank normalization: rank / size * n_ranks
    :param axis: int; None for global, 0 for by-column, and 1 for by-row normalization
    :return: pandas DataFrame or Series; normalized DataFrame or Series
    """

    if isinstance(pandas_object, Series):  # Series
        return normalize_series(pandas_object, method=method, n_ranks=n_ranks)

    elif isinstance(pandas_object, DataFrame):  # DataFrame
        if axis == 0 or axis == 1:  # Normalize by axis (Series)
            return pandas_object.apply(normalize_series, **{'method': method, 'n_ranks': n_ranks}, axis=axis)

        else:  # Normalize globally
            if method == '-0-':
                obj_mean = pandas_object.values.mean()
                obj_std = pandas_object.values.std()
                if obj_std == 0:
                    # print_log('Not \'-0-\' normalizing (standard deviation is 0), but \'/ size\' normalizing.')
                    return pandas_object / pandas_object.size
                else:
                    return (pandas_object - obj_mean) / obj_std

            elif method == '0-1':
                obj_min = pandas_object.values.min()
                obj_max = pandas_object.values.max()
                if obj_max - obj_min == 0:
                    # print_log('Not \'0-1\' normalizing (data range is 0), but \'/ size\' normalizing.')
                    return pandas_object / pandas_object.size
                else:
                    return (pandas_object - obj_min) / (obj_max - obj_min)

            elif method == 'rank':
                # TODO: implement global rank normalization
                raise ValueError('Normalizing combination of \'rank\' & axis=\'all\' has not been implemented yet.')


def normalize_series(series, method='-0-', n_ranks=10000):
    """
    Normalize a pandas series.
    :param series: pandas Series;
    :param method: str; normalization type; {'-0-', '0-1', 'rank'}
    :param n_ranks: number; normalization factor for rank normalization: rank / size * n_ranks
    :return: pandas Series; normalized Series
    """

    if method == '-0-':
        mean = series.mean()
        std = series.std()
        if std == 0:
            # print_log('Not \'-0-\' normalizing (standard deviation is 0), but \'/ size\' normalizing.')
            return series / series.size
        else:
            return (series - mean) / std
    elif method == '0-1':
        series_min = series.min()
        series_max = series.max()
        if series_max - series_min == 0:
            # print_log('Not \'0-1\' normalizing (data_range is 0), but \'/ size\' normalizing.')
            return series / series.size
        else:
            return (series - series_min) / (series_max - series_min)
    elif method == 'rank':
        # NaNs are raked lowest in the ascending ranking
        return series.rank(na_option='top') / series.size * n_ranks


# ======================================================================================================================
# Math
# ======================================================================================================================
def cross_validate(model, data, target, n_partitions):
    """
    Cross-validata.
    :param model:
    :param data:
    :param target:
    :param n_partitions:
    :return:
    """

    # Initialize indexes for cross validation folds
    folds = KFold(len(data), n_partitions, shuffle=True)

    # List to keep cross validation scores
    scores = []

    # For each fold
    for k, (train_index, test_index) in enumerate(folds):
        # Partition training and testing data sets
        x_train, x_test = data.iloc[train_index], data.iloc[test_index]
        y_train, y_test = target.iloc[train_index], target.iloc[test_index]

        # Learn
        model.fit(x_train, y_train.iloc[:, 0])

        # Score the learned fit
        score = pearsonr(model.predict(x_test), y_test.iloc[:, 0])
        scores.append(score)

    return average([s[0] for s in scores]), average([s[1] for s in scores])


# ======================================================================================================================
# Information theory
# ======================================================================================================================
def information_coefficient(x, y, n_grids=25, jitter=1E-10):
    """
    Compute the information coefficient between x and y, which can be composed of either continuous,
    categorical, or binary values.
    :param x: numpy array;
    :param y: numpy array;
    :param n_grids: int; number of grid lines in a dimention when estimating bandwidths
    :param jitter: number;
    :return: float;
    """

    # Can't work with missing any value
    # not_nan_filter = ~isnan(x)
    # not_nan_filter &= ~isnan(y)
    # x = x[not_nan_filter]
    # y = y[not_nan_filter]
    x, y = drop_nan_columns([x, y])

    # Need at least 3 values to compute bandwidth
    if len(x) < 3 or len(y) < 3:
        return 0

    x = asarray(x, dtype=float)
    y = asarray(y, dtype=float)

    # Add jitter
    x += random_sample(x.size) * jitter
    y += random_sample(y.size) * jitter

    # Compute bandwidths
    cor, p = pearsonr(x, y)
    bandwidth_x = asarray(bcv(x)[0]) * (1 + (-0.75) * abs(cor))
    bandwidth_y = asarray(bcv(y)[0]) * (1 + (-0.75) * abs(cor))

    # Compute P(x, y), P(x), P(y)
    fxy = asarray(kde2d(x, y, asarray([bandwidth_x, bandwidth_y]), n=asarray([n_grids]))[2]) + EPS
    dx = (x.max() - x.min()) / (n_grids - 1)
    dy = (y.max() - y.min()) / (n_grids - 1)
    pxy = fxy / (fxy.sum() * dx * dy)
    px = pxy.sum(axis=1) * dy
    py = pxy.sum(axis=0) * dx

    # Compute mutual information;
    mi = sum(pxy * log(pxy / (asarray([px] * n_grids).T * asarray([py] * n_grids)))) * dx * dy

    # # Get H(x, y), H(x), and H(y)
    # hxy = - sum(pxy * log(pxy)) * dx * dy
    # hx = -sum(px * log(px)) * dx
    # hy = -sum(py * log(py)) * dy
    # mi = hx + hy - hxy

    # Compute information coefficient
    ic = sign(cor) * sqrt(1 - exp(- 2 * mi))

    # TODO: debug when MI < 0 and |MI|  ~ 0 resulting in IC = nan
    if isnan(ic):
        ic = 0

    return ic


# ======================================================================================================================
# Association
# ======================================================================================================================
def compute_score_and_pvalue(x, y, function=information_coefficient, n_permutations=100):
    """
    Compute function(x, y) and p-value using permutation test.
    :param x: array-like;
    :param y: array-like;
    :param function: function;
    :param n_permutations: int; number of permutations for the p-value permutation test
    :return: float and float; score and p-value
    """

    # Compute score
    score = function(x, y)

    # Compute scores against permuted target
    permutation_scores = empty(n_permutations)
    shuffled_target = array(y)
    for p in range(n_permutations):
        shuffle(shuffled_target)
        permutation_scores[p] = function(x, shuffled_target)

    # Compute p-value
    p_val = sum(permutation_scores > score) / n_permutations
    return score, p_val


def compare_matrices(matrix1, matrix2, function, axis=0, is_distance=False):
    """
    Make association or distance matrix of matrix1 and matrix2 by row (axis=1) or by column (axis=0).
    :param matrix1: pandas DataFrame;
    :param matrix2: pandas DataFrame;
    :param function: function; function used to compute association or dissociation
    :param axis: int; 0 for row-wise and 1 column-wise comparison
    :param is_distance: bool; True for distance and False for association
    :return: pandas DataFrame; (n, n); association or distance matrix
    """

    # Rotate matrices to make the comparison by row
    if axis == 1:
        matrix1 = matrix1.copy()
        matrix2 = matrix2.copy()
    else:
        matrix1 = matrix1.T
        matrix2 = matrix2.T

    # Work with array
    m1 = asarray(matrix1)
    m2 = asarray(matrix2)

    # Number of comparables
    n_1 = m1.shape[0]
    n_2 = m2.shape[0]

    # Compare
    compared_matrix = empty((n_1, n_2))
    for i_1 in range(n_1):
        print_log('Computing associations (axis={}) between matrices ({}/{}) ...'.format(axis, i_1, n_1))
        for i_2 in range(n_2):
            compared_matrix[i_1, i_2] = function(m1[i_1, :], m2[i_2, :])

    if is_distance:  # Convert association to distance
        print_log('Converting association to distance (1 - association) ...')
        compared_matrix = 1 - compared_matrix

    return DataFrame(compared_matrix, index=matrix1.index, columns=matrix2.index)


def fit_matrix(matrix, function_to_fit, axis=0, sort_matrix=False, maxfev=1000):
    """
    Fit rows or columns of matrix to function_to_fit.
    :param matrix: pandas DataFrame;
    :param function_to_fit: function;
    :param axis: int;
    :param sort_matrix: bool;
    :param maxfev: int;
    :return: list; fit parameters
    """

    # Copy
    matrix = array(matrix)

    if axis == 1:  # Transpose
        matrix = matrix.T

    if sort_matrix:  # Sort by column
        matrix = sort(matrix, axis=0)

    x = array(range(matrix.shape[0]))
    y = sum(matrix, axis=1) / matrix.shape[1]
    fit_parameters = curve_fit(function_to_fit, x, y, maxfev=maxfev)[0]

    return fit_parameters


# ======================================================================================================================
# Network
# ======================================================================================================================
def make_network_from_similarity_matrix(similarity_matrix):
    """

    :param similarity_matrix:
    :return:
    """

    graph = Graph()
    for i, s in similarity_matrix.iterrows():
        for j in s.index:
            graph.add_edge(s.name, j, weight=s.ix[j])


def make_network_from_edge_file(edge_file, di=False, sep='\t'):
    """
    Make networkx graph from edge_file: from<sep>to.
    :param edge_file:
    :param di: boolean, directed or not
    :param sep: separator, default \t
    :return:
    """
    # Load edge
    e = read_csv(edge_file, sep=sep)

    # Make graph from edge
    if di:
        # Directed graph
        g = DiGraph()
    else:
        # Undirected graph
        g = Graph()

    g.add_edges_from(e.values)

    return g


# ======================================================================================================================
# Cluster
# ======================================================================================================================
def hierarchical_consensus_cluster(matrix, ks, distance_matrix=None, function=information_coefficient,
                                   n_clusterings=100):
    """
    Consensus cluster matrix's columns into k clusters.
    :param matrix: pandas DataFrame; (n_features, m_samples)
    :param ks: iterable; list of ks used for clustering
    :param distance_matrix: str or DataFrame;
    :param function: function; distance function
    :param n_clusterings: int; number of clusterings for the consensus clustering
    :return: DataFrame and Series; assignment matrix (n_ks, n_samples) and cophenetic correlation coefficients (n_ks)
    """

    if isinstance(distance_matrix, DataFrame):
        print_log('Loading distances between samples already computed ...')
        if isinstance(distance_matrix, str):
            distance_matrix = read_csv(distance_matrix, sep='\t', index_col=0)
    else:
        # Compute sample-distance matrix
        print_log('Computing distances between samples, making a distance matrix ...')
        distance_matrix = compare_matrices(matrix, matrix, function, is_distance=True)

    # Consensus cluster distance matrix
    print_log('Consensus clustering with {} clusterings ...'.format(n_clusterings))
    clusterings = DataFrame(index=ks, columns=list(matrix.columns))
    clusterings.index.name = 'k'
    cophenetic_correlation_coefficients = {}

    if isinstance(ks, int):
        ks = [ks]

    for k in ks:
        print_log('k={} ...'.format(k))

        # For n_clusterings times, permute distance matrix with repeat, and cluster

        # Make sample x clustering matrix
        sample_x_clustering = DataFrame(index=matrix.columns, columns=range(n_clusterings))
        for i in range(n_clusterings):
            if i % 10 == 0:
                print_log('\tPermuting distance matrix with repeat and clustering ({}/{}) ...'.format(i, n_clusterings))

            # Randomize samples with repeat
            random_indices = random_integers(0, distance_matrix.shape[0] - 1, distance_matrix.shape[0])

            # Cluster random samples
            hierarchical_clustering = AgglomerativeClustering(n_clusters=k)
            hierarchical_clustering.fit(distance_matrix.iloc[random_indices, random_indices])

            # Assign cluster labels to the random samples
            sample_x_clustering.iloc[random_indices, i] = hierarchical_clustering.labels_

        # Make consensus matrix using labels created by clusterings of randomized distance matrix
        print_log('\tMaking consensus matrix from {} hierarchical clusterings of randomized distance matrix ...'.format(
            n_clusterings))
        consensus_matrix = get_consensus(sample_x_clustering)

        # Hierarchical cluster consensus_matrix's distance matrix and compute cophenetic correlation coefficient
        hierarchical_clustering, cophenetic_correlation_coefficient = \
            _hierarchical_cluster_consensus_matrix(consensus_matrix, method='average')
        cophenetic_correlation_coefficients[k] = cophenetic_correlation_coefficient

        # Get labels from hierarchical clustering
        clusterings.ix[k, :] = fcluster(hierarchical_clustering, k, criterion='maxclust')

    return distance_matrix, clusterings, cophenetic_correlation_coefficients


def _hierarchical_cluster_consensus_matrix(consensus_matrix, force_diagonal=True, method='ward'):
    """
    Hierarchical cluster consensus_matrix and compute cophenetic correlation coefficient.
    Convert consensus_matrix into distance matrix. Hierarchical cluster the distance matrix. And compute the
    cophenetic correlation coefficient.
    :param consensus_matrix: DataFrame;
    :param force_diagonal: bool;
    :param method: str; method parameter for scipy.cluster.hierarchy.linkage
    :return: ndarray float; linkage (Z) and cophenetic correlation coefficient
    """

    # Convert consensus matrix into distance matrix
    distance_matrix = 1 - consensus_matrix
    if force_diagonal:
        for i in range(distance_matrix.shape[0]):
            distance_matrix.iloc[i, i] = 0

    # Cluster consensus matrix to assign the final label
    hierarchical_clustering = linkage(consensus_matrix, method=method)

    # Compute cophenetic correlation coefficient
    cophenetic_correlation_coefficient = pearsonr(pdist(distance_matrix), cophenet(hierarchical_clustering))[0]

    return hierarchical_clustering, cophenetic_correlation_coefficient


# ======================================================================================================================
# NMF
# ======================================================================================================================
def nmf_consensus_cluster(matrix, ks, n_jobs=1, n_clusterings=100,
                          init='random', solver='cd', tol=1e-6, max_iter=1000, random_state=SEED, alpha=0, l1_ratio=0,
                          shuffle_=False, nls_max_iter=2000, sparseness=None, beta=1, eta=0.1):
    """
    Perform NMF with k from ks and _score each NMF decomposition.
    :param matrix: numpy array or pandas DataFrame; (n_samples, n_features); the matrix to be factorized by NMF
    :param ks: iterable; list of ks to be used in the NMF
    :param n_jobs: int;
    :param n_clusterings:
    :param init:
    :param solver:
    :param tol:
    :param max_iter:
    :param random_state:
    :param alpha:
    :param l1_ratio:
    :param shuffle_:
    :param nls_max_iter:
    :param sparseness:
    :param beta:
    :param eta:
    :return: dict and dict; {k: {W:w_matrix, H:h_matrix, ERROR:reconstruction_error}} and
                            {k: cophenetic correlation coefficient}
    """

    if isinstance(ks, int):
        ks = [ks]
    else:
        ks = list(set(ks))

    nmf_results = {}
    cophenetic_correlation_coefficients = {}

    print_log('Computing cophenetic correlation coefficient of {} NMF consensus clusterings ...'.format(n_clusterings))

    if len(ks) > 1:
        print_log('Parallelizing ...')
        args = [[matrix, k, n_clusterings, init, solver, tol, max_iter, random_state, alpha, l1_ratio, shuffle_,
                 nls_max_iter, sparseness, beta, eta] for k in ks]

        for nmf_result, nmf_score in parallelize(_nmf_and_score, args, n_jobs=n_jobs):
            nmf_results.update(nmf_result)
            cophenetic_correlation_coefficients.update(nmf_score)
    else:
        print_log('Not parallelizing ...')
        nmf_result, nmf_score = _nmf_and_score([matrix, ks[0], n_clusterings, init, solver, tol, max_iter, random_state,
                                                alpha, l1_ratio, shuffle_, nls_max_iter, sparseness, beta, eta])
        nmf_results.update(nmf_result)
        cophenetic_correlation_coefficients.update(nmf_score)

    return nmf_results, cophenetic_correlation_coefficients


def _nmf_and_score(args):
    """
    NMF and score using 1 k.
    :param args:
    :return:
    """

    matrix, k, n_clusterings = args[:3]
    init, solver, tol, max_iter, random_state, alpha, l1_ratio, shuffle_, nls_max_iter, sparseness, beta, eta = args[3:]

    print_log('NMF and scoring k={} ...'.format(k))

    nmf_results = {}
    cophenetic_correlation_coefficients = {}

    # NMF cluster n_clustering
    # TODO: check initialization type for all arrays and dataframes
    sample_x_clustering = DataFrame(index=matrix.columns, columns=range(n_clusterings), dtype=int)
    for i in range(n_clusterings):
        if i % 10 == 0:
            print_log('\t(k={}) NMF ({}/{}) ...'.format(k, i, n_clusterings))

        # NMF
        nmf_result = nmf(matrix, k,
                         init=init, solver=solver, tol=tol, max_iter=max_iter, random_state=random_state,
                         alpha=alpha, l1_ratio=l1_ratio, shuffle_=shuffle_, nls_max_iter=nls_max_iter,
                         sparseness=sparseness, beta=beta, eta=eta)[k]

        # Save the first NMF decomposition for each k
        if i == 0:
            nmf_results[k] = nmf_result
            print_log('\t\t(k={}) Saved the 1st NMF decomposition.'.format(k))

        # Column labels are the row index holding the highest value
        sample_x_clustering.iloc[:, i] = argmax(asarray(nmf_result['H']), axis=0)

    # Make consensus matrix using NMF labels
    print_log('\t(k={}) Making consensus matrix from {} NMF ...'.format(k))
    consensus_matrix = get_consensus(sample_x_clustering)

    # Hierarchical cluster consensus_matrix's distance matrix and compute cophenetic correlation coefficient
    distance_matrix, hierarchical_clustering, cophenetic_correlation_coefficient = \
        _hierarchical_cluster_consensus_matrix(consensus_matrix)
    cophenetic_correlation_coefficients[k] = cophenetic_correlation_coefficient

    return nmf_results, cophenetic_correlation_coefficients


def nmf(matrix, ks, init='random', solver='cd', tol=1e-6, max_iter=1000, random_state=SEED,
        alpha=0, l1_ratio=0, shuffle_=False, nls_max_iter=2000, sparseness=None, beta=1, eta=0.1):
    """
    Nonenegative matrix factorize matrix with k from ks.
    :param matrix: numpy array or pandas DataFrame; (n_samples, n_features); the matrix to be factorized by NMF
    :param ks: iterable; list of ks to be used in the NMF
    :param init:
    :param solver:
    :param tol:
    :param max_iter:
    :param random_state:
    :param alpha:
    :param l1_ratio:
    :param shuffle_:
    :param nls_max_iter:
    :param sparseness:
    :param beta:
    :param eta:
    :return: dict; {k: {W:w_matrix, H:h_matrix, ERROR:reconstruction_error}}
    """

    if isinstance(ks, int):
        ks = [ks]
    else:
        ks = list(set(ks))

    nmf_results = {}
    for k in ks:

        # Compute W, H, and reconstruction error
        model = NMF(n_components=k, init=init, solver=solver, tol=tol, max_iter=max_iter, random_state=random_state,
                    alpha=alpha, l1_ratio=l1_ratio, shuffle=shuffle_, nls_max_iter=nls_max_iter, sparseness=sparseness,
                    beta=beta, eta=eta)
        w, h, err = model.fit_transform(matrix), model.components_, model.reconstruction_err_

        # Return pandas DataFrame if the input matrix is also a DataFrame
        if isinstance(matrix, DataFrame):
            w = DataFrame(w, index=matrix.index)
            h = DataFrame(h, columns=matrix.columns)

        # Save NMF results
        nmf_results[k] = {'W': w, 'H': h, 'ERROR': err}

    return nmf_results


def solve_matrix_linear_equation(a, b, method='nnls'):
    """
    Solve a * x = b of (n, k) * (k, m) = (n, m).
    :param a: numpy array; (n, k)
    :param b: numpy array; (n, m)
    :param method: str; {'nnls', 'pinv'}
    :return: numpy array; (k, m)
    """
    if method == 'nnls':
        x = DataFrame(index=a.columns, columns=b.columns)
        for i in range(b.shape[1]):
            x.iloc[:, i] = nnls(a, b.iloc[:, i])[0]
    elif method == 'pinv':
        a_pinv = pinv(a)
        x = dot(a_pinv, b)
        x[x < 0] = 0
        x = DataFrame(x, index=a.columns, columns=b.columns)
    else:
        raise ValueError('Unknown method {}. Choose from [\'nnls\', \'pinv\']'.format(method))
    return x


# ======================================================================================================================
# Simulation
# ======================================================================================================================
def simulate_dataframe_or_series(n_rows, n_cols, n_categories=None):
    """
    Simulate DataFrame (2D) or Series (1D).
    :param n_rows: int;
    :param n_cols: int;
    :param n_categories: None or int; continuous if None and categorical if int
    :return: pandas DataFrame or Series; (n_rows, n_cols) or (1, n_cols)
    """

    # Set up indices and column names
    indices = ['Feature {}'.format(i) for i in range(n_rows)]
    columns = ['Sample {}'.format(i) for i in range(n_cols)]

    # Set up data type: continuous, categorical, or binary
    if n_categories:
        features = DataFrame(random_integers(0, n_categories - 1, (n_rows, n_cols)), index=indices, columns=columns)
    else:
        features = DataFrame(random_sample((n_rows, n_cols)), index=indices, columns=columns)

    if n_rows == 1:  # Return as series if there is only 1 row
        return features.iloc[0, :]
    else:
        return features


# ======================================================================================================================
# Plotting
# ======================================================================================================================
FIGURE_SIZE = (16, 10)

SPACING = 0.05

# Fonts
FONT_TITLE = {'fontsize': 26, 'weight': 'bold'}
FONT_SUBTITLE = {'fontsize': 20, 'weight': 'bold'}
FONT = {'fontsize': 12, 'weight': 'bold'}

# Color maps
CMAP_CONTINUOUS = bwr
CMAP_CONTINUOUS.set_bad('wheat')

reds = [0.26, 0.26, 0.26, 0.39, 0.69, 1, 1, 1, 1, 1, 1]
greens_half = [0.26, 0.16, 0.09, 0.26, 0.69]
colordict = {'red': tuple([(0.1 * i, r, r) for i, r in enumerate(reds)]),
             'green': tuple([(0.1 * i, r, r) for i, r in enumerate(greens_half + [1] + list(reversed(greens_half)))]),
             'blue': tuple([(0.1 * i, r, r) for i, r in enumerate(reversed(reds))])}
CMAP_ASSOCIATION = LinearSegmentedColormap('association', colordict)
CMAP_ASSOCIATION.set_bad('wheat')

CMAP_CATEGORICAL = Paired
CMAP_CATEGORICAL.set_bad('wheat')

CMAP_BINARY = light_palette('black', n_colors=2, as_cmap=True)
CMAP_BINARY.set_bad('wheat')

DPI = 1000


def plot_heatmap(dataframe, data_type='continuous',
                 normalization_method=None, normalization_axis=0, sort_axis=None, vmin=None, vmax=None,
                 row_annotation=(), column_annotation=(),
                 center=None,
                 annot=None, fmt='.2g', annot_kws=None,
                 linewidth=0, linecolor='white',
                 mask=None,
                 square=False,
                 title=None, xlabel=None, ylabel=None, xlabel_rotation=0, ylabel_rotation=90,
                 xticklabels=True, yticklabels=True, yticklabels_rotation='auto',
                 filepath=None):
    """

    :param dataframe:
    :param data_type:
    :param normalization_method:
    :param normalization_axis:
    :param sort_axis:
    :param vmin:
    :param vmax:
    :param row_annotation:
    :param column_annotation:
    :param center:
    :param annot:
    :param fmt:
    :param annot_kws:
    :param linewidth:
    :param linecolor:
    :param mask:
    :param square:
    :param title:
    :param xlabel:
    :param ylabel:
    :param xlabel_rotation:
    :param ylabel_rotation:
    :param xticklabels:
    :param yticklabels:
    :param yticklabels_rotation:
    :param filepath:
    :return:
    """

    df = dataframe.copy()

    if normalization_method:
        df = normalize_pandas_object(df, normalization_method, axis=normalization_axis)
    values = unique(df.values)

    if any(row_annotation) or any(column_annotation):
        if any(row_annotation):
            if isinstance(row_annotation, Series):
                if not any(row_annotation.index & df.index):
                    row_annotation.index = df.index
            else:
                row_annotation = Series(row_annotation, index=df.index)

            row_annotation.sort_values(inplace=True)
            df = df.ix[row_annotation.index, :]

        if any(column_annotation):
            column_annotation = Series(column_annotation, index=df.columns)
            column_annotation.sort_values(inplace=True)
            df = df.ix[:, column_annotation.index]
    elif sort_axis in (0, 1):
        a = array(df)
        a.sort(axis=sort_axis)
        df = DataFrame(a)

    plt.figure(figsize=FIGURE_SIZE)

    if title:
        plt.suptitle(title, **FONT_TITLE)

    gridspec = GridSpec(10, 10)

    ax_top = plt.subplot(gridspec[0:1, 2:-2])
    ax_center = plt.subplot(gridspec[1:8, 2:-2])
    ax_bottom = plt.subplot(gridspec[8:10, 2:-2])
    ax_left = plt.subplot(gridspec[1:8, 1:2])
    ax_right = plt.subplot(gridspec[1:8, 8:9])

    ax_top.axis('off')
    ax_bottom.axis('off')
    ax_left.axis('off')
    ax_right.axis('off')

    if data_type == 'continuous':
        cmap = CMAP_CONTINUOUS
    elif data_type == 'categorical':
        cmap = CMAP_CATEGORICAL
    elif data_type == 'binary':
        cmap = CMAP_BINARY
    else:
        raise ValueError('Target data type must be one of {continuous, categorical, binary}.')

    heatmap(df, vmin=vmin, vmax=vmax, center=center, annot=annot, fmt=fmt, annot_kws=annot_kws,
            linewidths=linewidth, linecolor=linecolor, cbar=False, square=square, mask=mask,
            cmap=cmap, xticklabels=xticklabels, yticklabels=yticklabels, ax=ax_center)

    if xlabel:
        ax_center.set_xlabel(xlabel, rotation=xlabel_rotation, **FONT_SUBTITLE)
    if ylabel:
        ax_center.set_ylabel(ylabel, rotation=ylabel_rotation, **FONT_SUBTITLE)

    for t in ax_center.get_xticklabels():
        t.set(**FONT)

    yticks = ax_center.get_yticklabels()
    if any(yticks):
        if yticklabels_rotation == 'auto':
            if max([len(t.get_text()) for t in yticks]) <= 1:
                yticklabels_rotation = 0
            else:
                yticklabels_rotation = 90
        for t in yticks:
            t.set(rotation=yticklabels_rotation, **FONT)

    if data_type in ('categorical', 'binary'):
        if len(values) < 30:
            horizontal_span = ax_center.axis()[1]
            vertival_span = ax_center.axis()[3]
            for i, v in enumerate(values):
                x = (horizontal_span / len(values) / 2) + i * horizontal_span / len(values)
                y = 0 - vertival_span * 0.09
                ax_center.plot(x, y, 'o', markersize=16, aa=True, clip_on=False)
                ax_center.text(x, y - vertival_span * 0.05, v, horizontalalignment='center', **FONT)

    if data_type == 'continuous':
        cax, kw = make_axes(ax_bottom, location='bottom', fraction=0.16,
                            cmap=CMAP_CONTINUOUS,
                            norm=Normalize(values.min(), values.max()),
                            ticks=[values.min(), values.mean(), values.max()])
        ColorbarBase(cax, **kw)

    if any(row_annotation):
        if len(set(row_annotation)) <= 2:
            cmap = CMAP_BINARY
        else:
            cmap = CMAP_CATEGORICAL
        heatmap(DataFrame(row_annotation), ax=ax_right, cbar=False, xticklabels=False, yticklabels=False,
                cmap=cmap)

    if any(column_annotation):
        if len(set(column_annotation)) <= 2:
            cmap = CMAP_BINARY
        else:
            cmap = CMAP_CATEGORICAL
        heatmap(DataFrame(column_annotation).T, ax=ax_top, cbar=False, xticklabels=False, yticklabels=False,
                cmap=cmap)

    if filepath:
        save_plot(filepath)


def plot_clustermap(dataframe, cmap=CMAP_CONTINUOUS, row_colors=None, col_colors=None,
                    title=None, xlabel=None, ylabel=None, xticklabels=True, yticklabels=True,
                    filepath=None):
    """
    Plot heatmap for dataframe.
    :param dataframe: pandas DataFrame;
    :param cmap: colormap;
    :param row_colors: list-like or pandas DataFrame/Series; List of colors to label for either the rows.
        Useful to evaluate whether samples within a group_iterable are clustered together.
        Can use nested lists or DataFrame for multiple color levels of labeling.
        If given as a DataFrame or Series, labels for the colors are extracted from
        the DataFrames column names or from the name of the Series. DataFrame/Series colors are also matched to the data
        by their index, ensuring colors are drawn in the correct order.
    :param col_colors: list-like or pandas DataFrame/Series; List of colors to label for either the column.
        Useful to evaluate whether samples within a group_iterable are clustered together.
        Can use nested lists or DataFrame for multiple color levels of labeling.
        If given as a DataFrame or Series, labels for the colors are extracted from
        the DataFrames column names or from the name of the Series. DataFrame/Series colors are also matched to the data
        by their index, ensuring colors are drawn in the correct order.
    :param title: str;
    :param xlabel: str;
    :param ylabel: str;
    :param xticklabels: bool;
    :param yticklabels: bool;
    :param filepath: str;
    :return: None
    """

    # Initialize a figure
    plt.figure(figsize=FIGURE_SIZE)

    # Plot cluster map
    clustergrid = clustermap(dataframe, cmap=cmap, row_colors=row_colors, col_colors=col_colors,
                             xticklabels=xticklabels, yticklabels=yticklabels, )

    # Title
    if title:
        plt.suptitle(title, **FONT_TITLE)

    # X & Y labels
    if xlabel:
        clustergrid.ax_heatmap.set_xlabel(xlabel, **FONT_SUBTITLE)
    if ylabel:
        clustergrid.ax_heatmap.set_ylabel(ylabel, **FONT_SUBTITLE)

    # X & Y ticks
    for t in clustergrid.ax_heatmap.get_xticklabels():
        t.set(**FONT)
    for t in clustergrid.ax_heatmap.get_yticklabels():
        t.set(**FONT)

    # Save
    if filepath:
        save_plot(filepath)


def plot_x_vs_y(x, y, title='title', xlabel='xlabel', ylabel='ylabel', filepath=None):
    """
    Plot x vs y.
    :param x:
    :param y:
    :param title:
    :param xlabel:
    :param ylabel:
    :param filepath:
    :return:
    """

    plt.figure(figsize=FIGURE_SIZE)

    if title:
        plt.suptitle(title, **FONT_TITLE)

    pointplot(x, y)

    plt.gca().set_xlabel(xlabel, **FONT_SUBTITLE)
    plt.gca().set_ylabel(ylabel, **FONT_SUBTITLE)

    if filepath:
        save_plot(filepath)


def plot_nmf(nmf_results=None, k=None, w_matrix=None, h_matrix=None, normalize=False, max_std=3, title=None,
             filepath=None):
    """
    Plot nmf_results dictionary (can be generated by ccal.analyze.nmf function).
    :param nmf_results: dict; {k: {W:w, H:h, ERROR:error}}
    :param k: int; k for NMF
    :param w_matrix: pandas DataFrame
    :param h_matrix: Pandas DataFrame
    :param normalize: bool; normalize W and H matrices or not ('-0-' normalization on the component axis)
    :param max_std: number; threshold to clip standardized values
    :param title: str;
    :param filepath: str;
    :return: None
    """

    # Check for W and H matrix
    if isinstance(nmf_results, dict) and k:
        w_matrix = nmf_results[k]['W']
        h_matrix = nmf_results[k]['H']
    elif not (isinstance(w_matrix, DataFrame) and isinstance(h_matrix, DataFrame)):
        raise ValueError('Need either: 1) NMF result ({k: {W:w, H:h, ERROR:error}) and k; or 2) W and H matrices.')

    # Initialize a PDF
    if filepath:
        establish_filepath(filepath)
        if not filepath.endswith('.pdf'):
            filepath += '.pdf'
        pdf = PdfPages(filepath)

    # Plot W and H
    plt.figure(figsize=FIGURE_SIZE)
    gridspec = GridSpec(10, 16)
    ax_w = plt.subplot(gridspec[1:, :5])
    ax_h = plt.subplot(gridspec[3:8, 7:])
    if not title:
        title = 'NMF Result for k={}'.format(w_matrix.shape[1])
    plt.suptitle(title, **FONT_TITLE)
    # Plot W
    if normalize:
        w_matrix = normalize_pandas_object(w_matrix, method='-0-', axis=0).clip(-max_std, max_std)
    heatmap(w_matrix, cmap=CMAP_CONTINUOUS, yticklabels=False, ax=ax_w)
    ax_w.set_title('W Matrix for k={}'.format(w_matrix.shape[1]), **FONT_TITLE)
    ax_w.set_xlabel('Component', **FONT_SUBTITLE)
    ax_w.set_ylabel('Feature', **FONT_SUBTITLE)
    # Plot H
    if normalize:
        h_matrix = normalize_pandas_object(h_matrix, method='-0-', axis=1).clip(-max_std, max_std)
    heatmap(h_matrix, cmap=CMAP_CONTINUOUS, xticklabels=False, cbar_kws={'orientation': 'horizontal'}, ax=ax_h)
    ax_h.set_title('H Matrix for k={}'.format(h_matrix.shape[0]), **FONT_TITLE)
    ax_h.set_xlabel('Sample', **FONT_SUBTITLE)
    ax_h.set_ylabel('Component', **FONT_SUBTITLE)
    if filepath:
        plt.savefig(pdf, format='pdf', dpi=DPI, bbox_inches='tight')

    # Plot cluster map for W
    clustergrid = clustermap(w_matrix, standard_scale=0, figsize=FIGURE_SIZE, cmap=CMAP_CONTINUOUS)
    plt.suptitle('W Matrix for k={}'.format(w_matrix.shape[1]), **FONT_TITLE)
    clustergrid.ax_heatmap.set_xlabel('Component', **FONT_SUBTITLE)
    clustergrid.ax_heatmap.set_ylabel('Feature', **FONT_SUBTITLE)
    for t in clustergrid.ax_heatmap.get_xticklabels():
        t.set_fontweight('bold')
    for t in clustergrid.ax_heatmap.get_yticklabels():
        t.set_visible(False)
    if filepath:
        plt.savefig(pdf, format='pdf', dpi=DPI, bbox_inches='tight')

    # Plot cluster map for H
    clustergrid = clustermap(h_matrix, standard_scale=1, figsize=FIGURE_SIZE, cmap=CMAP_CONTINUOUS)
    plt.suptitle('H Matrix for k={}'.format(h_matrix.shape[0]), **FONT_TITLE)
    clustergrid.ax_heatmap.set_xlabel('Sample', **FONT_SUBTITLE)
    clustergrid.ax_heatmap.set_ylabel('Component', **FONT_SUBTITLE)
    for t in clustergrid.ax_heatmap.get_xticklabels():
        t.set_visible(False)
    for t in clustergrid.ax_heatmap.get_yticklabels():
        t.set_fontweight('bold')
        t.set_rotation(0)
    if filepath:
        plt.savefig(pdf, format='pdf', dpi=DPI, bbox_inches='tight')

    if filepath:
        pdf.close()


def save_plot(filepath, suffix='.pdf', dpi=DPI):
    """
    Establish filepath and save plot (.pdf) at dpi resolution.
    :param filepath: str;
    :param suffix: str;
    :param dpi: int;
    :return: None
    """

    if not filepath.endswith(suffix):
        filepath += suffix

    establish_filepath(filepath)

    plt.savefig(filepath, dpi=dpi, bbox_inches='tight')
