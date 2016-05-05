import pandas as pd
import numpy as np
import sys
import os
import random


def slice_list(a_list, n):
    """
    Given <a_list> and <n>, return a list containing <n> lists, with the items of <a_list> evenly distributed.
    :param a_list:
    :param n:
    :return:
    """
    input_size = len(a_list)
    slice_size = int(input_size / n)
    remain = input_size % n
    result = []
    iterator = iter(a_list)
    for i in range(n):
        result.append([])
        for j in range(slice_size):
            result[i].append(next(iterator))
        if remain:
            result[i].append(next(iterator))
            remain -= 1
    return result


def add_value(df, val_in, val_out):
    for i, (n, s) in enumerate(df.iterrows()):
        for j, c in enumerate(s.index):
            if val_in and n == c:
                df.iloc[i, j] = val_in
            if val_out and n != c:
                df.iloc[i, j] = val_out


def add_noise(df, mean_in, std_in, mean_out, std_out):
    for i, (n, s) in enumerate(df.iterrows()):
        for j, c in enumerate(s.index):
            if (mean_in or std_in) and n == c:
                df.iloc[i, j] += random.gauss(mean_in, std_in)
            if (mean_out or std_out) and n != c:
                df.iloc[i, j] += random.gauss(mean_out, std_out)


def shuffle(df, k, shuffle_fraction):
    assert k != 0, 'k cannot be 0'

    # Get the number of values in a cluster
    n_k_val = len(df.columns) * len(df.index) / k

    # Count
    c = 0
    while c < shuffle_fraction * n_k_val:
        # Pick 1st random index and column
        r_idx0 = random.randint(0, len(df.index) - 1)
        r_col0 = random.randint(0, len(df.columns) - 1)
        # If index and column locate inside a cluster
        if df.index[r_idx0] == df.columns[r_col0]:
            # Get cluster value located
            pick0 = df.iloc[r_idx0, r_col0]

            # Pick 2nd random index and column
            r_idx1 = random.randint(0, len(df.index) - 1)
            r_col1 = random.randint(0, len(df.columns) - 1)
            # If index and column locate outside a cluster
            if df.index[r_idx1] != df.columns[r_col1]:
                # Get non-cluster value located
                pick1 = df.iloc[r_idx1, r_col1]

                # Swap
                df.iloc[r_idx0, r_col0] = pick1
                df.iloc[r_idx1, r_col1] = pick0

                # Count
                c += 1


def initialize_simulation_matrix(df,
                                 val_in, noise_mean_in, noise_std_in,
                                 val_out, noise_mean_out, noise_std_out,
                                 shuffle_fraction):
    # For each row
    for i, (n, s) in enumerate(df.iterrows()):
        # For each column
        for j, c in enumerate(s.index):
            r = random.random()

            if shuffle_fraction and r <= shuffle_fraction:
                # Shuffle
                if n == c:
                    # In cluster gets out-value
                    df.iloc[i, j] = val_out
                    if noise_mean_out or noise_std_out:
                        df.iloc[i, j] += random.gauss(noise_mean_out, noise_std_out)
                else:
                    # Out cluster gets in-value
                    df.iloc[i, j] = val_in
                    if noise_mean_in or noise_std_in:
                        df.iloc[i, j] += random.gauss(noise_mean_in, noise_std_in)
            else:
                # No shuffle
                if n == c:
                    # In cluster gets in-value
                    df.iloc[i, j] = val_in
                    if noise_mean_in or noise_std_in:
                        df.iloc[i, j] += random.gauss(noise_mean_in, noise_std_in)
                else:
                    # Out cluster gets out-value
                    df.iloc[i, j] = val_out
                    if noise_mean_out or noise_std_out:
                        df.iloc[i, j] += random.gauss(noise_mean_out, noise_std_out)


def make_simulation_matrix(n_sample, n_feature, k,
                           val_in, val_out,
                           noise_mean_in=None, noise_std_in=None,
                           noise_mean_out=None, noise_std_out=None,
                           shuffle_fraction=None,
                           output_dir=None):
    assert k != 0, 'k cannot be 0'
    assert k <= n_feature, 'k cannot be greater than n_feature'

    # Make an empty sample x feature matrix filled with 0
    matrix = pd.DataFrame(index=range(n_sample), columns=range(n_feature)).fillna(0)

    # Slice matrix index and column and make lists of matrix indexes and columns for each index and column slice
    list_index_slice = toolK.slice_list(matrix.index, k)
    list_column_slice = toolK.slice_list(matrix.columns, k)

    # Make a dictionary of slice index to slice (list)
    dict_index_slice = {}
    for i, l in enumerate(list_index_slice):
        dict_index_slice[i] = l
    dict_column_slice = {}
    for i, l in enumerate(list_column_slice):
        dict_column_slice[i] = l

    # Update matrix index to be the slice index
    index = list(matrix.index)
    for i, l in dict_index_slice.items():
        for j in l:
            index[j] = i
    matrix.index = index
    columns = list(matrix.columns)
    for i, l in dict_column_slice.items():
        for j in l:
            columns[j] = i
    matrix.columns = columns

    # Initialize simulation matrix
    initialize_simulation_matrix(matrix,
                                 val_in, noise_mean_in, noise_std_in,
                                 val_out, noise_mean_out, noise_std_out,
                                 shuffle_fraction)

    # Save
    if output_dir:
        matrix.to_csv(os.path.join(output_dir, '{}x{}_k{}_shuffle{}.tsv'.format(n_sample, n_feature, k, shuffle_fraction), sep='\t'))


# Make simulation matrix

# Set number of samples
list_n_sample = [100, 500, 1000, 5000]
# Set number of features
list_n_feature = [100, 500, 1000, 5000]
# Set values in clusters
val_in = 1
# Set values out of clusters
val_out = 0
# Set Ks
list_k = [1, 2, 3, 4, 5, 6, 10, 15, 20, 25]
# Set the fractions of cluster values to be swapped between clusters and nonclusters
list_shuffle_fraction = [0, 0.05, 0.1, 0.2]
# Set noise in clusters
noise_mean_in = 0
noise_std_in = 0.1 * noise_mean_in
# Set noise out of clusters
noise_mean_out = 0
noise_std_out = 0.1 * noise_mean_out

# Simulate
for sample in list_n_sample:
    print('sample:', sample)

    for feature in list_n_feature:
        print('\tfeature:', feature)

        for k in list_k:
            print('\t\tk:', k)

            for shuffle_fraction in list_shuffle_fraction:
                print('\t\t\tshuffle_fraction:', shuffle_fraction)

                make_simulation_matrix(sample, feature, k,
                                       val_in, val_out,
                                       shuffle_fraction=shuffle_fraction,
                                       output_dir='/Users/Kwat/binf/ccba/data/test')
