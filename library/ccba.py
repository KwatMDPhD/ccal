"""
Cancer Computational Biology Analysis Library v0.1

Authors:
Pablo Tamayo
pablo.tamayo.r@gmail.com
Genomics and Computational Biology, UCSD Moore's Cancer Center

Huwate (Kwat) Yeerna (Medetgul-Ernar)
kwat.medetgul.ernar@gmail.com
Genomics and Computational Biology, UCSD Moore's Cancer Center

James Jensen
Email
Affiliation
"""


## Check dependencies and install missing ones
import pip
packages_installed = pip.get_installed_distributions()
package_names_installed = [pkg.key for pkg in packages_installed]
package_names_needed = ['rpy2', 'numpy', 'pandas', 'scikit-learn', 'matplotlib', 'seaborn']
for pkg in package_names_needed:
    if pkg not in package_names_installed:
        print('{} not found! Installing ......'.format(pkg))
        pip.main(['install', pkg])
print('Using the following packages:')
for pkg in packages_installed:
    if pkg.key in package_names_needed:
        print('{} v{}'.format(pkg.key, pkg.version))

import os
import numpy as np
import pandas as pd
from sklearn.decomposition import NMF
from library.support import *
from library.visualize import *
from library.information import *


## Define Global variable
TESTING = False
SEED = 20121020
# Path to CCBA dicrectory (repository)
PATH_CCBA = '/Users/Kwat/binf/ccba/'
# Path to testing data directory
PATH_TEST_DATA = os.path.join(PATH_CCBA, 'data', 'test')


def make_heatmap_panel(dataframe, reference, metric=['IC'], sort_column=['IC'], title=None, v=False):
    """
    Compute score[i] = <dataframe>[i] vs. <reference> and append score as a column to <dataframe>.
    
    :param 
    """
    # Compute score[i] = <dataframe>[i] vs. <reference> and append score as a column to <dataframe>
    if 'IC' in metric:
        dataframe.ix[:, 'IC'] = pd.Series([compute_information_coefficient(np.array(row[1]), reference) for row in dataframe.iterrows()], index=dataframe.index)

    # Sort
    dataframe.sort(sort_column, inplace=True)
    
    # Plot
    plot_heatmap_panel(dataframe, reference, metric, title=title)


def nmf(X, n_components, initialization='random', iteration=200, seed=SEED, randomize_coordinate_order=False, regulatizer=0, v=False):
    """
    Nonenegative matrix mactorize <X> and return W, H, and their reconstruction error.
    
    :param initialization: {'random', 'nndsvd', 'nndsvda', 'nndsvdar'}
    """
    model = NMF(n_components=n_components,
                init=initialization,
                max_iter=iteration,
                random_state=seed,
                alpha=regulatizer,
                shuffle=randomize_coordinate_order)
    if v: print('Reconstruction error: {}'.format(model.reconstruction_err_))
        
    # return W, H, and reconstruction error
    return model.fit_transform(X), model.components_, model.reconstruction_err_