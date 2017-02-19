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

from sys import platform
from os import mkdir, listdir, remove, rename, environ
from os.path import abspath, split, isdir, isfile, islink, join
import gzip
from csv import reader, writer, excel, excel_tab

from Bio import bgzf
from matplotlib.colors import ListedColormap
from pandas import Series, DataFrame, read_csv, concat

from .log import print_log
from .system import run_cmd


# ======================================================================================================================
# General functions
# ======================================================================================================================
def establish_filepath(filepath):
    """
    If the path up to the deepest directory in filepath doesn't exist, make the path up to the deepest directory.
    :param filepath: str; filepath
    :return: None
    """

    # prefix/suffix
    prefix, suffix = split(filepath)
    prefix = abspath(prefix)

    # Get missing directories
    missing_directories = []
    while not (isdir(prefix) or isfile(prefix) or islink(prefix)):  # prefix isn't file, directory, or link
        missing_directories.append(prefix)

        # Check prefix's prefix next
        prefix, suffix = split(prefix)

    # Make missing directories
    for d in reversed(missing_directories):
        mkdir(d)
        print_log('Created directory {}.'.format(d))


def get_home_dir():
    """

    :return: str; user-home directory
    """

    if 'linux' in platform or 'darwin' in platform:
        home_dir = environ['HOME']
    elif 'win' in platform:
        home_dir = environ['HOMEPATH']
    else:
        raise ValueError('Unknown platform {}.'.format(platform))

    return home_dir


def split_file_extension(filepath):
    """
    Get filepath without file suffix and the suffix from filepath; get foo and txt from foo.txt
    :param filepath: str; filepath
    :return: str and str; filepath without file suffix and the suffix
    """

    split_filepath = filepath.split('.')
    base = ''.join(split_filepath[:-1])
    extension = split_filepath[-1]
    return base, extension


def mark_filename(fname, mark, suffix):
    """
    Convert fname.suffix to fname.mark.suffix.
    :param fname: str;
    :param mark: str;
    :param suffix: str;
    :return: str;
    """

    if not suffix.startswith('.'):
        suffix = '.{}'.format(suffix)

    if suffix in fname:  # Suffix is found in the filename
        i = fname.find(suffix)
        fname = '{}.{}{}'.format(fname[:i], mark, fname[i:])

        if fname.endswith('.gz'):
            fname = fname[:-len('.gz')]

    else:  # Suffix is not found in the filename
        fname = '{}.{}{}'.format(fname, mark, suffix)

    return fname


def list_only_dirs(directory_path):
    """

    :param directory_path: str; directory with all libraries
    :return: list; sorted list of directories in directory_path
    """

    dirs = []
    for f in listdir(directory_path):
        fp = join(directory_path, f)
        if isdir(fp):
            dirs.append(fp)
    dirs = sorted(dirs)

    return dirs


def convert_csv_to_tsv(filepath):
    """
    Convert .csv file to .tsv file.
    :param filepath: str; filepath
    :return: None
    """

    # Open .csv
    with open(filepath, 'rU') as input_filepath:
        r = reader(input_filepath, dialect=excel)

    # Open .tsv
    with open(filepath.strip('.csv') + '.tsv', 'w') as output_filepath:
        w = writer(output_filepath, dialect=excel_tab)

    # Raed from .csv adn write to .tsv
    for line in r:
        w.writerow(line)


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


# ======================================================================================================================
# Compression functions
# ======================================================================================================================
def bgzip_tabix(fname):
    """
    bgzip and tabix <fname>.
    :param fname:
    :return: str;
    """

    if fname.endswith('.gz'):
        if not isfile('{}.tbi'.format(fname)):  # tabixed
            tabix(fname)
        return fname
    else:
        bgzip_output_fname = bgzip(fname)
        tabix(bgzip_output_fname)

    return bgzip_output_fname


def bgzip(fname):
    """
    bgzip <fname>.
    :param fname: str;
    :return: str;
    """

    cmd = 'bgzip -f {}'.format(fname)

    run_cmd(cmd)

    return '{}.gz'.format(fname)


def tabix(fname):
    """
    tabix <fname>.
    :param fname: str;
    :return: str;
    """

    cmd = 'tabix -f {}'.format(fname)

    run_cmd(cmd)

    return '{}.tbi'.format(fname)


def remove_file_family(filepath):
    """
    Remove filepath and its associated files.
    :param filepath:
    :return:
    """

    remove(filepath)
    print_log('Removed {}.'.format(filepath))

    # Remove other associated files
    if filepath.endswith('.gz'):
        remove('rm -rf {}.tbi'.format(filepath))
        print_log('Removed {}.tbi.'.format(filepath))


def convert_gzipped_to_bgzipped(gzipped_filename):
    """

    :param gzipped_filename:
    :return: None
    """

    temp_filename = '{}_temp'.format(gzipped_filename)
    with gzip.open(gzipped_filename, 'rt') as gzipped_file:
        with bgzf.open(temp_filename, 'wt') as temp_bgzipped_file:
            for line in gzipped_file:
                temp_bgzipped_file.write(line)
    remove(gzipped_filename)
    rename(temp_filename, gzipped_filename)


# ======================================================================================================================
# .dict functions
# ======================================================================================================================
def read_dict(filepath, comment_prefix='#', n_skipping_rows=0, sep='\t', switch_key_and_value=False):
    """
    Make a dictionary from mapping_file: key<sep>value.
    By default, 1st column is the key and the 2nd value, and use tab delimeter.
    :param filepath:
    :param comment_prefix: str;
    :param n_skipping_rows: int;
    :param sep:
    :param switch_key_and_value:
    :return: dict;
    """

    # Set column for key and value
    if switch_key_and_value:
        column_names = ['value', 'key']
    else:
        column_names = ['key', 'value']

    # Load mapping info; drop rows with NaN and duplicates
    mapping = read_csv(filepath, comment=comment_prefix, skiprows=n_skipping_rows, sep=sep, names=column_names).dropna()

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


def write_dict(dictionary, filepath, key_name, value_name, sep='\t'):
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


# ======================================================================================================================
# .colormap functions
# ======================================================================================================================
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


def write_colormap(cmap, filepath):
    """

    :param cmap:
    :param filepath:
    :return:
    """

    with open(filepath, 'w') as f:
        for i in range(cmap.N):
            f.writelines('\t'.join([str(c) for c in cmap(i)]) + '\n')


# ======================================================================================================================
# .data_table.txt functions
# ======================================================================================================================
def load_data_table(data_table, indices=None):
    """

    :param data_table: str or DataFrame; path to a .data_table.txt or data_table DataFrame
    :param indices: dict;
        {
            'mutation': {
                'index': ['GNAS_MUT', 'KRAS_MUT'],
                'alias': ['GNAS Mutation', 'KRAS Mutation']
            },
            'gene_expression': {
                'index': ['EGFR'],
                'alias': ['EGFR Expression']
            },
            ...
        }
    :return: dict;
        {
            data_name: {
                'dataframe': DataFrame,
                'data_type': str ('continuous', 'categorical', or 'binary'),
                'emphasis': str ('high' or 'low'),
            }
            ...
        }
    """

    if isinstance(data_table, str):
        data_table = read_data_table(data_table)

    data_bundle = {}
    for data_name, (data_type, emphasis, filepath) in data_table.iterrows():  # For each data
        if isinstance(indices, dict) and data_name not in indices:
            print_log('Skipped loading {} because indices were not specified.'.format(data_name))
            continue

        print_log('Making data bundle for {} ...'.format(data_name))
        data_bundle[data_name] = {}
        df = read_gct(join(filepath))
        print_log('\tLoaded {}.'.format(filepath))
        if isinstance(indices, dict):  # If indices is given
            if data_name in indices:  # Keep specific indices
                index = indices[data_name]['index']
                df = df.ix[index, :]

                # Save the original index names
                data_bundle[data_name]['original_index'] = index

                if 'alias' in indices[data_name]:  # Rename these specific indices
                    df.index = indices[data_name]['alias']

                print('\tSelected rows: {}.'.format(df.index.tolist()))

        data_bundle[data_name]['dataframe'] = df
        data_bundle[data_name]['data_type'] = data_type
        data_bundle[data_name]['emphasis'] = emphasis

    return data_bundle


def write_data_table(data, filepath, columns=('Data Name', 'Data Type', 'Emphasis', 'Filepath')):
    """
    Write <data> to filepath.
    :param data: iterable of tuples;
        [('mutation', 'binary', 'high', 'filepath),
        ('gene_expression', 'continuous', 'high', 'filepath),
        ('drug_sensitivity', 'continuous', 'low', 'filepath),
        ...]
    :param filepath: str; file path to a .data_table (.data_table suffix will be automatically added if not present)
    :param columns: iterable;
    :return: None
    """

    df = DataFrame(data)
    df.columns = columns
    df.set_index(columns[0], inplace=True)
    if not filepath.endswith('data_table.txt'):
        filepath = '{}.data_table.txt'.format(filepath)
    df.to_csv(filepath, sep='\t')


def read_data_table(filepath):
    """
    Read .data_table.
    :param filepath: str; file path to a .data_table
    :return: DataFrame
    """

    return read_csv(filepath, sep='\t', index_col=0)


# ======================================================================================================================
# .gct functions
# ======================================================================================================================
def load_gct(matrix):
    """
    If matrix is a DataFrame, return as is; if a filepath (.gct), read matrix from it as DaraFrame; else, raise Error.
    :param matrix: DataFrame or str; filepath to a .gct or DataFrame
    :return: DataFrame; matrix
    """

    if isinstance(matrix, str):  # Read .gct from a filepath
        matrix = read_gct(matrix)

    elif not isinstance(matrix, DataFrame):  # .gct is not a filepath or DataFrame
        raise ValueError('Matrix must be either a DataFrame or a path to a .gct file.')

    return matrix


def read_gct(filepath, fill_na=None, drop_description=True, row_name=None, column_name=None):
    """
    Read a .gct (filepath) and convert it into a DataFrame.

    :param filepath: str; filepath to .gct
    :param fill_na: *; value to replace NaN in the DataFrame
    :param drop_description: bool; drop the Description column (column 2 in the .gct) or not
    :param row_name: str;
    :param column_name: str;
    :return: DataFrame; [n_samples, n_features (or n_features + 1 if not dropping the Description column)]
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


def write_gct(matrix, filepath, descriptions=None):
    """
    Establish .gct filepath and write matrix to it.
    :param matrix: DataFrame or Serires; (n_samples, m_features)
    :param filepath: str; filepath; adds .gct suffix if missing
    :param descriptions: iterable; (n_samples); description column
    :return: None
    """

    # Copy
    obj = matrix.copy()

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
    establish_filepath(filepath)
    with open(filepath, 'w') as f:
        f.writelines('#1.2\n{}\t{}\n'.format(obj.shape[0], obj.shape[1] - 1))
        obj.to_csv(f, sep='\t')


# ======================================================================================================================
# .gmt functions
# ======================================================================================================================
def read_gmt(filepath, gene_sets=(), drop_description=True, save_clean=True, collapse=False):
    """
    Read .gmt.
    :param filepath: str; filepath to a .gmt file
    :param gene_sets: iterable;
    :param gmt: iterable: list of gene set names to keep
    :param drop_description: bool; drop Desctiption column (2nd column) or not
    :param save_clean: bool; Save as .gmt (cleaned version) or not
    :param collapse: bool; collapse into a list of unique genes or not
    :return: DataFrame or list; (n_gene_sets, size of the largest gene set) or (n_unique genes)
    """

    # Parse
    rows = []
    with open(filepath) as f:
        for line in f.readlines():
            line_split = line.split()
            rows.append(line_split[:2] + sorted(line_split[2:]))

    # Make a dataframe
    gmt = DataFrame(rows)

    # Set index
    gmt.set_index(0, inplace=True)
    gmt.index.name = 'Gene Set'
    gmt.sort_index(inplace=True)
    gmt.columns = ['Description'] + ['Gene {}'.format(i) for i in range(1, gmt.shape[1])]

    if save_clean:  # Save the cleaned version
        gmt.to_csv(filepath, sep='\t', header=False)

    if drop_description or collapse:
        gmt.drop('Description', axis=1, inplace=True)

    if isinstance(gene_sets, str):
        gene_sets = [gene_sets]
    if any(gene_sets):
        gene_sets = sorted(set(gmt.index) & set(gene_sets))
        gmt = gmt.ix[gene_sets, :]
        gmt.dropna(axis=1, how='all', inplace=True)

    if collapse:
        return sorted(set(gmt.unstack().dropna()))
    else:
        return gmt


def read_gmts(filepaths, gene_sets=(), drop_description=True, save_clean=True, collapse=False):
    """
    Read 1 or more .gmt.
    :param filepath: str; filepath to a .gmt file
    :param gmt: iterable: list of gene set names to keep
    :param drop_description: bool; drop Desctiption column (2nd column) or not
    :param save_clean: bool; Save as .gmt (cleaned version) or not
    :param collapse: bool; collapse into a list of unique genes or not
    :return: DataFrame or list; (n_gene_sets, size of the largest gene set) or (n_unique genes)
    """

    gmts = []
    if isinstance(filepaths, str):
        filepaths = [filepaths]
    for fp in filepaths:
        gmt = read_gmt(fp, gene_sets=gene_sets, drop_description=drop_description, save_clean=save_clean)
        gmts.append(gmt)
    gmt = concat(gmts)
    gmt.dropna(axis=1, how='all', inplace=True)
    gmt.sort_index(inplace=True)

    if 'Description' in gmt.columns:
        gmt.columns = ['Description'] + ['Gene {}'.format(i) for i in range(1, gmt.shape[1])]
    else:
        gmt.columns = ['Gene {}'.format(i) for i in range(1, gmt.shape[1] + 1)]

    if collapse:
        return sorted(set(gmt.unstack().dropna()))
    else:
        return gmt


# TODO: check for description?
def write_gmt(gmt, filepath):
    """
    Write a gmt DataFrame to a filepath.gmt.
    :param gmt: DataFrame;
    :param filepath: str; filepath to a .gmt file
    :return: None
    """

    if not filepath.endswith('gmt'):
        filepath += '.gmt'

    gmt.to_csv(filepath, header=None, sep='\t')


# ======================================================================================================================
# .rnk functions
# ======================================================================================================================
def write_rnk(series_or_dataframe, filepath, gene_column=None, score_column=None, comment=None):
    """
    Write .rnk.
    :param series_or_dataframe: Series or DataFrame;
    :param filepath: str;
    :param gene_column: str; column name; dataframe index is the default
    :param score_column: str; column name; 1st column is the default
    :param comment: str; comments; '# comments' is added to the beginning of the file
    :return: None
    """

    if isinstance(series_or_dataframe, Series):
        s = series_or_dataframe.sort_values(ascending=False)
    else:
        df = series_or_dataframe.copy()

        if gene_column:
            df = df.set_index(gene_column)

        if not score_column:
            score_column = df.columns[0]

        s = df.ix[:, score_column]
        s = s.sort_values(ascending=False)

    if not filepath.endswith('.rnk'):
        filepath += '.rnk'

    with open(filepath, 'w') as f:
        if comment:
            f.writelines('# {}\n'.format(comment))
        s.to_csv(f, sep='\t', header=False)


# ======================================================================================================================
# GEO functions
# ======================================================================================================================
def load_geo_annotations(filepath, annotation_names=('!Sample_geo_accession', '!Sample_characteristics_ch1')):
    """
    Parse rows of GEO file.
    If the 1st column (annotation name) matches any of the annotation names in annotation_names,
    split the row by '\t' and save the split list as a row in the dataframe to returned.
    :param filepath: str; filepath to a GEO file (.txt or .gz)
    :param annotation_names: iterable; list of str
    :return DataFrame; (n_matched_annotation_names, n_tabs (n_samples))
    """

    df = DataFrame()

    # Open GEO file
    if filepath.endswith('.gz'):
        f = gzip.open(filepath)
    else:
        f = open(filepath)

    # Parse rows
    for line in f.readlines():

        if isinstance(line, bytes):
            line = line.decode()

        if any([line.startswith(a_n) for a_n in annotation_names]):  # Annotation name matches

            # Parse row
            split_line = line.strip().split('\t')
            name = split_line[0]
            annotation = [s[1:-1] for s in split_line[1:]]

            # Avoid same names
            i = 2
            formatter = '_${}'
            while name in df.columns:
                name = name.split(formatter.format(i - 1))[0] + formatter.format(i)
                i += 1

            # Make Series
            s = Series(annotation, name=name)

            # Concatenate to DataFrame
            df = concat([df, s], axis=1)

    # Close GEO file
    f.close()

    df = df.T

    if '!Sample_geo_accession' in annotation_names:  # Set columns indices to be sample accessions
        df.columns = df.ix['!Sample_geo_accession', :]
        df.columns.name = 'GEO Sample Accession'
        df.drop('!Sample_geo_accession', inplace=True)

    return df


# ======================================================================================================================
# .fpkm_tracking functions
# ======================================================================================================================
def read_fpkm_tracking(filepath, signature=None):
    """

    :param filepath: filepath to cufflinks output
    :param signature:
    :return:
    """

    print('Reading {} ...'.format(filepath))
    fpkm_tracking = read_csv(filepath, sep='\t', index_col=4)
    print('\t{}'.format(fpkm_tracking.shape))

    print('Keeping only \'FPKM\' column ...')
    fpkm = fpkm_tracking[['FPKM']]
    if signature:
        fpkm.columns = ['{} FPKM'.format(signature)]
    print('\t{}'.format(fpkm.shape))

    print('Dropping rows where gene_short_name is \'-\' (transcripts without gene_short_name) ...')
    fpkm = fpkm.ix[fpkm.index != '-', :]
    print('\t{}'.format(fpkm.shape))

    return fpkm.groupby(level=0).mean()


# ======================================================================================================================
# .g*f functions
# ======================================================================================================================
def read_gff3(feature_filename, sources, types):
    """
    Parse feature_filename and return:
    1) a dictionary of features keyed by IDs and 2) a dictionary mapping features to IDs.
    :param feature_filename:
    :param sources:
    :param types:
    :return:
    """

    sources = set(sources)
    types = set(types)

    features, feature_to_id = {}, {}

    for line_num, line in enumerate(feature_filename):
        # Parse non-headers
        if line[0] != '#':
            split_line = line.split('\t')
            assert len(split_line) >= 7, 'Column error on line {}: {}'.format(line_num, split_line)

            source, feature_type = split_line[1], split_line[2]

            if source in sources and feature_type in types:
                contig = split_line[0]
                start = int(split_line[3])
                end = int(split_line[4])
                if split_line[6] == '+':
                    strand = 1
                elif split_line[6] == '-':
                    strand = -1
                else:
                    strand = 0

                fields = dict(field_value_pair.split('=') for field_value_pair in split_line[8].split(';'))
                version = float(fields['version'])
                gene_name = fields['Name']
                ensembl_id = fields['gene_id']

                # Make sure not duplicated ensembl IDs
                assert ensembl_id not in features

                features[ensembl_id] = {'contig': contig,
                                        'start': start - 1,  # Convert 1-based to 0-based
                                        'end': end,
                                        # Convert 1-based to 0-based and account for fully-closed interval
                                        # of GFF (they cancel out)
                                        'strand': strand,
                                        'version': version,
                                        'gene_name': gene_name,
                                        'ensembl_id': ensembl_id}

                # Save a new feature or an existing feature with an an updated version
                if gene_name not in feature_to_id or version > features[ensembl_id]['version']:
                    feature_to_id[gene_name] = ensembl_id

    return features, feature_to_id


def read_gtf(feature_filename, sources, types):
    """
    Parse feature_filename and return:
    1) a dictionary of features keyed by IDs and 2) a dictionary mapping features to IDs.
    :param feature_filename:
    :param sources:
    :param types:
    :return:
    """

    types = set(types)

    features, feature_to_id = {}, {}

    for line_num, line in enumerate(feature_filename):
        if line[0] != '#':
            split_line = line.strip().split('\t')
            assert len(split_line) >= 7, 'Column error on line {}: {}'.format(line_num, split_line)

            feature_type = split_line[2]

            if feature_type in types:
                fields = dict([field_value_pair.strip().split(' ') for field_value_pair in split_line[8].split(';')
                               if len(field_value_pair) > 0])

                if fields['gene_source'].strip('"') in sources:
                    chrom = split_line[0]
                    start = int(split_line[3])
                    end = int(split_line[4])
                    if split_line[6] == '+':
                        strand = 1
                    elif split_line[6] == '-':
                        strand = -1
                    else:
                        strand = 0

                    gene_name = fields['gene_name'].strip('"')
                    ensembl_id = fields['gene_id'].strip('"')

                    # Make sure not duplicated ensembl IDs
                    assert ensembl_id not in features

                    features[ensembl_id] = {'contig': chrom,
                                            'start': start,
                                            'end': end,
                                            'strand': strand,
                                            'gene_name': gene_name,
                                            'ensembl_id': ensembl_id}

                    if gene_name not in feature_to_id:
                        feature_to_id[gene_name] = ensembl_id

    return features, feature_to_id
