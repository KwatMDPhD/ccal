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

from os import environ, listdir, mkdir
from os.path import abspath, isdir, isfile, islink, join, split
from sys import platform

from Bio import bgzf
from pandas import DataFrame, Series, concat, read_csv

from .str_ import remove_nested_quotes, split_ignoring_inside_quotes


# ==============================================================================
# General functions
# ==============================================================================
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
    return sorted(dirs)


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
    while not (isdir(prefix) or isfile(prefix) or
               islink(prefix)):  # prefix isn't compress, directory, or link
        missing_directories.append(prefix)

        # Check prefix's prefix next
        prefix, suffix = split(prefix)

    # Make missing directories
    for d in reversed(missing_directories):
        mkdir(d)
        print('Created directory {}/.'.format(d))


def mark_filename(filepath, mark):
    """
    Convert fname.extension to fname.mark.extension.
    :param filepath: str;
    :param mark: str;
    :return: str;
    """

    prefix, extension = split_file_extension(filepath)
    if mark.startswith('.'):
        mark = mark[1:]
    new_filepath = '{}.{}.{}'.format(prefix, mark, extension)
    print('{} ==> {}.'.format(filepath, new_filepath))
    return new_filepath


def split_file_extension(filepath):
    """
    Get filepath without compress suffix and the suffix from filepath; get foo and txt from foo.txt
    :param filepath: str; filepath
    :return: str and str; filepath without compress suffix and the suffix
    """

    filepath = abspath(filepath)
    i = filepath.rindex('.')
    return filepath[:i], filepath[i + 1:]


def write_dict(dict_, filepath, key_name, value_name):
    """
    Write dictionary as 2 column table.
    :param dict_: dict;
    :param filepath: str;
    :param key_name: str;
    :param value_name: str;
    :return: None
    """

    s = Series(dict_)
    s.index.name = key_name
    s.name = value_name
    s.to_csv(filepath, sep='\t')


# ==============================================================================
# GCT functions
# ==============================================================================
def load_gct(matrix):
    """
    If matrix is a DataFrame, return as is; if a filepath (.gct), read matrix from it as DaraFrame; else, raise Error.
    :param matrix: DataFrame or str; filepath to a .gct or DataFrame
    :return: DataFrame; matrix
    """

    if isinstance(matrix, str):  # Read .gct from a filepath
        matrix = read_gct(matrix)

    elif not isinstance(matrix,
                        DataFrame):  # .gct is not a filepath or DataFrame
        raise ValueError(
            'Matrix must be either a DataFrame or a path to a .gct compress.')

    return matrix


def read_gct(filepath,
             fill_na=None,
             drop_description=True,
             row_name=None,
             column_name=None):
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
            raise ValueError(
                'Column 1 has more than 1 extra space around \'Name\'. Please strip it.'
            )
    df.set_index('Name', inplace=True)

    # Check if the 2nd column is 'Description'; is so drop it as necessary
    if c2 != 'Description':
        if c2.strip() != 'Description':
            raise ValueError('Column 2 != \'Description\'')
        else:
            raise ValueError(
                'Column 2 has more than 1 extra space around \'Description\'. Please strip it.'
            )
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


# ==============================================================================
# .data_table.txt functions
# ==============================================================================
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
    for data_name, (data_type, emphasis,
                    filepath) in data_table.iterrows():  # For each data
        if isinstance(indices, dict) and data_name not in indices:
            print('Skipped loading {} because indices were not specified.'.
                  format(data_name))
            continue

        print('Making data bundle for {} ...'.format(data_name))
        data_bundle[data_name] = {}
        df = read_gct(join(filepath))
        print('\tLoaded {}.'.format(filepath))
        if isinstance(indices, dict):  # If indices is given
            if data_name in indices:  # Keep specific indices
                index = indices[data_name]['index']
                df = df.ix[index, :]

                # Save the original index names
                data_bundle[data_name]['original_index'] = index

                if 'alias' in indices[
                        data_name]:  # Rename these specific indices
                    df.index = indices[data_name]['alias']

                print('\tSelected rows: {}.'.format(df.index.tolist()))

        data_bundle[data_name]['dataframe'] = df
        data_bundle[data_name]['data_type'] = data_type
        data_bundle[data_name]['emphasis'] = emphasis

    return data_bundle


def read_data_table(filepath):
    """
    Read .data_table.
    :param filepath: str; compress path to a .data_table
    :return: DataFrame
    """

    return read_csv(filepath, sep='\t', index_col=0)


def write_data_table(
        data, filepath,
        columns=('Data Name', 'Data Type', 'Emphasis', 'Filepath')):
    """
    Write <data> to filepath.
    :param data: iterable of tuples;
        [('mutation', 'binary', 'high', 'filepath),
        ('gene_expression', 'continuous', 'high', 'filepath),
        ('drug_sensitivity', 'continuous', 'low', 'filepath),
        ...]
    :param filepath: str; compress path to a .data_table (.data_table suffix will be automatically added if not present)
    :param columns: iterable;
    :return: None
    """

    df = DataFrame(data)
    df.columns = columns
    df.set_index(columns[0], inplace=True)
    if not filepath.endswith('data_table.txt'):
        filepath = '{}.data_table.txt'.format(filepath)
    df.to_csv(filepath, sep='\t')


# ==============================================================================
# GMT functions
# ==============================================================================
def read_gmts(filepaths,
              gene_sets=(),
              drop_description=True,
              save_clean=False,
              collapse=False):
    """
    Read 1 or more GMTs.
    :param filepaths: str; filepath to a .gmt compress
    :param gene_sets: iterable: list of gene set names to keep
    :param drop_description: bool; drop Description column (2nd column) or not
    :param save_clean: bool; Save as .gmt (cleaned version) or not
    :param collapse: bool; collapse into a list of unique genes or not
    :return: DataFrame or list; (n_gene_sets, size of the largest gene set) or (n_unique genes)
    """

    gmts = []
    if isinstance(filepaths, str):
        filepaths = [filepaths]
    for fp in filepaths:
        gmt = read_gmt(
            fp,
            gene_sets=gene_sets,
            drop_description=drop_description,
            save_clean=save_clean)
        gmts.append(gmt)
    gmt = concat(gmts)
    gmt.dropna(axis=1, how='all', inplace=True)
    gmt.sort_index(inplace=True)

    if 'Description' in gmt.columns:
        gmt.columns = ['Description'] + [
            'Gene {}'.format(i) for i in range(1, gmt.shape[1])
        ]
    else:
        gmt.columns = ['Gene {}'.format(i) for i in range(1, gmt.shape[1] + 1)]

    if collapse:
        return sorted(set(gmt.unstack().dropna()))
    else:
        return gmt


def read_gmt(filepath,
             gene_sets=(),
             drop_description=True,
             save_clean=False,
             collapse=False):
    """
    Read GMT.
    :param filepath: str; filepath to a .gmt compress
    :param gene_sets: iterable: list of gene set names to keep
    :param drop_description: bool; drop Description column (2nd column) or not
    :param save_clean: bool; Save as .gmt (cleaned version) or not
    :param collapse: bool; collapse into a list of unique genes or not
    :return: DataFrame or list; (n_gene_sets, size of the largest gene set) or (n_unique genes)
    """

    # Parse
    rows = []
    with open(filepath) as f:
        for line in f.readlines():
            line_split = line.strip().split('\t')
            # Sort genes and add as a GMT gene set (row)
            rows.append(line_split[:2] + sorted(
                [g for g in line_split[2:] if g]))

    # Make a DataFrame
    gmt = DataFrame(rows)

    # Set index
    gmt.set_index(0, inplace=True)
    gmt.index.name = 'Gene Set'
    gmt.sort_index(inplace=True)
    gmt.columns = ['Description'
                   ] + ['Gene {}'.format(i) for i in range(1, gmt.shape[1])]

    if save_clean:  # Save the cleaned version
        gmt.to_csv(filepath, sep='\t', header=False)

    if drop_description or collapse:
        gmt.drop('Description', axis=1, inplace=True)

    # Keep specific gene sets
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


def write_gmt(gmt, filepath):
    """
    Write a GMT DataFrame to filepath.gmt.
    :param gmt: DataFrame;
    :param filepath: str; filepath to a GMT compress
    :return: DataFrame; GMT
    """

    if 'Description' not in gmt.columns:
        gmt['Description'] = gmt.index
        gmt = gmt.reindex(
            columns=gmt.columns[-1:].tolist() + gmt.columns[:-1].tolist())

    if not filepath.endswith('gmt'):
        filepath += '.gmt'

    gmt.to_csv(filepath, header=None, sep='\t')

    return gmt


# ==============================================================================
# RNK functions
# ==============================================================================
def write_rnk(series_or_dataframe,
              filepath,
              gene_column=None,
              score_column=None,
              comment=None):
    """
    Write .rnk.
    :param series_or_dataframe: Series or DataFrame;
    :param filepath: str;
    :param gene_column: str; column name; dataframe index is the default
    :param score_column: str; column name; 1st column is the default
    :param comment: str; comments; '# comments' is added to the beginning of the compress
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


# ==============================================================================
# FPKM_TRACKING functions
# ==============================================================================
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

    print(
        'Dropping rows where gene_short_name is \'-\' (transcripts without gene_short_name) ...'
    )
    fpkm = fpkm.ix[fpkm.index != '-', :]
    print('\t{}'.format(fpkm.shape))

    return fpkm.groupby(level=0).mean()


# ==============================================================================
# VCF functions
# ==============================================================================
def read_vcf(filepath):
    """
    Read a VCF.
    :param filepath: str;
    :return: dict;
    """

    vcf = {
        'meta_information': {
            'INFO': {},
            'FILTER': {},
            'FORMAT': {},
            'reference': {},
        },
        'header': [],
        'samples': [],
        'data': None,
    }

    # Open VCF
    try:
        f = open(filepath)
        f.readline()
        f.seek(0)
        bgzipped = False
    except UnicodeDecodeError:
        f = bgzf.open(filepath)
        bgzipped = True

    for row in f:

        if bgzipped:
            row = row.decode()
        row = row.strip()

        if row.startswith('##'):  # Meta-information
            # Remove '##' prefix
            row = row[2:]

            # Find the 1st '='
            ei = row.find('=')

            # Get field name and field line
            fn, fl = row[:ei], row[ei + 1:]

            if fl.startswith('<') and fl.endswith('>'):
                # Strip '<' and '>'
                fl = fl[1:-1]

                # Split field line
                fl_split = split_ignoring_inside_quotes(fl, ',')

                # Get ID
                id_ = fl_split[0].split('=')[1]

                # Parse field line
                fd_v = {}
                for s in fl_split[1:]:
                    ei = s.find('=')
                    k, v = s[:ei], s[ei + 1:]
                    fd_v[k] = remove_nested_quotes(v)

                # Save
                if fn in vcf['meta_information']:
                    if id_ in vcf['meta_information'][fn]:
                        raise ValueError('Duplicated ID {}.'.format(id_))
                    else:
                        vcf['meta_information'][fn][id_] = fd_v
                else:
                    vcf['meta_information'][fn] = {id_: fd_v}
            else:
                print('Didn\'t parse: {}.'.format(fl))

        elif row.startswith('#CHROM'):  # Header
            # Remove '#' prefix
            row = row[1:]

            # Get header line number
            vcf['header'] = row.split('\t')
            vcf['samples'] = vcf['header'][9:]
        else:
            break

    # Close VCF
    f.close()

    # Read data
    vcf['data'] = read_csv(
        filepath, sep='\t', comment='#', header=None, names=vcf['header'])

    return vcf


# ==============================================================================
# G*F functions
# ==============================================================================
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
            assert len(split_line) >= 7, 'Column error on line {}: {}'.format(
                line_num, split_line)

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

                fields = dict(
                    field_value_pair.split('=')
                    for field_value_pair in split_line[8].split(';'))
                version = float(fields['version'])
                gene_name = fields['Name']
                ensembl_id = fields['gene_id']

                # Make sure not duplicated ensembl IDs
                assert ensembl_id not in features

                features[ensembl_id] = {
                    'contig': contig,
                    'start': start - 1,  # Convert 1-based to 0-based
                    'end': end,
                    # Convert 1-based to 0-based and account for fully-closed interval
                    # of GFF (they cancel out)
                    'strand': strand,
                    'version': version,
                    'gene_name': gene_name,
                    'ensembl_id': ensembl_id
                }

                # Save a new feature or an existing feature with an an updated version
                if gene_name not in feature_to_id or version > features[
                        ensembl_id]['version']:
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
            assert len(split_line) >= 7, 'Column error on line {}: {}'.format(
                line_num, split_line)

            feature_type = split_line[2]

            if feature_type in types:
                fields = dict([
                    field_value_pair.strip().split(' ')
                    for field_value_pair in split_line[8].split(';')
                    if len(field_value_pair) > 0
                ])

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

                    features[ensembl_id] = {
                        'contig': chrom,
                        'start': start,
                        'end': end,
                        'strand': strand,
                        'gene_name': gene_name,
                        'ensembl_id': ensembl_id
                    }

                    if gene_name not in feature_to_id:
                        feature_to_id[gene_name] = ensembl_id

    return features, feature_to_id
