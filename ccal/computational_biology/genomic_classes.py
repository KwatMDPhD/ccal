"""
Computational Cancer Analysis Library

Author:
    Huwate (Kwat) Yeerna (Medetgul-Ernar)
        kwat.medetgul.ernar@gmail.com
        Computational Cancer Analysis Laboratory, UCSD Cancer Center
"""

import collections
import datetime
import gzip
import os
import pickle
import random
import re

import intervaltree
import tables
from Bio import bgzf

from genomics.genomic_utilities import convert_gzipped_to_bgzipped, reverse_complement, parse_gff3, parse_gtf, \
    convert_chromosome, get_alleles
from utilities import verbose_print, try_finding_gzipped_file, WHITESPACE


# Sequence
SEQUENCE_FILENAME = os.environ['G_DEVICE_SEQUENCE']

# Feature
TRANSCRIPT_FILENAME = os.environ['G_DEVICE_TRANSCRIPT']
FEATURE_FILENAME = os.environ['G_DEVICE_FEATURE']
GENE_FILENAME = os.environ['G_DEVICE_GENE']
FEATURE_SOURCES = ['ensembl_havana', 'insdc']
FEATURE_TYPES = ['gene', 'mt_gene']

# Supplemental variant
SNP_FILENAME = os.environ['G_SNP']
KNOWN_VARIANT_FILENAME = os.environ['G_DEVICE_KNOWN_VARIANT']
DESCRIBED_VARIANT_FILENAME = os.environ['G_DEVICE_DESCRIBED_VARIANT']
SUPPLEMENTAL_FILE_FIELD_MODEL = (('rs_id', lambda s: s.encode()),
                                 ('contig', lambda s: s.encode()),
                                 ('start', int),
                                 ('ref', lambda s: s.encode()),
                                 ('alt', lambda s: s.encode()),
                                 ('gene', lambda s: s.encode()),
                                 ('af', lambda s: s.encode()),
                                 ('clinvar_allele', lambda s: s.encode()),
                                 ('clin_sig', lambda s: s.encode()),
                                 ('phenotype', lambda s: s.encode()),
                                 ('description', lambda s: s.encode()))
SUPPLEMENTAL_VARIANT_COMPRESSOR = 'blosc'
SUPPLEMENTAL_VARIANT_COMPRESSION_LEVEL = 9

# Client variant
VARIANT_FILENAME = os.environ['G_DEVICE_VARIANT']
FREE_BAYES = {'primary_key': {'rs_id': (2, str)},
              'core_fields':
                  {'contig': (0, str),
                   'start': (1, int),
                   'ref': (3, str),
                   'alt': (4, str),
                   'qual': (5, float)
                   },
              'info_position': 7,
              'secondary_field_keys': 8,
              'secondary_field_values': 9,
              }
VARIANT_INDEX_FILENAME = os.environ['G_DEVICE_VARIANT_INDEX']
MINIMUM_VARIANT_QUALITY = 30
VARIANT_HDF5_FILENAME = os.environ['G_DEVICE_VARIANT_HDF5']
VARIANT_HDF5_COMPRESSOR = 'blosc'
VARIANT_HDF5_COMPRESSION_LEVEL = 1


# ======================================================================================================================
# Genomics Interface
# ======================================================================================================================
class GenomicsInterface:
    def __init__(self,
                 sequence_file=SEQUENCE_FILENAME,
                 feature_filename=TRANSCRIPT_FILENAME,
                 vcf_filename=VARIANT_FILENAME,
                 feature_pickle_filename=FEATURE_FILENAME,
                 variant_table_filename=VARIANT_HDF5_FILENAME,
                 variant_rs_id_index_filename=VARIANT_INDEX_FILENAME,
                 force_rebuild=False,
                 force_rebuild_vcf=False):
        """
        Implements various high-level methods for working with and querying a linked set of human genomic data
        including reference sequence, genomic features and sequence variants
        """
        verbose_print('============================ Creating Genomic Data Interface... =============================')
        verbose_print('\nGenome ...')
        self.genome = Genome(sequence_file=sequence_file,
                             feature_filename=feature_filename,
                             feature_pickle_filename=feature_pickle_filename,
                             force_rebuild=force_rebuild)
        verbose_print('\nSupplemental Gene Info ...')
        self.supplemental_gene_info = IndexedTsv(bgzipped_filename=GENE_FILENAME,
                                                 index_filename='',
                                                 index_field='SYMBOL',
                                                 sep='\t',
                                                 force_reindex=force_rebuild)
        verbose_print('\nKnown Supplemental Variant Info ...')
        self.known_variant_info = SupplementalVariantInfo(table_description=KnownVariant,
                                                          gzipped_tsv_filename=SNP_FILENAME,
                                                          supplemental_variant_table_filename=KNOWN_VARIANT_FILENAME,
                                                          force_rebuild=force_rebuild)
        verbose_print('\nDescribed Supplemental Variant Info ...')
        self.described_variant_info = SupplementalVariantInfo(table_description=DescribedVariant,
                                                              gzipped_tsv_filename=SNP_FILENAME,
                                                              supplemental_variant_table_filename=DESCRIBED_VARIANT_FILENAME,
                                                              force_rebuild=force_rebuild)
        verbose_print('\nVCF ...')
        self.variants = Variants(vcf_filename=vcf_filename,
                                 variant_table_filename=variant_table_filename,
                                 variant_rs_id_index_filename=variant_rs_id_index_filename,
                                 force_rebuild=force_rebuild_vcf)
        verbose_print('============================ Created Genomic Data Interface. =============================')

    def __del__(self):
        """
        Destructor. Close open files by calling destructor on specific instance fields.
        """
        verbose_print('Deleting DataInterface, closing files ...')
        if self.variants.variant_file:
            self.variants.variant_file.close()
        del self.variants
        del self.known_variant_info
        del self.described_variant_info

    @property
    def contig_names(self):
        """
        Return a list of valid contig names for this genome.
        """
        return self.genome.contig_names

    @property
    def contig_lengths(self):
        """
        Return a dictionary of contig lengths keyed by contig name.
        """
        return self.genome.contig_lengths

    def get_reference_sequence_by_location(self, contig_name, start, end, strand=1):
        """
        Return a list of characters representing the DNA sequence in the reference genome for the genomic region
        specified by `contig_name`, `start`, and `end`.
        """
        verbose_print('Exploring {}:{}-{} ...'.format(contig_name, start, end))
        assert strand in {-1, 1}
        return self.genome.get_dna_sequence(contig=contig_name, start=int(start), end=int(end), strand=strand)

    def get_variant_by_id(self, rs_id, restrict_to_client=False):
        """
        Return a dictionary of attributes for a single sequence variant identified by `rs_id`.
        if :param:`restrict_to_client` is `True`, only return information if the client has this
        variant, otherwise return whatetver is known about the variant.
        """
        verbose_print('Exploring variant {} ...'.format(rs_id))
        client_info = self.variants.get_variant_by_rs_id(rs_id=rs_id, allow_snps=True, allow_indels=True)
        if client_info:
            verbose_print('Variant {} found in client.'.format(rs_id))
        else:
            client_info = {}
            verbose_print('Variant {} not found in client.'.format(rs_id))

        if client_info or not restrict_to_client:
            known_info = self.known_variant_info.get_supplemental_variant_info_by_rs_id(rs_id)
            if known_info:
                verbose_print('Variant {} is known and getting its supplementary information ...'.format(rs_id))
                if client_info:
                    # Prioritize client values for shared fields
                    known_info.update(client_info)
                    client_info = known_info
                else:
                    client_info.update(known_info)
                    client_info['alleles'] = get_alleles('0/0',
                                                         client_info['ref'],
                                                         client_info['alt'])
            else:
                verbose_print('Variant is not known.')

            described_info = self.described_variant_info.get_supplemental_variant_info_by_rs_id(rs_id)
            if described_info:
                verbose_print('Variant {} is described and getting its information ...'.format(rs_id))
                client_info.update(described_info)
            else:
                verbose_print('Variant is not described.')

        return client_info

    def get_variants_by_gene_name(self, gene_name):
        """
        Return a list of variant dictionaries, sorted by genomic position, that fall within (or overlap) the boundaries
        of the gene specified by `gene_name`

        :param gene_name:
        :return:
        """
        verbose_print('Exploring gene {} ...'.format(gene_name))
        gene_info = self.genome.get_gene_by_name(gene_name=gene_name)
        if gene_info:
            variant_list = self.variants.get_variants_by_location(contig=gene_info['contig'], start=gene_info['start'],
                                                                  end=gene_info['end'],
                                                                  minimum_quality=MINIMUM_VARIANT_QUALITY,
                                                                  allow_snps=True,
                                                                  allow_indels=False,
                                                                  convert_to_dict=True)
            return variant_list
        else:
            return None

    def get_variants_by_location(self, contig_name, start=0, end=0, convert_to_dict=False):
        """
        Return a tuople consisting of a list of variant tuples, sorted by genomic position,
        that overlap the genomic region specified by `contig_name`, `start`, `end`,
        and a dictionary mapping field name to tuple positions.
        Will not returns supplemental information.
        """
        return self.variants.get_variants_by_location(contig=contig_name, start=start, end=end,
                                                      minimum_quality=MINIMUM_VARIANT_QUALITY,
                                                      allow_snps=True,
                                                      allow_indels=False,
                                                      convert_to_dict=convert_to_dict)

    def get_features_by_location(self, contig_name, start, end):
        """
        Return a list of genomic features (e.g. genes), sorted by genomic position, that overlap the genomic region
         specified by `contig_name`, `start`, `end`
        """
        return self.genome.get_genes_by_location(contig=contig_name, start=start, end=end)

    def get_gene_by_ensemblid(self, ensembl_id):
        """
        Return a dictionary of attributes for the gene corresponding to `ensembl_id`
        """
        return self.genome.get_gene_by_ensembl_id(ensembl_id=ensembl_id)

    def get_gene_by_name(self, gene_name):
        """
        Return a dictionary of attributes for the gene corresponding to `gene_name`. This will return
        additional data from the supplemental gene file.
        """
        result = self.genome.get_gene_by_name(gene_name=gene_name)
        if result is None:
            return None
        base_info = result
        if base_info:
            verbose_print('Gene {} found in genome.'.format(gene_name))
            if gene_name in self.supplemental_gene_info:
                verbose_print('Gene {} has supplemental info'.format(gene_name))
                base_info.update(self.supplemental_gene_info[gene_name])
            else:
                verbose_print('No supplemental info for gene {}'.format(gene_name))
        else:
            verbose_print('Gene {} not found in genome.'.format(gene_name))
            raise GeneNotFound(requested_gene=gene_name, message='Gene {} not found in genome.'.format(gene_name))
        return base_info

    def benchmark_random_access(self, sample_size=297):
        """
        Return the time taken to query <sample_size> randomly-chosen client variants.
        """
        start_time = datetime.datetime.now()
        query_variants = random.sample(self.variants.rs_id_index.keys(), sample_size)

        for rs_id in query_variants:
            self.get_variant_by_id(rs_id)
        elapsed_time = datetime.datetime.now() - start_time
        verbose_print('Done in {}'.format(elapsed_time))
        return elapsed_time

    def benchmark_contiguous_access(self, gene_name='EGFR'):
        """
        Return the time taken to query all the client variants in <gene_name>.
        """
        start_time = datetime.datetime.now()
        for v in self.get_variants_by_gene_name(gene_name):
            self.get_variant_by_id(v['rs_id'])
        elapsed_time = datetime.datetime.now() - start_time
        return elapsed_time


# ======================================================================================================================
# Genomics Data Structures
# ======================================================================================================================
class GenomeSequence:
    """
    Wrapper around a single FASTA file that has been compressed with block gzip (bgzip), a utility
    that accompanies samtools. Block gzip allows fast random access to 64 kb segments of the compressed
    file. This class implements the loading, saving and generation of an index to permit identification
    of the blocks containing the sequence of any arbitrary region of the genome. It also implements
    methods to decompress and return that sequence.

    I used the bgzf module in order to determine determine the file intervals of the compressed blocks. Then I go
    through the file and find the contig intervals in terms of the file offsets. Next I match compressed blocks to
    contigs based on their overlapping file intervals. Then I convert the block file coordinates to sequence
    coordinates using the offset conversion code (need to subtract 1 for the newline character on each line).

    Now we have an interval tree for each contig giving the start position of each compressed block that contains part
    of the sequence, and what part of the sequence it contains.

    So to query, we use the interval tree for the requested contig to get the coordinates of the compressed block
    containing the sequence start position and compute the offset of the start position within that block. Now we can
    use the bgzf code to seek directly to that spot and then read until we have the requested sequence.

    This solution is:
    * Fast (blocks can be found in O(log N), and minimum decompression operation is now only 64 kb)
    * Disk space efficient (sequence is stored compressed on disk).
    * Memory efficient (all the unrequested sequence stays on disk, and since the index consists only of one entry per
     block (~48K of them), it has a very small memory footprint).
    """

    def __init__(self, bgzipped_fasta_filename, force_rebuild=False):
        """
        Create a new object with no index
        """
        self.bgzipped_fasta_filename = bgzipped_fasta_filename
        self._contig_lengths = {}
        self._index = {}
        self.line_length_file_distance = None
        self.line_length_text_distance = None

        contig_length_filename = self.bgzipped_fasta_filename + '_contig_length.txt'
        index_filename = self.bgzipped_fasta_filename + '_index.gz'

        if not force_rebuild:
            try:
                self.load_contig_lengths(contig_length_filename)
            except (IOError, OSError, ValueError, EOFError):
                force_rebuild = True
        if not force_rebuild:
            try:
                self.load_index_from_text(index_filename)
            except (IOError, OSError, ValueError, EOFError):
                force_rebuild = True

        if force_rebuild:
            self.generate_index()
            self.save_contig_lengths(contig_length_filename)
            self.save_index_to_text(index_filename)

    @property
    def contig_lengths(self):
        """
        Read only property returning a dictionary containing  the length (in base pairs) of each contig in the genome
        """
        return self._contig_lengths.copy()

    @property
    def contig_names(self):
        """
        Read only property returning a list of the names of each contig in the genome.
        """
        return sorted(self._contig_lengths.keys())

    def _text_distance_to_file_distance(self, offset_sequence):
        """
        Converts a distance from the start of a genomic sequence (or other string) into a distance from the start
        of a multi-line file (as in a FASTA).
        """
        num_lines = int(offset_sequence / self.line_length_text_distance)
        partial_line_length = offset_sequence % self.line_length_text_distance
        return num_lines * self.line_length_file_distance + partial_line_length

    def _file_distance_to_text_distance(self, offset_file_distance, sequence_start_file_distance):
        """
        Convert a distance from the start of a multi-line file string (as in a FASTA) into a genomic sequence
        (or other string) into a distance from the start of a genomic sequence (or other string).
        """
        file_distance = offset_file_distance - sequence_start_file_distance
        num_lines = int(file_distance / self.line_length_file_distance) - 1
        partial_line_length = file_distance % self.line_length_file_distance
        return num_lines * self.line_length_text_distance + partial_line_length

    def _get_blocks(self):
        """
        Return a tuple for each compressed block in the bzgipped FASTA file, consisting of
        (binary_start, file_end, file_block_start)
        """
        start_time = datetime.datetime.now()
        verbose_print('\tFinding block boundaries ...')

        def populate_blocks():
            with open(self.bgzipped_fasta_filename, 'rb') as fasta_file_for_blocks:
                # Store uncompressed data start, uncompressed data end, compressed block start as a tuple
                blocks = [(b[2], b[2] + b[3], b[0]) for b in bgzf.BgzfBlocks(fasta_file_for_blocks)]
                verbose_print('\t\tFound {} blocks in {}'.format(len(blocks), datetime.datetime.now() - start_time))
            return blocks

        try:
            blocks = populate_blocks()
        except ValueError:
            verbose_print('This does not appear to be a valid block-gzipped file. Converting to bgzipped format ...')
            convert_start_time = datetime.datetime.now()
            convert_gzipped_to_bgzipped(self.bgzipped_fasta_filename)
            verbose_print('\tDone in {}.'.format(datetime.datetime.now() - convert_start_time))
            blocks = populate_blocks()

        return blocks[:-1]  # Omit the last, empty block

    def _compute_file_line_length(self):
        """
        Inspect the file on disk and compute the binary length of the first non-header line.
        """
        with gzip.open(self.bgzipped_fasta_filename, 'rb') as fasta_file_binary:
            first_line = fasta_file_binary.readline()
            assert first_line.startswith(b'>')
            file_line_length = len(fasta_file_binary.readline())
        return file_line_length

    def _compute_text_line_length(self):
        """
        Inspect the file on disk and compute the _text length of the first non-header line.
        """
        with gzip.open(self.bgzipped_fasta_filename, 'rt') as fasta_file_text:
            first_line = fasta_file_text.readline()
            assert first_line.startswith('>')
            text_line_length = len(fasta_file_text.readline().strip())
        return text_line_length

    def _get_contig_intervals_file_distance(self):
        """
        Determine the start and end locations of each contig sequence (not including headers) in the file.
        """
        start_time = datetime.datetime.now()
        contig_intervals_file_distance = {}

        verbose_print('\tFinding contig locations ...')
        previous_sequence = None
        previous_start = 0

        line = None
        with gzip.open(self.bgzipped_fasta_filename, 'rb') as fasta_file:
            for line_num, line in enumerate(fasta_file):
                if line_num % 10000000 == 0:
                    verbose_print('\t\tprocessing line {:>10} ...'.format(line_num + 1))

                if line.startswith(b'>'):
                    sequence_name = re.split(WHITESPACE, line[1:].decode())[0]

                    if previous_sequence:
                        contig_intervals_file_distance[previous_sequence] = (
                            previous_start, fasta_file.tell() - len(line))

                    previous_start = fasta_file.tell()
                    previous_sequence = sequence_name
            contig_intervals_file_distance[previous_sequence] = (previous_start, fasta_file.tell() - len(line))

        verbose_print('\t\tFound {} sequences in {}.'.format(len(contig_intervals_file_distance),
                                                             datetime.datetime.now() - start_time))
        return contig_intervals_file_distance

    @staticmethod
    def _assign_blocks_to_contigs(contig_intervals_file_distance, block_interval_tree):
        """
        For each contig, create an interval tree that stores the sequence interval stored in each block
        (for all blocks that contain part of the contig), as well as the offset of the start of that block.
        :param contig_intervals_file_distance: A dictionary of intervals, keyed by contig name,
            storing the locations in the file spanned by each contig.
        :param block_interval_tree:  An interval tree storing the start and end locations in the uncompressed
            file spanned by each compressed block, as well as the offset of the block start.
        :return: Return a dictionary of such interval trees keyed by contig name.
        """
        start_time = datetime.datetime.now()
        verbose_print('\tAssigning compressed blocks to sequence contigs ...')

        sequence_blocks = {}

        for contig in sorted(contig_intervals_file_distance):

            if contig not in sequence_blocks:
                sequence_blocks[contig] = intervaltree.IntervalTree()

            for block_interval in block_interval_tree.search(*contig_intervals_file_distance[contig]):
                block_start_text_distance = block_interval.begin - contig_intervals_file_distance[contig][0]
                block_end_text_distance = block_interval.end - contig_intervals_file_distance[contig][0]
                sequence_blocks[contig].addi(block_start_text_distance, block_end_text_distance,
                                             block_interval.data)

        verbose_print('\t\tDone in {}.'.format(datetime.datetime.now() - start_time))
        return sequence_blocks

    def _compute_contig_lengths(self, contig_intervals_file_distance):
        verbose_print('\tComputing contig lengths ...')
        self._contig_lengths = {}

        for contig in sorted(contig_intervals_file_distance):
            self._contig_lengths[contig] = self._file_distance_to_text_distance(
                offset_file_distance=contig_intervals_file_distance[contig][1],
                sequence_start_file_distance=contig_intervals_file_distance[contig][0]) - \
                                           self._file_distance_to_text_distance(
                                               offset_file_distance=contig_intervals_file_distance[contig][0],
                                               sequence_start_file_distance=contig_intervals_file_distance[contig][
                                                   0]) - 1
        verbose_print('\t\tDone.')

    def generate_index(self):
        """
        Generate an index for the FASTA file and store it in memory.
        """
        overall_start_time = datetime.datetime.now()

        verbose_print('Generating index for sequence file {} ...'.format(self.bgzipped_fasta_filename))

        block_intervals_file_distance = self._get_blocks()

        start_time = datetime.datetime.now()
        verbose_print('\tGenerating interval tree from block spans ...')
        block_interval_tree = intervaltree.IntervalTree.from_tuples(block_intervals_file_distance)
        del block_intervals_file_distance
        verbose_print('\t\tDone in {}.'.format(datetime.datetime.now() - start_time))

        self.line_length_file_distance = self._compute_file_line_length()
        verbose_print('\tEstimated file line size as {}.'.format(self.line_length_file_distance))
        self.line_length_text_distance = self._compute_text_line_length()
        verbose_print('\tEstimated _text line size as {}.'.format(self.line_length_text_distance))

        contig_intervals_file_distance = self._get_contig_intervals_file_distance()

        self._index = self._assign_blocks_to_contigs(contig_intervals_file_distance=contig_intervals_file_distance,
                                                     block_interval_tree=block_interval_tree)
        self._compute_contig_lengths(contig_intervals_file_distance=contig_intervals_file_distance)

        verbose_print('\tDone in {}.'.format(datetime.datetime.now() - overall_start_time))

    def save_index_to_text(self, index_filename=''):
        """
        Save the current index to a _text file on disk.
        """
        start_time = datetime.datetime.now()
        verbose_print('Saving index to {} ...'.format(index_filename))
        with gzip.open(index_filename, 'wt') as index_file:
            index_file.write('{}\t{}\n'.format(self.line_length_file_distance, self.line_length_text_distance))
            for contig, intervals in sorted(self._index.items()):
                index_file.write('>{}\n'.format(contig))
                for interval in intervals:
                    index_file.write('{}\t{}\t{}\n'.format(interval.begin, interval.end, interval.data))
        verbose_print('\tDone in {}.'.format(datetime.datetime.now() - start_time))

    def load_index_from_text(self, index_filename=''):
        """
        Load an index from a _text file on disk.
        """
        start_time = datetime.datetime.now()
        verbose_print('Loading index from {} ...'.format(index_filename))

        # It's faster to create an interval tree from a list of tuples than from adding intervals one at a time
        index_tuples = {}

        with gzip.open(index_filename, 'rt') as index_file:
            line = index_file.readline()
            split_line = line.rstrip().split('\t')
            self.line_length_file_distance = int(split_line[0])
            self.line_length_text_distance = int(split_line[1])
            line = index_file.readline()

            contig = None
            while line is not '':
                if line.startswith('>'):
                    contig = line.rstrip()[1:]
                    index_tuples[contig] = []
                else:
                    index_tuples[contig].append([int(val) for val in line.rstrip().split('\t')])
                line = index_file.readline()

        self._index = {}
        for contig in index_tuples:
            self._index[contig] = intervaltree.IntervalTree.from_tuples(index_tuples[contig])

        verbose_print('\tDone in {}.'.format(datetime.datetime.now() - start_time))

    def save_contig_lengths(self, contig_length_filename):
        """
        Saves the length of each contig to a tab-delimited 2-column table in :param:`contig_length_filename`
        """
        verbose_print('Saving {} contig lengths to {} ...'.format(len(self._contig_lengths), contig_length_filename))
        with open(contig_length_filename, 'wt') as contig_length_file:
            for contig_name, contig_length in sorted(self._contig_lengths.items()):
                contig_length_file.write('{}\t{}\n'.format(contig_name, contig_length))
        verbose_print('\tDone.')

    def load_contig_lengths(self, contig_length_filename):
        """
        Retrieve the length of each contig from a tab-delimited 2-column table in :param:`contig_length_filename`
        :param contig_length_filename:
        :return:
        """
        verbose_print('Loading contig lengths from {} ...'.format(contig_length_filename))
        self._contig_lengths = {}
        with open(contig_length_filename, 'rt') as contig_length_file:
            for line in contig_length_file:
                split_line = line.split('\t')
                contig_name = split_line[0]
                contig_length = int(split_line[1])
                self._contig_lengths[contig_name] = contig_length
        verbose_print('Loaded {} contig lengths.'.format(len(self._contig_lengths)))

    def get_sequence(self, contig, start, end, strand=1):
        """
        Return the genomic DNA sequence spanning [start, end) on contig.
        :param contig: The name of the contig on which the start and end coordinates are located
        :param start: The start location of the sequence to be returned (this endpoint is included in the interval)
        :param end: The end location of the sequence to be returned (tis endpoint is not included in the interval)
        :param strand: The DNA strand of the sequence to be returned (-1 for negative strand, 1 for positive strand)
        :return: A string of DNA nucleotides of length end-start
        """
        if contig not in self._index:
            raise ContigNotFound(message='Contig {} not found'.format(contig),
                                 requested_contig=contig, valid_contigs=list(self._index.keys()))
        if start < 0:
            raise CoordinateOutOfBounds(message='Start coordinate below 0',
                                        problematic_coordinate=start,
                                        problem_with_start=True,
                                        coordinate_too_small=True,
                                        valid_coordinate_range=(0, self.contig_lengths[contig]),
                                        current_contig=contig)
        if start > self.contig_lengths[contig]:
            raise CoordinateOutOfBounds(message='Start coordinate past end of contig',
                                        problematic_coordinate=start,
                                        problem_with_start=True,
                                        coordinate_too_small=False,
                                        valid_coordinate_range=(0, self.contig_lengths[contig]),
                                        current_contig=contig)
        if end > self.contig_lengths[contig]:
            raise CoordinateOutOfBounds(message='End coordinate past end of contig',
                                        problematic_coordinate=end,
                                        problem_with_start=False,
                                        coordinate_too_small=False,
                                        valid_coordinate_range=(0, self.contig_lengths[contig]),
                                        current_contig=contig)
        if end < 0:
            raise CoordinateOutOfBounds(message='End coordinate below 0',
                                        problematic_coordinate=end,
                                        problem_with_start=False,
                                        coordinate_too_small=True,
                                        valid_coordinate_range=(0, self.contig_lengths[contig]),
                                        current_contig=contig)
        if start >= end:
            raise InvalidCoordinates(start=start, end=end)

        query_length = end - start
        start_pos_file_distance = self._text_distance_to_file_distance(start)

        start_block = sorted(self._index[contig].search(start_pos_file_distance))[0]
        start_block_offset = start_block.data
        verbose_print('Retrieving sequence for {} [{},{}) ...'.format(contig, start, end))

        sequence_start_offset = start_pos_file_distance - start_block.begin

        retrieved_sequence = ''
        with bgzf.BgzfReader(self.bgzipped_fasta_filename, 'rt') as fasta_file:
            fasta_file.seek(bgzf.make_virtual_offset(start_block_offset, sequence_start_offset))
            while len(retrieved_sequence) < query_length:
                retrieved_sequence += fasta_file.readline().rstrip()
        trimmed_sequence = retrieved_sequence[:query_length]

        if strand == -1:
            return reverse_complement(trimmed_sequence)
        else:
            return trimmed_sequence


class Genome:
    """
    Serve up gene locations and sequences.
    """

    def __init__(self,
                 sequence_file=SEQUENCE_FILENAME,
                 feature_filename=TRANSCRIPT_FILENAME,
                 feature_pickle_filename=FEATURE_FILENAME,
                 force_rebuild=False):
        self.sequence_file = sequence_file
        self.feature_filename = feature_filename
        if self.feature_filename.endswith('.gz'):
            self.feature_filename = self.feature_filename[:-3]
        self.feature_pickle_filename = feature_pickle_filename

        # Initialize sequence object
        self.genome_sequence = GenomeSequence(self.sequence_file)

        # Initialize features either from pre-generated data or de-novo from a G*F file
        if not force_rebuild:
            try:
                feature_pickle_file = try_finding_gzipped_file(self.feature_pickle_filename, 'rb')
                if feature_pickle_file:
                    start_time = datetime.datetime.now()
                    self.features, self.feature_to_id = pickle.load(feature_pickle_file)
                    feature_pickle_file.close()
                    verbose_print('Loaded genome features from {} in {}.'.format(self.feature_pickle_filename,
                                                                                 datetime.datetime.now() - start_time))
                else:
                    verbose_print('Pre-made feature file not found; Will generate now ...')
                    force_rebuild = True
            except (ImportError, IOError, OSError, pickle.UnpicklingError, AttributeError, EOFError, ValueError) as e:
                verbose_print('Failed to load cached features: {}'.format(e))
                force_rebuild = True

        if force_rebuild:
            self._populate_features()
            # self._populate_contig_lengths()

    @property
    def contig_names(self):
        """
        Return a list of all the valid sequence contigs in this genome.
        """
        return self.genome_sequence.contig_lengths.keys()

    @property
    def contig_lengths(self):
        """
        Return a dictionary of sequence contigs and their lengths.
        """
        return self.genome_sequence.contig_lengths

    def get_dna_sequence(self, contig, start, end, strand):
        """
        Return the sequence of the specified genomic region.

        :param contig:
        :param start:
        :param end:
        :param strand:
        :return:
        """
        if contig not in self.contig_names:
            raise ContigNotFound(contig, self.contig_names)
        return self.genome_sequence.get_sequence(contig=contig, start=start, end=end, strand=strand)

    def get_dna_sequence_by_name(self, gene_name):
        """
        Return the genomic sequence of the specified <gene_name>.

        :param gene_name:
        :return:
        """
        if gene_name not in self.feature_to_id:
            raise GeneNotFound(gene_name)
        ensembl_id = self.feature_to_id[gene_name]
        return self.features.get_sequence(ensembl_id)

    def _populate_features(self):
        """
        Generate the internal data structures based on information in the G*F file and the contigs' lengths.
        """
        start_time = datetime.datetime.now()
        verbose_print('Generating gene info and translation data from GFF file {} ...'.format(self.feature_filename))

        feature_file = try_finding_gzipped_file(self.feature_filename, 'rt')
        if feature_file is None:
            raise Exception(
                'G*F file not found! Looked for {} and {}'.format(self.feature_filename, self.feature_filename + '.gz'))
        # Generate from G*F
        if self.feature_filename.endswith('gff3'):
            features, self.feature_to_id = parse_gff3(feature_file, sources=FEATURE_SOURCES, types=FEATURE_TYPES)
        elif self.feature_filename.endswith('gtf'):
            features, self.feature_to_id = parse_gtf(feature_file, sources=FEATURE_SOURCES, types=FEATURE_TYPES)
        else:
            raise Exception('Invalid GFF filename. Must end in .gff3 or .gtf')
        feature_file.close()

        # Get features
        self.features = IntervalDict(features)

        # Pickle for future use
        with gzip.open(self.feature_pickle_filename + '.gz', 'wb') as feature_pickle_file:
            pickle.dump((self.features, self.feature_to_id), feature_pickle_file, protocol=-1)

        verbose_print(
            '{} genes by Ensembl ID, {} gene symbols in {}'.format(len(self.features), len(self.feature_to_id),
                                                                   datetime.datetime.now() - start_time))

    def get_gene_by_ensembl_id(self, ensembl_id):
        """
        Return a dictionary of information fields about the gene specified by the given <ensembl_id>.

        :param ensembl_id:
        :return:
        """
        return self.features[ensembl_id]

    def get_gene_by_name(self, gene_name):
        """
        Return a dictionary of information fields about the gene specified by the <gene_name>.
        Return None if the <gene_name> has no Ensembl ID or if the Ensembl ID is not found.

        :param gene_name:
        :return:
        """
        # Look up this <gene_name> in the gene-name-to-Ensembl-ID dictionary
        if gene_name in self.feature_to_id:
            # If found, then get this <gene_name>'s Ensembl ID
            ensembl_id = self.feature_to_id[gene_name]

            # Then look up this Ensembl ID within the list of Ensembl IDs stored
            if ensembl_id in self.features:
                # Return found feature information corresponding to this Ensembl ID
                return self.features[ensembl_id]
        raise GeneNotFound(gene_name)

    def get_genes_by_location(self, contig, start=0, end=0, strict=False):
        """
        Return an ordered  dictionary of genes overlapping the specified region.
        Currently ignores <strand> parameter, returning genes on both strands.
        :param contig:
        :param start:
        :param end:
        :param strict:
        :return:
        """
        if contig not in self.contig_names:
            raise ContigNotFound(contig, self.contig_names)
            # if the requested contig is in self.contig names, it's valid, but if there are no genes there it
            # may not exist in the features object.
        if contig in self.features.contig_names:
            return self.features.overlapping(contig=contig, start=start, end=end, strict=strict)
        else:
            return []


class Variants:
    """
    Serve up variants from a tabix-indexed VCF file.
    """

    class VariantTable(tables.IsDescription):
        rs_id = tables.StringCol(16)
        is_snp = tables.BoolCol()
        contig = tables.StringCol(64)
        start = tables.Int32Col()
        end = tables.Int32Col()
        ref = tables.StringCol(256)
        alt = tables.StringCol(256)
        qual = tables.Float32Col()
        affected_area = tables.StringCol(64)
        putative_impact = tables.StringCol(8)
        affected_gene = tables.StringCol(16)
        pathogenicity = tables.Int32Col()
        GT = tables.StringCol(16)

    def __init__(self, vcf_filename=VARIANT_FILENAME, vcf_field_model=FREE_BAYES,
                 variant_table_filename=VARIANT_HDF5_FILENAME,
                 variant_rs_id_index_filename=VARIANT_INDEX_FILENAME,
                 force_rebuild=False):
        """
        Create an object used to interface with tables of client Variant information on disk.
        :param vcf_filename:
        :param vcf_field_model:
        :param variant_table_filename:
        :param variant_rs_id_index_filename:
        :param force_rebuild:
        """
        self.vcf_filename = vcf_filename
        self.vcf_field_model = vcf_field_model
        self.variant_rs_id_index_filename = variant_rs_id_index_filename

        self.variant_file = None
        self.rs_id_index = {}

        # Try opening the cached data file:
        start_time = datetime.datetime.now()
        if not force_rebuild:
            try:
                verbose_print('Opening Variant table file {}.'.format(variant_table_filename))
                self.variant_file = tables.open_file(filename=variant_table_filename, mode='r')
                verbose_print('\tDone in {}'.format(datetime.datetime.now() - start_time))
                self._load_rs_id_index(self.variant_rs_id_index_filename)
            except (OSError, IOError, tables.HDF5ExtError):
                verbose_print('\tFailed!')
                force_rebuild = True
                if self.variant_file:
                    self.variant_file.close()

        # Force extraction of information from VCF and generating new table
        if force_rebuild:
            self._populate_from_vcf(vcf_filename, variant_table_filename)
            self._save_rs_id_index(self.variant_rs_id_index_filename)
            # re-open the table file in read-only mode for later use.
            self.variant_file = tables.open_file(filename=variant_table_filename, mode='r')

        self.variant_group = self.variant_file.get_node('/variants')

    def __del__(self):
        """
        Destructor. Makes sure that the HDF5 variant file is explicitly closed.
        :return:
        """
        if self.variant_file:
            print('closing variant file...')
            self.variant_file.close()
        print('variant file closed.')

    def _save_rs_id_index(self, variant_rs_id_index_filename):
        """
        Saves the index mapping rs_ids to contigs to a pickle contained in :param:`variant_rs_id_index_filename`
        :param variant_rs_id_index_filename:
        :return:
        """
        start_time = datetime.datetime.now()
        verbose_print('Saving rs_id index to {}...'.format(variant_rs_id_index_filename))
        with gzip.open(variant_rs_id_index_filename, 'wb') as index_file:
            pickle.dump(self.rs_id_index, index_file, protocol=-1, fix_imports=True)
        verbose_print('\tDone in {}'.format(datetime.datetime.now() - start_time))

    def _load_rs_id_index(self, variant_rs_id_index_filename):
        """
        Load the index mapping rs_ids to contigs into memory from :param:`variant_rs_id_index_filename`

        :param variant_rs_id_index_filename:
        :return:
        """
        start_time = datetime.datetime.now()
        verbose_print('Loading rs_id index from {}...'.format(variant_rs_id_index_filename))
        with gzip.open(variant_rs_id_index_filename, 'rb') as index_file:
            self.rs_id_index = pickle.load(index_file, fix_imports=True)
        verbose_print('\tDone in {}'.format(datetime.datetime.now() - start_time))

    def _insert_vcf_row(self, variant_tuple, variant_cursor):
        """
        Parse a tuple of VCF 4.2 entries and insert them into the current row specified by <variant_cursor>.
        Annotation of CLNSIG field: "Variant Clinical Significance:
            0 - Uncertain significance,
            1 - Not provided,
            2 - Benign,
            3 - Likely benign,
            4 - Likely pathogenic,
            5 - Pathogenic,
            6 - Drug response,
            7 - Histocompatibility, and
            255 - Other".
        """
        # Split up multiple entries in rs_id field (if present)
        primary_key_field = list(self.vcf_field_model['primary_key'].keys())[0]
        primary_key_pos, primary_key_parse_func = self.vcf_field_model['primary_key'][primary_key_field]

        primary_keys = primary_key_parse_func(variant_tuple[primary_key_pos]).split(';')

        # TODO: also use the 2nd or later RSID
        # if len(primary_keys) > 1:
        #     verbose_print('Got multiple ids for same variant: {}'.format(', '.join(primary_keys)))

        for primary_key in primary_keys:
            # Assign the primary key to this variant. Note that re-doing the parsing for each of the ids in a
            # multi-id variant, as we do below, is not the most efficient way of handling this situation.
            # But since this is expected to be rare, we don't optimize yet.
            variant_cursor[primary_key_field] = primary_key

            # Get core fields
            for field_name, (field_position, parse_func) in list(self.vcf_field_model['core_fields'].items()):
                variant_cursor[field_name] = parse_func(variant_tuple[field_position])
            variant_cursor['contig'] = convert_chromosome(variant_cursor['contig'].decode())
            variant_cursor['start'] -= 1

            # end position is with respect to reference sequence
            variant_cursor['end'] = variant_cursor['start'] + len(variant_cursor['ref'])
            # get info fields
            split_info = variant_tuple[self.vcf_field_model['info_position']].split(';')

            for field_atom in split_info:
                try:
                    field_key, field_value = field_atom.split('=')
                except ValueError:
                    pass
                else:
                    # Parse the annotation field
                    if field_key == 'ANN':
                        annotations = field_value.split('|')
                        variant_cursor['affected_area'] = annotations[1]
                        variant_cursor['putative_impact'] = annotations[2]
                        variant_cursor['affected_gene'] = annotations[3]

                    if field_key == 'CLNSIG':
                        clinsig_values = [int(v) for v in field_value if field_value in {'2', '3', '4', '5'}]
                        if len(clinsig_values) > 0:
                            variant_cursor['pathogenicity'] = max(clinsig_values)

                    # Look at the comma-separated _elements of the alt field and the ref field. If any
                    # of them are longer than 1 nucleotide, classify this variant as an indel, otherwise it is a SNP.
                    variant_cursor['is_snp'] = True
                    for allele in variant_cursor['alt'].decode().split(',') + [variant_cursor['ref']]:
                        if len(allele) > 1:
                            variant_cursor['is_snp'] = False

            # Process "secondary" fields that have field names in column 8 and values in column 9, all
            # colon-delimited.
            try:
                for key, value in zip(variant_tuple[self.vcf_field_model['secondary_field_keys']].split(':'),
                                      variant_tuple[self.vcf_field_model['secondary_field_values']].split(':')):
                    if key == 'GT':
                        if value == 1 or value == str(1):
                            value = '0/1'
                        variant_cursor[key] = value
            except IndexError:
                verbose_print('Error in FORMAT and/or SAMPLE column')

            # update map of rs_id to contigs
            self.rs_id_index[variant_cursor['rs_id'].decode()] = variant_cursor['contig'].decode()

            variant_cursor.append()

    def _populate_from_vcf(self, vcf_filename, variant_table_filename):
        """
        Generate internal data structures from VCF file and cache them to disk for future reference.
        """
        start_time = datetime.datetime.now()
        verbose_print('Parsing VCF file and building variant table ...')

        # Open the vcf file and iterate over it
        data_start_pos = None
        with gzip.open(vcf_filename, 'rt') as vcf_file:
            # Skip past comments and header
            line = vcf_file.readline()
            while line.startswith('#'):
                # Remember this position so we can seek back to it
                data_start_pos = vcf_file.tell()
                line = vcf_file.readline()

            # Count rows
            verbose_print('Pass 1 of 2: Counting number of rows in each contig ...')
            precount_start_time = datetime.datetime.now()

            rows_per_contig = collections.defaultdict(lambda: 0)
            line = vcf_file.readline()
            while line != '':
                rows_per_contig[
                    convert_chromosome(line.split('\t')[self.vcf_field_model['core_fields']['contig'][0]])] += 1
                line = vcf_file.readline()

            verbose_print('\tDone in {}'.format(datetime.datetime.now() - precount_start_time))
            vcf_file.seek(data_start_pos)

            # Initialize table file
            with tables.open_file(filename=variant_table_filename, mode='w', title='Client variant information',
                                  filters=tables.Filters(complevel=VARIANT_HDF5_COMPRESSION_LEVEL,
                                                         complib=VARIANT_HDF5_COMPRESSOR)) as h5file:
                # Initialize group
                self.variant_group = h5file.create_group('/', 'variants', 'Client sequence variants')
                # Create a dictionary of cursors (pointer to the currently active row)
                variant_cursors = {}

                # Parse the tuples in each row and stick them into our tables
                verbose_print('Pass 2 of 2: Populating variant table from VCF contents...')
                for row_num, vcf_row in enumerate(vcf_file):
                    if row_num % 1000000 == 0:
                        verbose_print('\tProcessing VCF row {} ...'.format(row_num + 1))
                    if vcf_row is not '':
                        split_row = vcf_row.rstrip().split('\t')
                        contig = convert_chromosome(split_row[0])
                        if contig not in variant_cursors:
                            verbose_print('\tCreating variant table for contig {} ...'.format(contig))
                            new_table = h5file.create_table(where=self.variant_group,
                                                            name='contig_{}_variants'.format(contig),
                                                            description=self.VariantTable,
                                                            title='Variants for contig {}'.format(contig),
                                                            expectedrows=rows_per_contig[contig])

                            variant_cursors[contig] = new_table.row
                        self._insert_vcf_row(split_row, variant_cursors[contig])

                for contig in variant_cursors:
                    variant_table = self.variant_group._f_get_child('contig_{}_variants'.format(contig))
                    variant_table.flush()
                    # create index
                    for col_name in sorted((
                            'contig', 'start', 'end', 'pathogenicity', 'putative_impact', 'qual', 'rs_id')):
                        verbose_print('\tGenerating index for column {} in contig {}'.format(col_name, contig))
                        variant_table.cols._f_col(col_name).create_csindex(tmp_dir='/tmp')

        verbose_print('\tDone in {}'.format(datetime.datetime.now() - start_time))

    @staticmethod
    def _tuple_to_dict(variant_tuple, col_names):
        """
        Converts a tuple of variant information as output from a PyTables query, and returns a dictionary
        of field-value pairs. Bytes will be converted to strings as appropriate.
        """
        variant_dict = {}
        for key, value in zip(col_names, variant_tuple):
            try:
                variant_dict[key] = value.decode()
            except AttributeError:
                variant_dict[key] = value
        variant_dict['alleles'] = get_alleles(gt_field=variant_dict['GT'],
                                              ref=variant_dict['ref'],
                                              alt=variant_dict['alt'])
        return variant_dict

    def get_variant_by_rs_id(self, rs_id, allow_snps=True, allow_indels=True):
        """
        Return a dictionary of variant attributes for the variant matching <rs_id>.
        If <rs_id> is not found, return None.
        """
        if rs_id in self.rs_id_index:
            contig = self.rs_id_index[rs_id]

            query_string = '(rs_id == {})'.format(rs_id.encode())
            if allow_snps and allow_indels:
                pass
            elif allow_snps:
                query_string += ' & (is_snp == True)'
            else:
                query_string += ' & (is_snp == False)'
            verbose_print('Query: {}'.format(query_string))

            table = self.variant_group._f_get_child('contig_{}_variants'.format(contig))
            query_results = table.read_where(query_string)

            if len(query_results) > 0:
                return self._tuple_to_dict(query_results[0], table.colnames)
        return None

    def get_variants_by_location(self, contig, start=0, end=0, minimum_quality=0, allow_snps=True, allow_indels=False,
                                 minimum_pathogenicity=0, include_putative_impacts=(), exclude_putative_impacts=(),
                                 convert_to_dict=False):
        """
        Given a genomic region specified by <contig>, <start>, <end>,
        return all variants overlapping that region.

        :param contig:
        :param start:
        :param end:
        :param minimum_quality:
        :param allow_snps: Whether or not SNPs (variants with either alt or ref fields equal to one nucleotide) should be returned.
        :param allow_indels: Whether or not indels (variants with either alt or ref fields greater than one nucleotide) should be returned.
        :param minimum_pathogenicity: An integer. If set, any variants below this pathogenicity level will be excluded.
        :param include_putative_impacts: An iterable. If set, only variants with these putative impact classifications will be returned
        :param exclude_putative_impacts: An iterable. If set, variants with these putative impact classifications will not be returned
        :param convert_to_dict: If True, return a list of dictionaries of field-value pairs,
         otherwise return a tuple consisting of a list of tuples, and a dictionary mapping field
         names to tuple indices.
        :return:
        """
        start_time = datetime.datetime.now()
        verbose_print(
            'Finding all variants of in contig {} ({},{}) with quality > {}, SNPS: {}, Indels: {}...'.format(contig,
                                                                                                             start,
                                                                                                             end,
                                                                                                             minimum_quality,
                                                                                                             allow_snps,
                                                                                                             allow_indels))
        try:
            table = self.variant_group._f_get_child('contig_{}_variants'.format(contig))
        except tables.exceptions.NoSuchNodeError:
            # We have no variant info for this contig. Either because it's not a valid contig or there were no variants
            # there.
            if convert_to_dict:
                return []
            else:
                return [], {}

        if start > 0 or end > 0:
            query_string = (
                '(contig == {}) & (start >= {}) & (end <= {}) & (qual > {})'.format(contig.encode(), start, end,
                                                                                    minimum_quality))
        else:
            query_string = ('(contig == {}) & (qual > {})'.format(contig.encode(), minimum_quality))

        if allow_snps and allow_indels:
            pass
        elif allow_snps:
            query_string += ' & (is_snp == True)'
        else:
            query_string += ' & (is_snp == False)'

        include_putative_impacts = set(include_putative_impacts)
        exclude_putative_impacts = set(exclude_putative_impacts)
        include_putative_impacts.difference_update(exclude_putative_impacts)

        if include_putative_impacts:
            query_string += ' & ({})'.format(' | '.join(
                ['(putative_impact == {})'.format(included_impact.encode()) for included_impact in
                 include_putative_impacts]))

        if exclude_putative_impacts:
            query_string += ' & ({})'.format(' & '.join(
                ['(putative_impact != {})'.format(excluded_impact.encode()) for excluded_impact in
                 exclude_putative_impacts]))

        if minimum_pathogenicity:
            query_string += ' & (pathogenicity >= {}'.format(minimum_pathogenicity)

        verbose_print('Query: {}'.format(query_string))

        query_results = table.read_where(query_string)

        verbose_print('\tFound {} variants in {}'.format(len(query_results), datetime.datetime.now() - start_time))

        if convert_to_dict:
            start_time = datetime.datetime.now()
            verbose_print('Converting variants to dictionaries...')
            query_results = [self._tuple_to_dict(var, table.colnames) for var in query_results]
            verbose_print('\tDone in {}'.format(datetime.datetime.now() - start_time))
            return query_results
        else:
            field_mapping = dict([t[::-1] for t in enumerate(table.colnames)])
            return query_results, field_mapping

    def get_variant_columns_by_location(self, contig, start=0, end=0, minimum_quality=0, allow_snps=True,
                                        allow_indels=False,
                                        minimum_pathogenicity=0, include_putative_impacts=(),
                                        exclude_putative_impacts=(),
                                        fields=(
                                                'start', 'end', 'putative_impact', 'pathogenicity', 'ref', 'alt',
                                                'GT')):
        """
        Similar to .get_variants_by_location except that instead of returning an array of structured arrays
        or a list of dictionaries, this method returns a dictionary of arrays, one for each column in the
        result. Iterating over these arrays is much faster than over the row iterator returned by standard
        query methods.

        :param contig:
        :param start:
        :param end:
        :param minimum_quality:
        :param allow_snps:
        :param allow_indels:
        :param minimum_pathogenicity:
        :param include_putative_impacts:
        :param exclude_putative_impacts:
        :param fields: Which fields will be included as columns in the result
        :return:
        """
        start_time = datetime.datetime.now()
        verbose_print(
            'Finding all variants of in contig {} ({},{}) with quality > {}, SNPS: {}, Indels: {}...'.format(contig,
                                                                                                             start,
                                                                                                             end,
                                                                                                             minimum_quality,
                                                                                                             allow_snps,
                                                                                                             allow_indels))

        verbose_print('Returning arrays for fields {}.'.format(', '.join(fields)))

        try:
            table = self.variant_group._f_get_child('contig_{}_variants'.format(contig))
        except tables.exceptions.NoSuchNodeError:
            # We have no variant info for this contig. Either because it's not a valid contig or there were no variants
            # there.
            return dict([(field, []) for field in fields])

        if start > 0 or end > 0:
            query_string = (
                '(contig == {}) & (start >= {}) & (end <= {}) & (qual > {})'.format(contig.encode(), start, end,
                                                                                    minimum_quality))
        else:
            query_string = ('(contig == {}) & (qual > {})'.format(contig.encode(), minimum_quality))

        if allow_snps and allow_indels:
            pass
        elif allow_snps:
            query_string += ' & (is_snp == True)'
        else:
            query_string += ' & (is_snp == False)'

        include_putative_impacts = set(include_putative_impacts)
        exclude_putative_impacts = set(exclude_putative_impacts)
        include_putative_impacts.difference_update(exclude_putative_impacts)

        if include_putative_impacts:
            query_string += ' & ({})'.format(' | '.join(
                ['(putative_impact == {})'.format(included_impact.encode()) for included_impact in
                 include_putative_impacts]))

        if exclude_putative_impacts:
            query_string += ' & ({})'.format(' & '.join(
                ['(putative_impact != {})'.format(excluded_impact.encode()) for excluded_impact in
                 exclude_putative_impacts]))

        if minimum_pathogenicity:
            query_string += ' & (pathogenicity >= {}'.format(minimum_pathogenicity)

        verbose_print('Query: {}'.format(query_string))

        row_coordinates = [r.nrow for r in table.where(query_string)]
        query_results = {}
        for field in fields:
            query_results[field] = table.read_coordinates(row_coordinates, field=field)

        verbose_print(
            '\tFound {} variants in {}'.format(len(query_results[list(query_results.keys())[0]]),
                                               datetime.datetime.now() - start_time))
        return query_results


class KnownVariant(tables.IsDescription):
    rs_id = tables.StringCol(11)
    contig = tables.StringCol(2)
    start = tables.Int32Col()
    ref = tables.StringCol(250)
    alt = tables.StringCol(320)
    gene = tables.StringCol(178)
    af = tables.StringCol(77)


class DescribedVariant(tables.IsDescription):
    rs_id = tables.StringCol(11)
    clinvar_allele = tables.StringCol(55)
    clin_sig = tables.StringCol(39)
    phenotype = tables.StringCol(419)
    description = tables.StringCol(12390)


class SupplementalVariantInfo:
    def __init__(self, table_description, gzipped_tsv_filename, supplemental_variant_table_filename,
                 force_rebuild=False):
        self.supplemental_variant_table_filename = supplemental_variant_table_filename
        self.supplemental_variant_file = None
        start_time = datetime.datetime.now()

        if not force_rebuild:
            try:
                verbose_print('Opening supplemental variant table {} ...'.format(supplemental_variant_table_filename))
                self.supplemental_variant_file = tables.open_file(filename=supplemental_variant_table_filename,
                                                                  mode='r')
            except (IOError, OSError):
                verbose_print('\tFailed!')
                force_rebuild = True
            else:
                verbose_print('\tDone in {}'.format(datetime.datetime.now() - start_time))

        if force_rebuild:
            self.generate_pytable_from_tsv(table_description=table_description,
                                           gzipped_tsv_filename=gzipped_tsv_filename,
                                           supplemental_variant_table_filename=supplemental_variant_table_filename)
            self.supplemental_variant_file = tables.open_file(filename=supplemental_variant_table_filename, mode='r')

        self.table = self.supplemental_variant_file.get_node('/supplemental_variant_info')

    def __del__(self):
        '''
        Destructor. Closes open supplemental variant file
        :return:
        '''
        if self.supplemental_variant_file:
            print('closing supplemental_variant_file...')
            self.supplemental_variant_file.close()
        print('supplemental_variant_file closed.')

    def generate_pytable_from_tsv(self, table_description, gzipped_tsv_filename, supplemental_variant_table_filename):
        """
        Generates a PyTables file from the data in the TSV file
        :param table_description: A class generated according to the PyTables documentation that descrivbes the record structure of the table.
        :param gzipped_tsv_filename:
        :param supplemental_variant_table_filename:
        :return:
        """
        overall_start_time = datetime.datetime.now()
        verbose_print('Generating {} from {} ...'.format(supplemental_variant_table_filename, gzipped_tsv_filename))
        with gzip.open(gzipped_tsv_filename, 'rt') as tsv_file:
            verbose_print('\tPass 1: Counting number of rows ...')
            pass_1_start_time = datetime.datetime.now()
            line_counter = 0
            for line in tsv_file:
                line_counter += 1
            verbose_print('\t\tFound {} rows in {}'.format(line_counter, datetime.datetime.now() - pass_1_start_time))
            tsv_file.seek(0)
            # line_counter = 149125208

            pass_2_start_time = datetime.datetime.now()
            with tables.open_file(filename=supplemental_variant_table_filename, mode='w',
                                  title='Client variant information',
                                  filters=tables.Filters(complevel=SUPPLEMENTAL_VARIANT_COMPRESSION_LEVEL,
                                                         complib=SUPPLEMENTAL_VARIANT_COMPRESSOR)) as h5file:

                table = h5file.create_table(where='/',
                                            name='supplemental_variant_info',
                                            description=table_description,
                                            title='Supplemental Variant Information',
                                            expectedrows=line_counter - 1)

                verbose_print('\tPass 2: Populating table with {} byte rows ...'.format(table.rowsize))

                cursor = table.row
                tsv_file.readline()  # throw away header line

                for line_num, line in enumerate(tsv_file):
                    if line_num % 1000000 == 0:
                        verbose_print('\t\tProcessing line {}'.format(line_num + 1, cursor.nrow + 1))

                    split_line = line.rstrip().split('\t')
                    new_row = {}
                    row_empty = True  # this flag will be set to False if any of the non-primary-key fields have data

                    try:
                        for field_pos, field_value in enumerate(split_line):
                            field_name, field_func = SUPPLEMENTAL_FILE_FIELD_MODEL[field_pos]

                            if field_name in table_description.columns:
                                new_row[field_name] = field_func(field_value)
                                if field_name != 'rs_id' and len(field_value) > 0:
                                    row_empty = False

                    except TypeError as te:
                        verbose_print(str(te))
                        verbose_print(line)
                    else:
                        if not row_empty:
                            # verbose_print('Keeping {}'.format(new_row))
                            for field in new_row:
                                # verbose_print(field, new_row[field])
                                cursor[field] = new_row[field]

                            cursor.append()

                table.flush()
                verbose_print('\t\tDone populating {} rows in {}'.format(table.nrows,
                                                                         datetime.datetime.now() - pass_2_start_time))

                index_start_time = datetime.datetime.now()
                for col_name in ('rs_id',):
                    verbose_print('\tGenerating index for column {}'.format(col_name))
                    table.cols._f_col(col_name).create_csindex(tmp_dir='/tmp')
                verbose_print('\t\tDone in {}'.format(datetime.datetime.now() - index_start_time))

        verbose_print('All done generating table in {}'.format(datetime.datetime.now() - overall_start_time))

    @staticmethod
    def _tuple_to_dict(variant_tuple, col_names):
        """
        Convert a tuple of field information into a dictionary of values keyed by field name
        according to the list of columns in :param:`col_names`
        :param variant_tuple:
        :param col_names:
        :return:
        """
        variant_dict = {}
        for key, value in zip(col_names, variant_tuple):
            try:
                variant_dict[key] = value.decode()
            except AttributeError:
                variant_dict[key] = value
        return variant_dict

    def get_supplemental_variant_info_by_rs_id(self, rs_id):
        """
        Return a single variant dictionary containing the fields in this table for `rs_id`
        using supplemental information.
        """
        query_string = '(rs_id == {})'.format(rs_id.encode())
        query_results = (self.table.read_where(query_string))
        if len(query_results) > 0:
            return self._tuple_to_dict(query_results[0], self.table.colnames)
        else:
            return None


class IntervalDict:
    """
    This class stores genomic regions and their locations in a way that facilitates indexing by region_id or location.
    It supports setting and deleting, and querying by position with the .overlapping() method.
    """

    def __init__(self, region_dict=None):
        """
        Create a new IntervalDict, optionally populating with the regions in <region_dict>,
        a dictionary, keyed by region_identifier, of sub-dictionaries defining the 
        regions. These subdictionaries contain, at a minimum, fields called
        `contig`, `start`, and `end` to define their genomic location.
        """
        self._regions = collections.OrderedDict()
        self._locations = {}
        if region_dict is None: region_dict = {}

        # Update with a given region_dict
        for region_id, region in list(region_dict.items()):
            self.__setitem__(region_id, region)

    def __getitem__(self, region_id):
        """
        Getter.
        :param region_id: str
        :return: region dictionary {chrom: X, start: Y, stop: Z} query result
        """
        return self._regions[region_id]

    def __setitem__(self, new_region_id, new_region):
        """
        Setter.
        :param new_region_id: str
        :param new_region: dict
        :return: None
        """
        assert 'contig' in new_region
        assert 'start' in new_region
        assert 'end' in new_region

        if new_region['end'] <= new_region['start']:
            raise InvalidCoordinates(start=new_region['start'], end=new_region['end'],
                                     message='Start coordinate {} is greater than end coordinate {} when trying to create region {}'.format(
                                         new_region['start'], new_region['end'], new_region_id))

        # If rewriting this region with different coordinates, need to first delete it from the region tree
        if new_region_id in self._regions and (
                        new_region['start'] != self._regions[new_region_id]['start'] or new_region['end'] !=
                    self._regions[new_region_id]['end']):
            self.__delitem__(new_region_id)
        self._regions[new_region_id] = new_region

        # Now add the updated region to the location tree
        if new_region['contig'] not in self._locations:
            # Add as a new region entry if region ID is a new ID
            self._locations[new_region['contig']] = intervaltree.IntervalTree()
        self._locations[new_region['contig']].addi(new_region['start'], new_region['end'], new_region_id)

    def __delitem__(self, region_id):
        """
        Delete :param:`region_id` from the IntervalDict.
        :param region_id: str
        :return: None
        """
        self._locations[self._regions[region_id]['contig']].removei(self._regions[region_id]['start'],
                                                                    self._regions[region_id]['end'],
                                                                    region_id)
        del (self._regions[region_id])

    def __len__(self):
        """
        Get length.
        :return: int
        """
        return len(self._regions)

    def __repr__(self):
        repr_text = ''
        for region_id, region in list(self._regions.items()):
            repr_text += '{}: {}\n'.format(region_id, ', '.join(
                ['='.join([str(pair) for pair in item]) for item in list(region.items())]))
        return repr_text

    def __iter__(self):
        return iter(list(self._regions.keys()))

    def keys(self):
        return list(self._regions.keys())

    def values(self):
        return list(self._regions.values())

    def items(self):
        return list(self._regions.items())

    def iterkeys(self):
        return iter(self._regions.keys())

    def itervalues(self):
        return iter(self._regions.values())

    def iteritems(self):
        return iter(self._regions.items())

    def overlapping(self, contig, start=0, end=0, strict=False):
        """
        If <strict> is False (default), return a new RegionDict of all regions overlapping the query region by at least one base pair
        If <strict> is True, return a new RegionDict of all regions completely enclosed by the query region
        """
        if end < start:
            raise InvalidCoordinates(start=start, end=end)

        results = IntervalDict()
        if start or end:
            for overlap in self._locations[contig].search(start, end, strict=strict):
                results[overlap.data] = self._regions[overlap.data]
        else:
            for overlap in self._locations[contig]:
                results[overlap.data] = self._regions[overlap.data]
        return results

    def intersection(self, other):
        """
        Return the regions in self that overlap with <other>.
        """
        result = IntervalDict()
        for chrom in self._locations:
            for location in self._locations[chrom]:
                if other._locations[chrom].search(location):
                    result[location.data] = self._regions[location.data]
        return result

    def __and__(self, other):
        return self.intersection(other)

    def difference(self, other):
        """
        Return the regions in self that do not overlap with <other>
        """
        result = IntervalDict()
        for chrom in self._locations:
            for location in self._locations[chrom]:
                if not other._locations[chrom].search(location):
                    result[location.data] = self._regions[location.data]
        return result

    def __sub__(self, other):
        return self.difference(other)

    @property
    def contig_names(self):
        return self._locations.keys()


class IndexedTsv:
    """
    Interface for a block-gzipped tsv file stored on disk that allows rapid random access to rows of
    that file. Very similar to tabix except more flexible in that it is not restricted to indexing
    by genomic coordinates.
    """

    def __init__(self, bgzipped_filename, index_filename='', sep='\t', index_field='SYMBOL', force_reindex=False):
        """
        Returns a new IndexedTsv object that indexes the data in :param:`bgzipped_filename`. It will attempt
        to load a pre-generated index from `index_filename` (default is :param:`bgzipped_filename + .idx`.
        If that fails, it will create a new index and save it to :param:`index_filename`.

        Assumes that the first row of the file is a header row with field names. The values in :param:`index_field`
        will be indexed such that an entire row can be rapidly retrieved given the value in :param:`index_field`.

        :param bgzipped_filename: the filename of the compressed (with block gzip) tsv file to index.
        :param index_filename: the filename of the index. Defaults to :param:`bgzipped_filename + .idx`
        :param sep: the separator between fields. Defaults to `tab`
        :param index_field: which field to use for indexing
        :param force_reindex: forces the index to be rebuilt from scratch.
        :return: an IndexedTsv object.
        """
        self.bgzipped_filename = bgzipped_filename
        self.index_field = index_field
        self.sep = sep

        if not index_filename:
            index_filename = self.bgzipped_filename + '.idx'

        try:
            self._load_index(index_filename)
        except (OSError, IOError, ValueError, EOFError):
            force_reindex = True

        if force_reindex:
            try:
                self._create_index(index_filename=index_filename)
            except ValueError:
                verbose_print(
                    '{} does not appear to be a valid bgzipped file. Converting now...'.format(self.bgzipped_filename))
                convert_gzipped_to_bgzipped(self.bgzipped_filename)
                self._create_index(index_filename=index_filename)

    def _load_index(self, index_filename):
        """
        Load the index from :param:`index_filename`

        :param index_filename:
        :return:
        """
        self._offset_index = {}

        with open(index_filename, 'rt') as index_file:
            self.fields = index_file.readline().rstrip().split(self.sep)
            for line in index_file:
                split_line = line.rstrip().split('\t')
                self._offset_index[split_line[0]] = int(split_line[1])

    def _create_index(self, index_filename):
        """
        Create a new index and save it to :param:`index_filename`
        :param index_filename:
        :return:
        """
        start_time = datetime.datetime.now()
        verbose_print('Indexing {} ...'.format(self.bgzipped_filename))
        self._offset_index = {}
        line_num = 0
        with bgzf.BgzfReader(self.bgzipped_filename, 'rt') as in_file:
            # process header line
            line = in_file.readline()
            self.fields = line.rstrip().split(self.sep)
            self.index_col = self.fields.index(self.index_field)  # find the column containing the index field

            start_pos = in_file.tell()  # record position at beginning of line
            # get the first data line
            line = in_file.readline().rstrip()

            key = None
            while line is not '':

                if line_num % 1000000 == 0:
                    verbose_print('\tIndexing line {} ...'.format(line_num + 1))
                try:
                    key = line.split(self.sep)[self.index_col]
                except IndexError:
                    verbose_print('Malformed line {} : {}'.format(line_num, line))
                self._offset_index[key] = start_pos
                line_num += 1
                start_pos = in_file.tell()  # record position at beginning of line
                line = in_file.readline().rstrip()

            verbose_print('\tRead {} lines.'.format(line_num))

        verbose_print('Saving index to {} ...'.format(index_filename))
        with open(index_filename, 'wt') as index_file:
            index_file.write(self.sep.join(self.fields) + '\n')
            for key in self._offset_index:
                index_file.write('{}\t{}\n'.format(key, self._offset_index[key]))
        verbose_print('Done in {}'.format(datetime.datetime.now() - start_time))

    def _parserow(self, row_text):
        """
        Parse :param:`row_text` and return it as a dictionary.
        :param row_text:
        :return:
        """
        return dict(zip(self.fields, row_text.split(self.sep)))

    def __getitem__(self, item):
        """
        Retrieves a single row using Python bracket notation.
        :param item:
        :return:
        """
        if item in self._offset_index:
            with bgzf.BgzfReader(self.bgzipped_filename, 'rt') as in_file:
                in_file.seek(self._offset_index[item])
                return self._parserow(in_file.readline().rstrip())

    def __contains__(self, key):
        """
        Performs the membership test for :param:`key`.
        :param key:
        :return:
        """
        return key in self._offset_index


# ======================================================================================================================
# Exceptions
# ======================================================================================================================
class VariantNotFound(Exception):
    def __init__(self, requested_variant, message=''):
        """
        Exception indicating that the requested variant was not found in the resource (whether client variants,
        described variants, or known variants)
        """
        if not message:
            message = 'Variant {} not found.'.format(requested_variant)
        self.requested_variant = requested_variant
        super().__init__(message)


class GeneNotFound(Exception):
    def __init__(self, requested_gene, message=''):
        """
        Exception indicating that a gene requested does not exist in the genome.
        """
        if not message:
            message = 'Gene {} not found.'.format(requested_gene)
        self.requested_gene = requested_gene
        super().__init__(message)


class ContigNotFound(Exception):
    def __init__(self, requested_contig, valid_contigs, message=''):
        """
        Exception indicating that a contig requested does not exist in the genome.
        """
        if not message:
            message = 'Contig {} not found. Valid contigs: {}.'.format(requested_contig, ', '.join(valid_contigs))
        self.requested_contig = requested_contig
        self.valid_contigs = valid_contigs
        super().__init__(message)


class CoordinateOutOfBounds(Exception):
    def __init__(self, problematic_coordinate, problem_with_start, coordinate_too_small, valid_coordinate_range,
                 current_contig, message=''):
        """
        One of the requested genome coordinates is either below 0 or greater than the size of the current contig.

        A pair of boolean variables indicates which sub-condition is active:
            If problem_with_start, it's the start coordinate, otherwise the end.
            If coordinate_too_small, it's too small, otherwise it's too big.
        """
        self.problematic_coordinate = problematic_coordinate
        self.problem_with_start = problem_with_start
        self.coordinate_too_small = coordinate_too_small
        self.valid_coordinate_range = valid_coordinate_range
        self.current_contig = current_contig

        if not message:
            if self.problem_with_start:
                message = 'Start'
            else:
                message = 'End'
            message += ' coordinate {} is too '.format(problematic_coordinate)

            if self.coordinate_too_small:
                message += 'small.'
            else:
                message += 'big.'

            message += ' Valid coordinates: [{}, {}).'.format(valid_coordinate_range[0], valid_coordinate_range[1])
        super().__init__(message)


class InvalidCoordinates(Exception):
    def __init__(self, start, end, message=''):
        """
        Handles all coordinate problems not captured by CoordinateOutOfBounds.
        Currently only implements the situation where the start is after the end.
        """
        self.start = start
        self.end = end
        if not message:
            message = 'Start coordinate >= end coordinate.'
        super().__init__(message)


class InvalidStrandValue(Exception):
    def __init__(self, requested_strand, valid_strands=(-1, 1), message=''):
        """
        Exception indicating that an invalid strand value has been passed.
        """
        if not message:
            message = 'Invalid strand: {}. Allowable values: {}.'.format(requested_strand, ', '.join(valid_strands))
        super().__init__(message)

        # TODO: consider deleting
        self.requested_strand = requested_strand


def validate_coordinates(start, end, current_contig, valid_range):
    """
    Convenience function to automatically check for problems with coordinates and raise the appropriate exceptions.
    """
    if start < valid_range[0]:
        raise CoordinateOutOfBounds(problematic_coordinate=start,
                                    problem_with_start=True,
                                    coordinate_too_small=True,
                                    valid_coordinate_range=valid_range,
                                    current_contig=current_contig)
    if start > valid_range[1]:
        raise CoordinateOutOfBounds(problematic_coordinate=start,
                                    problem_with_start=True,
                                    coordinate_too_small=False,
                                    valid_coordinate_range=valid_range,
                                    current_contig=current_contig)
    if end < valid_range[0]:
        raise CoordinateOutOfBounds(problematic_coordinate=end,
                                    problem_with_start=False,
                                    coordinate_too_small=True,
                                    valid_coordinate_range=valid_range,
                                    current_contig=current_contig)
    if end > valid_range[1]:
        raise CoordinateOutOfBounds(problematic_coordinate=end,
                                    problem_with_start=False,
                                    coordinate_too_small=False,
                                    valid_coordinate_range=valid_range,
                                    current_contig=current_contig)
    if end <= start:
        raise InvalidCoordinates(start=start, end=end)


# ======================================================================================================================
# Test
# ======================================================================================================================
def test():
    """
    Run benchmarks if module is executed directly.
    :return:
    """
    d = GenomicsInterface()

    # TODO: resolve invalid argument of group_queries
    indv_random_access_time = d.benchmark_random_access(sample_size=297)
    contiguous_acccess_time = d.benchmark_contiguous_access()
    verbose_print('Random access (individual queries): {}'.format(indv_random_access_time))
    verbose_print('Contiguous access: {}'.format(contiguous_acccess_time))


if __name__ == '__main__':
    test()
