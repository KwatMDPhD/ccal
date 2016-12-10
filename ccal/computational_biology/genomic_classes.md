# Module: genomic_tools 

The genomic_tools module contains a variety of functions for working with sequence data, as well as several classes that can be used to interface with genomic data stored in a variety of formats.

## Class: DataInterface
Of special interest to the end user is the DataInterface class, which implements methods that implement commonly-used queries in an integrative fashion without having to know the low-level details of how the underlying data is stored.

### Initialization

`genomic_tools.DataInterface(sequence_directory=SEQUENCE_DIRECTORY,
                 feature_filename=FEATURE_FILENAME,
                 vcf_filename=VARIANT_FILENAME,
                 feature_pickle_filename=FEATURE_PICKLE_FILENAME,
                 variant_pickle_filename=VARIANT_PICKLE_FILENAME,
                 chromosome_dialect=CHROMOSOME_DIALECT,
                 species='human',
                 genome_build=GENOME_ASSEMBLY,
                 force_rebuild=False)`
                 
Parameters: These should all be left to the defaults, so you can simply initialize with:

`genomic_tools.DataInterface()`

### Properties

`DataInterface.contig_names(self)`

A list of valid contig names for this genome.

`DataInterface.contig_lengths(self)`

A dictionary of contig lengths keyed by contig name.

### Methods

`DataInterface.get_variant_by_id(self, rs_id)`

Return a dictionary of attributes for a single sequence variant identified by `rs_id`

`DataInterface.get_variants_by_location(self, contig_name, start, end)`

Return a list of variant tuples, sorted by genomic position, that overlap the genomic region specified by `contig_name`, `start`, `end`

`DataInterface.get_variants_by_gene_name(self, gene_name)`

Return a list of variant dictionaries, sorted by genomic position, that fall within (or overlap) the boundaries of the gene specified by `gene_name`

`DataInterface.get_features_by_location(self, contig_name, start, end)`

Return a list of genomic features (e.g. genes), sorted by genomic position, that overlap the genomic region specified by `contig_name`, `start`, `end`

`DataInterface.get_gene_by_ensemblid(self, ensembl_id)`

Return a dictionary of attributes for the gene corresponding to `ensembl_id`

`DataInterface.get_gene_by_name(self, gene_name)`

Return a dictionary of attributes for the gene corresponding to `gene_name`

`DataInterface.get_reference_sequence_by_location(self, contig_name, start, end)`

Return a list of characters representing the DNA sequence in the reference genome for the genomic region specified by `contig_name`, `start`, `end`

`DataInterface.get_client_sequence_by_location(self, contig_name, start, end)`

Return a list of characters representing the DNA sequence in the client's genome for the genomic region specified by `contig_name`, `start`, `end`

## Variant dictionaries

Individual sequence variants are represented as dictionaries of field-value pairs. Currently the following fields are defined (this will change in future versions):

* rs_id (str): The RS_ID of the variant
* contig (str): The sequence contig where the variant occurs 
* start (int): The reference sequence coordinate where the variant starts
* end (int): The reference sequence coordinate where the variant ends
* qual (float): A number representing the quality of the variant prediction (higher numbers are more likely to be real and not artifacts)
* affected_area (str): The functional classification of the genomic region where the variant occurs (intron, intergenic, etc.)
* putative_impact (MODIFIER, LOW, MODERATE, HIGH):
* affected_gene (str): The HUGO gene symbol of the gene affected by the variant (if any)
* ref (str): The sequence at the variant site in the reference genome
* alt (str): The sequence at the variant site in the affected genome
* snp_or_indel (snp, indel): One of two strings indicating whether this is a SNP or indel (insertion / deletion) variant. Currently only SNPs are supported and returned. 

## Gene dictionaries

Genes are represented as dictionaries of field-value pairs. Currently the following fields are defined (this will change in future versions):

* contig: The contig / chromosome where the gene is found
* start: The start location of the gene on the genome
* end: The end location of the gene on the genome
* version: Not sure what the source of these version numbers is
* gene_name: The HUGO gene symbol for the gene
* ensembl_id: The Ensembl ID for the gene

The following optional fields may be present if there is an entry for the gene in the supplemental gene info file:

* SUMMARY
* PHENOTYPES
* SYMBOL
* TYPE
* ACTIONABLES
* ID
* FULL_NAME
