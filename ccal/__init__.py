from os.path import abspath, dirname

VERSION = "0.9.0"

print("CCAL version {} @ {}".format(VERSION, abspath(__file__)))

DATA_DIRECTORY_PATH = "{}/../data".format(dirname(__file__))

from .compute_mutational_signature_enrichment import (
    compute_mutational_signature_enrichment,
)
from .hierarchical_consensus_cluster_with_ks import (
    hierarchical_consensus_cluster_with_ks,
)
