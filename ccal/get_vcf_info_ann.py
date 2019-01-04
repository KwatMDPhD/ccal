from .get_vcf_info import get_vcf_info

VCF_ANN_FIELDS = (
    "ALT",
    "effect",
    "impact",
    "gene_name",
    "gene_id",
    "feature_type",
    "feature_id",
    "transcript_biotype",
    "rank",
    "hgvsc",
    "hgvsp",
    "cdna_position",
    "cds_position",
    "protein_position",
    "distance_to_feature",
    "error",
)


def get_vcf_info_ann(info, field, n_ann=None):

    ann = get_vcf_info(info, "ANN")

    if ann:

        field_index = VCF_ANN_FIELDS.index(field)

        return [ann_.split(sep="|")[field_index] for ann_ in ann.split(sep=",")[:n_ann]]

    else:

        return []
