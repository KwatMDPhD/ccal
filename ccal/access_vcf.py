from gzip import open as gzip_open
from warnings import warn

from tabix import open as tabix_open

VCF_COLUMNS = ("CHROM", "POS", "ID", "REF", "ALT", "QUAL", "FILTER", "INFO", "FORMAT")

BAD_IDS = (".",)

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

VARIANT_EFFECTS = (
    # Loss of transcript or exon
    "transcript_ablation",
    "exon_loss_variant",
    # Altered splicing
    "splice_acceptor_variant",
    "splice_donor_variant",
    # Nonsense mutation
    "stop_gained",
    # Frameshift
    "frameshift_variant",
    # Nonstop mutation 1
    "stop_lost",
    # Nonstart mutation
    "start_lost",
    "initiator_codon_variant",
    # Altered transcript 1
    "transcript_amplification",
    "protein_protein_contact",
    "transcript_variant",
    # InDel
    "disruptive_inframe_insertion",
    "disruptive_inframe_deletion",
    "inframe_insertion",
    "inframe_deletion",
    # Altered transcript 2
    "conservative_missense_variant",
    "rare_amino_acid_variant",
    "missense_variant",
    "protein_altering_variant",
    # Altered intragenic region 1
    "splice_region_variant",
    # Nonstop mutation 2
    "incomplete_terminal_codon_variant",
    # Silent mutation
    "start_retained_variant",
    "stop_retained_variant",
    "synonymous_variant",
    # Mutation
    "coding_sequence_variant",
    "exon_variant",
    # Altered miRNA
    "mature_miRNA_variant",
    # Altered 5'UTR
    "5_prime_UTR_variant",
    "5_prime_UTR_premature_start_codon_gain_variant",
    # Altered 3'UTR
    "3_prime_UTR_variant",
    # Altered non-coding exon region
    "non_coding_exon_variant",
    "non_coding_transcript_exon_variant",
    # Altered intragenic region 2
    "intragenic_variant",
    "conserved_intron_variant",
    "intron_variant",
    "INTRAGENIC",
    # Altered nonsense-mediated-decay-target region
    "NMD_transcript_variant",
    # Altered non-coding region
    "non_coding_transcript_variant",
    "nc_transcript_variant",
    # Altered 5'flank site
    "upstream_gene_variant",
    # Altered 3'flank site
    "downstream_gene_variant",
    # Altered transcription-factor-binding region
    "TF_binsing_site_ablation",
    "TFBS_ablation",
    "TF_binding_site_amplification",
    "TFBS_amplification",
    "TF_binding_site_variant",
    "TFBS_variant",
    # Altered regulatory region
    "regulatory_region_ablation",
    "regulatory_region_amplification",
    "regulatory_region_variant",
    "regulatory_region",
    "feature_elongation",
    "feature_truncation",
    # Altered intergenic region
    "conserved_intergenic_variant",
    "intergenic_variant",
    "intergenic_region",
    # Others
    "sequence_feature",
)


def get_variants_from_vcf_gz(
    chromosome, start_position, end_position, pytabix_handle=None, vcf_gz_file_path=None
):

    if pytabix_handle is None:

        if vcf_gz_file_path is None:

            raise ValueError("Provide either pytabix_handle or vcf_gz_file_path.")

        else:

            pytabix_handle = tabix_open(vcf_gz_file_path)

    variants = pytabix_handle.query(chromosome, start_position, end_position)

    varinat_dicts = [
        parse_vcf_row_and_make_variant_dict(variant) for variant in variants
    ]

    for variant_dict in varinat_dicts:

        update_variant_dict(variant_dict)

    return varinat_dicts


def parse_vcf_row_and_make_variant_dict(vcf_row, n_info_ann=None):

    variant_dict = {
        column: vcf_row[i]
        for i, column in enumerate(VCF_COLUMNS[: VCF_COLUMNS.index("FILTER") + 1])
    }

    info_without_field = []

    for info in vcf_row[VCF_COLUMNS.index("INFO")].split(sep=";"):

        if "=" in info:

            info_field, info_value = info.split(sep="=")

            if info_field == "ANN":

                variant_dict["ANN"] = {}

                for ann_index, ann in enumerate(info_value.split(sep=",")[:n_info_ann]):

                    ann_values = ann.split(sep="|")

                    variant_dict["ANN"][ann_index] = {
                        ann_field: ann_values[ann_field_index + 1]
                        for ann_field_index, ann_field in enumerate(VCF_ANN_FIELDS[1:])
                    }

            else:

                variant_dict[info_field] = info_value

        else:

            info_without_field.append(info)

    if len(info_without_field):

        variant_dict["INFO_without_field"] = ";".join(info_without_field)

    vcf_column_format_index = VCF_COLUMNS.index("FORMAT")

    format_fields = vcf_row[vcf_column_format_index].split(sep=":")

    variant_dict["sample"] = {}

    for sample_index, sample in enumerate(vcf_row[vcf_column_format_index + 1 :]):

        variant_dict["sample"][sample_index] = {
            format_field: sample_value
            for format_field, sample_value in zip(format_fields, sample.split(sep=":"))
        }

    return variant_dict


def update_variant_dict(variant_dict):

    ref = variant_dict["REF"]

    alt = variant_dict["ALT"]

    variant_dict["variant_type"] = get_variant_type(ref, alt)

    start_position, end_position = get_variant_start_and_end_positions(
        int(variant_dict["POS"]), ref, alt
    )

    variant_dict["start_position"] = start_position

    variant_dict["end_position"] = end_position

    caf = variant_dict.get("CAF")

    if caf:

        variant_dict[
            "population_allelic_frequencies"
        ] = get_population_allelic_frequencies(caf)

    for ann_dict in variant_dict["ANN"].values():

        ann_dict["variant_classification"] = get_maf_variant_classification(
            ann_dict["effect"], ref, alt
        )

    for sample_dict in variant_dict["sample"].values():

        if "GT" in sample_dict:

            sample_dict["genotype"] = get_genotype(ref, alt, sample_dict["GT"])

        if "AD" in sample_dict and "DP" in sample_dict:

            sample_dict["allelic_frequency"] = get_allelic_frequencies(
                sample_dict["AD"], sample_dict["DP"]
            )


def count_gene_impacts_from_variant_dicts(variant_dicts, gene_name):

    impact_counts = {"HIGH": 0, "MODERATE": 0, "LOW": 0, "MODIFIER": 0}

    for variant_dict in variant_dicts:

        if variant_dict["gene_name"] == gene_name:

            impact_counts[variant_dict["impact"]] += 1

    return impact_counts


def get_vcf_info(info, field):

    for info_ in info.split(sep=";"):

        if "=" in info_:

            info_field, info_value = info_.split(sep="=")

            if info_field == field:

                return info_value


def get_vcf_info_ann(info, field, n_ann=None):

    ann = get_vcf_info(info, "ANN")

    if ann:

        field_index = VCF_ANN_FIELDS.index(field)

        return [ann_.split(sep="|")[field_index] for ann_ in ann.split(sep=",")[:n_ann]]

    else:

        return []


def get_vcf_sample_format(format_, sample, format_field):

    return sample.split(sep=":")[format_.split(sep=":").index(format_field)]


def get_variant_start_and_end_positions(pos, ref, alt):

    if len(ref) == len(alt):

        start_position, end_position = pos, pos + len(alt) - 1

    elif len(ref) < len(alt):

        start_position, end_position = pos, pos + 1

    else:

        start_position, end_position = pos + 1, pos + len(ref) - len(alt)

    return start_position, end_position


def get_variant_type(ref, alt):

    if len(ref) == len(alt):

        if len(ref) == 1:

            variant_type = "SNP"

        elif len(ref) == 2:

            variant_type = "DNP"

        elif len(ref) == 3:

            variant_type = "TNP"

        else:

            variant_type = "ONP"

    elif len(ref) < len(alt):

        variant_type = "INS"

    else:

        variant_type = "DEL"

    return variant_type


def is_inframe(ref, alt):

    return not ((len(ref) - len(alt)) % 3)


def get_maf_variant_classification(effect, ref, alt):

    variant_type = get_variant_type(ref, alt)

    inframe = is_inframe(ref, alt)

    if effect in (
        "transcript_ablation",
        "exon_loss_variant",
        "splice_acceptor_variant",
        "splice_donor_variant",
    ):

        variant_classification = "Splice_Site"

    elif effect in ("stop_gained",):

        variant_classification = "Nonsense_Mutation"

    elif variant_type == "INS" and (
        effect in ("frameshift_variant",)
        or (
            not inframe
            and effect in ("protein_protein_contact", "protein_altering_variant")
        )
    ):

        variant_classification = "Frame_Shift_Ins"

    elif variant_type == "DEL" and (
        effect in ("frameshift_variant",)
        or (
            not inframe
            and effect in ("protein_protein_contact", "protein_altering_variant")
        )
    ):

        variant_classification = "Frame_Shift_Del"

    elif effect in ("stop_lost",):

        variant_classification = "Nonstop_Mutation"

    elif effect in ("start_lost", "initiator_codon_variant"):

        variant_classification = "Translation_Start_Site"

    elif (
        variant_type == "INS"
        and inframe
        and effect
        in (
            "protein_protein_contact",
            "disruptive_inframe_insertion",
            "inframe_insertion",
            "protein_altering_variant",
        )
    ):

        variant_classification = "In_Frame_Ins"

    elif (
        variant_type == "DEL"
        and inframe
        and effect
        in (
            "protein_protein_contact",
            "disruptive_inframe_deletion",
            "inframe_deletion",
            "protein_altering_variant",
        )
    ):

        variant_classification = "In_Frame_Del"

    elif effect in (
        "transcript_variant",
        "conservative_missense_variant",
        "rare_amino_acid_variant",
        "missense_variant",
        "coding_sequence_variant",
    ) or (
        variant_type not in ("INS", "DEL") and effect in ("protein_protein_contact",)
    ):

        variant_classification = "Missense_Mutation"

    elif effect in (
        "transcript_amplification",
        "splice_region_variant",
        "intragenic_variant",
        "conserved_intron_variant",
        "intron_variant",
        "INTRAGENIC",
    ):

        variant_classification = "Intron"

    elif effect in (
        "incomplete_terminal_codon_variant",
        "start_retained_variant",
        "stop_retained_variant",
        "synonymous_variant",
        "NMD_transcript_variant",
    ):

        variant_classification = "Silent"

    elif effect in (
        "exon_variant",
        "mature_miRNA_variant",
        "non_coding_exon_variant",
        "non_coding_transcript_exon_variant",
        "non_coding_transcript_variant",
        "nc_transcript_variant",
    ):

        variant_classification = "RNA"

    elif effect in (
        "5_prime_UTR_variant",
        "5_prime_UTR_premature_start_codon_gain_variant",
    ):

        variant_classification = "5'UTR"

    elif effect in ("3_prime_UTR_variant",):

        variant_classification = "3'UTR"

    elif effect in (
        "TF_binding_site_ablation",
        "TFBS_ablation",
        "TF_binding_site_amplification",
        "TFBS_amplification",
        "TF_binding_site_variant",
        "TFBS_variant",
        "regulatory_region_ablation",
        "regulatory_region_amplification",
        "regulatory_region_variant",
        "regulatory_region",
        "feature_elongation",
        "feature_truncation",
        "conserved_intergenic_variant",
        "intergenic_variant",
        "intergenic_region",
    ):

        variant_classification = "IGR"

    elif effect in ("upstream_gene_variant",):

        variant_classification = "5'Flank"

    elif effect in ("downstream_gene_variant",):

        variant_classification = "3'Flank"

    elif effect in ("sequence_feature",):

        variant_classification = "Targeted_Region"

    else:

        warn(
            "No variant classification for: effect={} & variant_type={} & inframe={}.".format(
                effect, variant_type, inframe
            )
        )

        variant_classification = "Targeted_Region"

    return variant_classification


def get_genotype(ref, alt, gt):

    return [
        ([ref] + alt.split(sep=","))[int(allelic_index)]
        for allelic_index in gt.replace("/", "|").split(sep="|")
    ]


def get_allelic_frequencies(ad, dp):

    return [int(allelic_depth) / int(dp) for allelic_depth in ad.split(sep=",")]


def get_population_allelic_frequencies(caf):

    try:

        return [
            float(population_allelic_frequency)
            for population_allelic_frequency in caf.split(sep=",")
        ]

    except ValueError:

        warn("Bad CAF {}.".format(caf))

        return [
            float(population_allelic_frequency)
            for population_allelic_frequency in caf.split(sep=",")
            if population_allelic_frequency and population_allelic_frequency != "."
        ]


def count_vcf_gz_rows(
    vcf_gz_file_path, info_fields_to_count=None, format_fields_to_count=None
):

    fields = ()

    if info_fields_to_count is not None:

        fields += info_fields_to_count

    if format_fields_to_count is not None:

        fields += format_fields_to_count

    counts = {field: 0 for field in fields}

    print("Validating {} ...".format(vcf_gz_file_path))

    with gzip_open(vcf_gz_file_path) as vcf_gz_file:

        line = vcf_gz_file.readline().decode()

        while line.startswith("##"):

            line = vcf_gz_file.readline().decode()

        else:

            if not line.startswith("#CHROM"):

                raise ValueError(
                    "The line follwoing the meta-information lines ({}) is not the column header.".format(
                        line
                    )
                )

            elif len(line.split(sep="\t")) < 10:

                raise ValueError(
                    "Column header does not contain all of {} and at least 1 sample.".format(
                        ", ".join(VCF_COLUMNS), line
                    )
                )

            elif 10 < len(line.split(sep="\t")):

                raise NotImplementedError(
                    "There are 1< samples and multi-sample .vcf file is not supported yet."
                )

        for i, line in enumerate(vcf_gz_file):

            if i % 1e5 == 0:

                print("\t{:,} ...".format(i))

            line_split = line.decode().split(sep="\t")

            chrom, pos, id_, ref, alt, qual, filter_, info, format_, sample = line_split

            if info_fields_to_count is not None:

                info_fields = [
                    field_value.split(sep="=")[0] for field_value in info.split(sep=";")
                ]

                for field in info_fields_to_count:

                    if field in info_fields:

                        counts[field] += 1

            if format_fields_to_count is not None:

                format_fields = format_.split(sep=":")

                for field in format_fields_to_count:

                    if field in format_fields:

                        counts[field] += 1

    counts["n"] = i + 1

    return counts
