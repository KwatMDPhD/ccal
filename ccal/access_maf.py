from pandas import DataFrame, read_table

from .access_vcf import (
    get_maf_variant_classification,
    get_variant_start_and_end_positions,
    get_variant_type,
    get_vcf_info_ann,
)
from .access_vcf_dict import read_vcf_gz_and_make_vcf_dict
from .path import establish_path

VARIANT_CLASSIFICATION_MUTSIG_EFFECT = {
    "3'-UTR": "noncoding",
    "3'Flank": "noncoding",
    "3'Promoter": "noncoding",
    "3'UTR": "noncoding",
    "5'-Flank": "noncoding",
    "5'-UTR": "noncoding",
    "5'Flank": "noncoding",
    "5'Promoter": "noncoding",
    "5'UTR": "noncoding",
    "De_novo_Start": "null",
    "De_novo_Start_InFrame": "null",
    "De_novo_Start_OutOfFrame": "null",
    "Frame_Shift_Del": "null",
    "Frame_Shift_Ins": "null",
    "IGR": "noncoding",
    "In_Frame_Del": "null",
    "In_Frame_Ins": "null",
    "Intron": "noncoding",
    "Missense": "nonsilent",
    "Missense_Mutation": "nonsilent",
    "NCSD": "noncoding",
    "Non-coding_Transcript": "noncoding",
    "Nonsense": "null",
    "Nonsense_Mutation": "null",
    "Nonstop_Mutation": "null",
    "Promoter": "noncoding",
    "RNA": "noncoding",
    "Read-through": "null",
    "Silent": "silent",
    "Splice": "null",
    "Splice_Region": "null",
    "Splice_Site": "null",
    "Splice_Site_DNP": "null",
    "Splice_Site_Del": "null",
    "Splice_Site_Ins": "null",
    "Splice_Site_ONP": "null",
    "Splice_Site_SNP": "null",
    "Start_Codon_DNP": "null",
    "Start_Codon_Del": "null",
    "Start_Codon_Ins": "null",
    "Start_Codon_ONP": "null",
    "Stop_Codon_DNP": "null",
    "Stop_Codon_Del": "null",
    "Stop_Codon_Ins": "null",
    "Synonymous": "silent",
    "Targeted_Region": "silent",
    "Translation_Start_Site": "null",
    "Variant_Classification": "effect",
    "downstream": "noncoding",
    "miRNA": "noncoding",
    "upstream": "noncoding",
    "upstream;downstream": "noncoding",
}


def split_maf_by_tumor_sample_barcode(maf_file_path):

    maf_df = read_table(maf_file_path, comment="#", encoding="ISO-8859-1")

    for i, maf_df_ in maf_df.groupby("Tumor_Sample_Barcode"):

        output_directory_path = "{}.split_maf_by_tumor_sample_barcode".format(
            maf_file_path
        )

        establish_path(output_directory_path, "directory")

        maf_df_.to_csv("{}/{}.maf".format(output_directory_path, i), sep="\t", index=0)


def make_maf_from_vcf(
    vcf_file_path, ensg_entrez, sample_name="Sample", maf_file_path=None
):

    vcf_df = read_vcf_gz_and_make_vcf_dict(vcf_file_path, simplify=False)["vcf_df"]

    maf_df = DataFrame(
        index=vcf_df.index,
        columns=(
            "Hugo_Symbol",
            "Entrez_Gene_Id",
            "Center",
            "NCBI_Build",
            "Chromosome",
            "Start_Position",
            "End_Position",
            "Strand",
            "Variant_Classification",
            "Variant_Type",
            "Reference_Allele",
            "Tumor_Seq_Allele1",
            "Tumor_Seq_Allele2",
            "dbSNP_RS",
            "dbSNP_Val_Status",
            "Tumor_Sample_Barcode",
            "Matched_Norm_Sample_Barcode",
            "Matched_Norm_Seq_Allele1",
            "Matched_Norm_Seq_Allele2",
            "Tumor_Validation_Allele1",
            "Tumor_Validation_Allele2",
            "Match_Norm_Validation_Allele1",
            "Match_Norm_Validation_Allele2",
            "Verification_Status",
            "Validation_Status",
            "Mutation_Status",
            "Sequencing_Phase",
            "Sequence_Source",
            "Validation_Method",
            "Score",
            "BAM_File",
            "Sequencer",
            "Tumor_Sample_UUID",
            "Matched_Norm_Sample_UUID",
        ),
    )

    ensg_entrez_dict = read_table(ensg_entrez, index_col=0, squeeze=True).to_dict()

    print("Iterating through .vcf file DataFrame rows ...")

    n = vcf_df.shape[0]

    n_per_print = max(1, n // 10)

    for i, row in vcf_df.iterrows():

        if i % n_per_print == 0:

            print("\t{}/{} ...".format(i + 1, n))

        chrom, pos, id_, ref, alt, info = row[[0, 1, 2, 3, 4, 7]]

        gene_name = get_vcf_info_ann(info, "gene_name")[0]

        gene_id = get_vcf_info_ann(info, "gene_id")[0]

        entrez_gene_id = ensg_entrez_dict.get(gene_id)

        effect = get_vcf_info_ann(info, "effect")[0]

        variant_classification = get_maf_variant_classification(effect, ref, alt)

        start_position, end_position = get_variant_start_and_end_positions(
            int(pos), ref, alt
        )

        variant_type = get_variant_type(ref, alt)

        maf_df.loc[
            i,
            [
                "Hugo_Symbol",
                "Entrez_Gene_Id",
                "Chromosome",
                "Start_Position",
                "End_Position",
                "Variant_Classification",
                "Variant_Type",
                "dbSNP_RS",
                "Reference_Allele",
                "Tumor_Seq_Allele1",
            ],
        ] = (
            gene_name,
            entrez_gene_id,
            chrom,
            start_position,
            end_position,
            variant_classification,
            variant_type,
            id_,
            ref,
            alt,
        )

    maf_df["Strand"] = "+"

    maf_df[
        [
            "Tumor_Sample_Barcode",
            "Matched_Norm_Sample_Barcode",
            "Tumor_Sample_UUID",
            "Matched_Norm_Sample_UUID",
        ]
    ] = sample_name

    if maf_file_path is not None:

        maf_df.to_csv(maf_file_path, sep="\t", index=None)

    return maf_df
