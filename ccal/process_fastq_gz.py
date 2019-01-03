from os.path import dirname, isdir, isfile
from sys import platform

from . import DATA_DIRECTORY_PATH
from ._print_and_run_command import _print_and_run_command
from .support.support.get_function_name import get_function_name
from .support.support.multiprocess import multiprocess

GENERAL_BAD_SEQUENCES_FILE_PATH = "{}/general_bad_sequences.fasta".format(
    DATA_DIRECTORY_PATH
)


def _check_fastq_gzs(fastq_gz_file_paths):

    if len(fastq_gz_file_paths) not in (1, 2):

        raise ValueError(
            "fastq_gz_file_paths should contain either 1 (unpaired) or 2 (paired) .fastq.gz file path(s)."
        )


def check_fastq_gzs_using_fastqc(fastq_gz_file_paths, n_job=1, overwrite=False):

    for fastq_gz_file_path in fastq_gz_file_paths:

        html_file_path = "{}_fastqc.html".format(fastq_gz_file_path)

        if not overwrite and isfile(html_file_path):

            raise FileExistsError(html_file_path)

    _print_and_run_command(
        "fastqc --threads {} {}".format(n_job, " ".join(fastq_gz_file_paths))
    )


def trim_fastq_gzs_using_skewer(
    fastq_gz_file_paths,
    forward_bad_sequences_fasta_file_path=GENERAL_BAD_SEQUENCES_FILE_PATH,
    reverse_bad_sequences_fasta_file_path=GENERAL_BAD_SEQUENCES_FILE_PATH,
    snv_error_rate=0,
    indel_error_rate=0,
    overlap_length=12,
    end_quality=30,
    min_length_after_trimming=30,
    remove_n=True,
    n_job=1,
    output_directory_path=None,
    overwrite=False,
):

    _check_fastq_gzs(fastq_gz_file_paths)

    if output_directory_path is None:

        output_directory_path = "{}/{}".format(
            dirname(fastq_gz_file_paths[0]), get_function_name()
        )

    if not output_directory_path.endswith("/"):

        output_directory_path += "/"

    if not overwrite and isdir(output_directory_path):

        raise FileExistsError(output_directory_path)

    additional_arguments = []

    if len(fastq_gz_file_paths) == 1:

        additional_arguments.append("-m tail")

        additional_arguments.append("-k {}".format(overlap_length))

    elif len(fastq_gz_file_paths) == 2:

        additional_arguments.append("-m pe")

        additional_arguments.append(
            "-y {}".format(reverse_bad_sequences_fasta_file_path)
        )

    if remove_n:

        remove_n = "-n"

    else:

        remove_n = ""

    _print_and_run_command(
        "skewer -x {} -r {} -d {} --end-quality {} --min {} {} --output {} --masked-output --excluded-output --threads {} {}".format(
            forward_bad_sequences_fasta_file_path,
            snv_error_rate,
            indel_error_rate,
            end_quality,
            min_length_after_trimming,
            remove_n,
            output_directory_path,
            n_job,
            " ".join(additional_arguments + list(fastq_gz_file_paths)),
        )
    )

    log_file_path = "{}/trimmed.log".format(output_directory_path)

    print("{}:".format(log_file_path))

    with open(log_file_path) as log_file:

        print(log_file.read())

    return multiprocess(
        _gzip_compress,
        (
            (outptu_fastq_file_path,)
            for outptu_fastq_file_path in (
                "{}/trimmed-pair{}.fastq".format(output_directory_path, i)
                for i in (1, 2)
            )
        ),
        n_job=n_job,
    )


def _gzip_compress(file_path):

    _print_and_run_command("gzip --force {}".format(file_path))

    return "{}.gz".format(file_path)


def align_fastq_gzs_using_bwa_mem(
    fastq_gz_file_paths,
    fasta_gz_file_path,
    n_job=1,
    output_bam_file_path=None,
    overwrite=False,
):

    _check_fastq_gzs(fastq_gz_file_paths)

    if not all(
        isfile("{}{}".format(fasta_gz_file_path, file_extension))
        for file_extension in (".bwt", ".pac", ".ann", ".amb", ".sa")
    ):

        _print_and_run_command("bwa index {}".format(fasta_gz_file_path))

    if not isfile("{}.alt".format(fasta_gz_file_path)):

        raise FileNotFoundError(
            "ALT-aware BWA-MEM alignment needs {}.alt.".format(fasta_gz_file_path)
        )

    if output_bam_file_path is None:

        output_bam_file_path = "{}/{}.bam".format(
            dirname(fastq_gz_file_paths[0]), get_function_name()
        )

    if not overwrite and isfile(output_bam_file_path):

        raise FileExistsError(output_bam_file_path)

    _print_and_run_command(
        "bwa mem -t {0} -v 3 {1} {2} | {3}/k8-0.2.3/k8-{4} {3}/bwa-postalt.js {5}.alt | samtools view -Sb --threads {1} > {6}".format(
            n_job,
            fasta_gz_file_path,
            " ".join(fastq_gz_file_paths),
            DATA_DIRECTORY_PATH,
            platform,
            fasta_gz_file_path,
            output_bam_file_path,
        )
    )

    return output_bam_file_path


def align_fastq_gzs_using_hisat2(
    fastq_gz_file_paths,
    fasta_file_path,
    sequence_type,
    n_job=1,
    output_bam_file_path=None,
    overwrite=False,
):

    _check_fastq_gzs(fastq_gz_file_paths)

    if not all(
        isfile("{}.{}.ht2".format(fasta_file_path, i)) for i in (1, 2, 3, 4, 5, 6, 7, 8)
    ):

        _print_and_run_command("hisat2-build {0} {0}".format(fasta_file_path))

    if output_bam_file_path is None:

        output_bam_file_path = "{}/{}.bam".format(
            dirname(fastq_gz_file_paths[0]), get_function_name()
        )

    if not overwrite and isfile(output_bam_file_path):

        raise FileExistsError(output_bam_file_path)

    additional_arguments = []

    if len(fastq_gz_file_paths) == 1:

        additional_arguments.append("-U {}".format(fastq_gz_file_paths[0]))

    elif len(fastq_gz_file_paths) == 2:

        additional_arguments.append("-1 {} -2 {}".format(*fastq_gz_file_paths))

    if sequence_type not in ("DNA", "RNA"):

        raise ValueError("Unknown sequence_type: {}.".format(sequence_type))

    elif sequence_type == "DNA":

        additional_arguments.append("--no-spliced-alignment")

    elif sequence_type == "RNA":

        additional_arguments.append("--dta --dta-cufflinks")

    _print_and_run_command(
        "hisat2 -x {0} --summary-file {1}.summary --threads {2} {3} | samtools view -Sb --threads {2} > {1}".format(
            fasta_file_path, output_bam_file_path, n_job, " ".join(additional_arguments)
        )
    )

    return output_bam_file_path


def count_transcripts_using_kallisto_quant(
    fastq_gz_file_paths,
    fasta_gz_file_path,
    output_directory_path,
    n_bootstrap=0,
    fragment_length=180,
    fragment_length_standard_deviation=20,
    n_job=1,
    overwrite=False,
):

    _check_fastq_gzs(fastq_gz_file_paths)

    fasta_gz_kallisto_index_file_path = "{}.kallisto.index".format(fasta_gz_file_path)

    if not isfile(fasta_gz_kallisto_index_file_path):

        _print_and_run_command(
            "kallisto index --index {} {}".format(
                fasta_gz_kallisto_index_file_path, fasta_gz_file_path
            )
        )

    abundance_file_path = "{}/abundance.tsv".format(output_directory_path)

    if not overwrite and isfile(abundance_file_path):

        raise FileExistsError(abundance_file_path)

    if len(fastq_gz_file_paths) == 1:

        sample_argument = "--single --fragment-length {} --sd {} {}".format(
            fragment_length, fragment_length_standard_deviation, fastq_gz_file_paths[0]
        )

    elif len(fastq_gz_file_paths) == 2:

        sample_argument = "{} {}".format(*fastq_gz_file_paths)

    _print_and_run_command(
        "kallisto quant --index {} --output-dir {} --bootstrap-samples {} --threads {} {}".format(
            fasta_gz_kallisto_index_file_path,
            output_directory_path,
            n_bootstrap,
            n_job,
            sample_argument,
        )
    )

    return output_directory_path
