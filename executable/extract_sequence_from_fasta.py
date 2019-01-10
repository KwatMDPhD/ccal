from click import argument, command


@command()
@argument("fasta_file_path")
@argument("pattern")
def extract_sequence_from_fasta(fasta_file_path, pattern):

    with open(fasta_file_path) as fasta_file:

        lines = []

        in_sequence = False

        for line in fasta_file:

            if line.startswith(">"):

                in_sequence = pattern in line

            if in_sequence:

                lines.append(line)

    print("".join(lines).strip())


if __name__ == "__main__":

    extract_sequence_from_fasta()
