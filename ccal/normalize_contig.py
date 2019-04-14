def normalize_contig(contig, format):

    contig = str(contig)

    if format == "chr":

        if not contig.startswith("chr"):

            contig = "chr{}".format(contig).replace("MT", "M")

    elif format == "":

        if contig.startswith("chr"):

            contig = contig[3:].replace("M", "MT")

    return contig
