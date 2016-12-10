from urllib import request


def get_variant_position(rsid):
    """

    :param rsid:
    :return:
    """

    # TODO: fix InDel position error
    ncbi_snp_link = 'http://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?db=SNP&id='
    for line in request.urlopen(ncbi_snp_link + rsid[2:]).read().decode('UTF-8').splitlines():
        if 'CHRPOS' in line:
            contig, start_position = line.split('>')[1].split('<')[0].split(':')
            return contig + ':' + start_position + '-' + start_position


def get_gene_id(gene):
    """

    :param gene:
    :return:
    """

    ncbi_gene_link = 'http://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=gene&'
    for line in request.urlopen(ncbi_gene_link + 'term=(' + gene + '[gene]+AND+Human[organism])').read().decode(
            'UTF-8').splitlines():
        if '<Id>' in line:
            return line.split('>')[1].split('<')[0]


def get_gene_position(gene):
    """

    :param gene:
    :return:
    """

    ncbi_gene_link = 'http://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?db=gene&'
    gene_id = get_gene_id(gene)
    if gene_id:
        contig = start_position = stop_position = None
        for line in request.urlopen(ncbi_gene_link + 'id=' + gene_id).read().decode('UTF-8').splitlines():
            if '<ChrLoc>' in line:
                contig = line.split('>')[1].split('<')[0]
            if '<ChrStart>' in line:
                start_position = line.split('>')[1].split('<')[0]
            if '<ChrStop>' in line:
                stop_position = line.split('>')[1].split('<')[0]
            if contig and start_position and stop_position:
                if int(start_position) > int(stop_position):
                    return contig + ':' + stop_position + '-' + start_position
                else:
                    return contig + ':' + start_position + '-' + stop_position
