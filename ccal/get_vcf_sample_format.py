def get_vcf_sample_format(format, sample, format_field):

    return sample.split(sep=":")[format.split(sep=":").index(format_field)]
