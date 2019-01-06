from numpy import finfo, log

eps = finfo(float).eps


def compute_kullback_leibler_divergence(pdf_0, pdf_1):

    pdf_0[pdf_0 < eps] = eps

    pdf_1[pdf_1 < eps] = eps

    return pdf_0 * log(pdf_0 / pdf_1)
