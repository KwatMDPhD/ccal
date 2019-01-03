from warnings import warn

from numpy import absolute, concatenate, cumsum, finfo, full, log, nan, zeros_like


def _compute_context_indices(
    grid,
    pdf,
    pdf_reference,
    minimum_kl,
    scale_with_kl,
    multiply_distance_from_reference_argmax,
):

    eps = finfo(float).eps

    pdf = pdf.clip(min=eps)

    pdf_reference = pdf_reference.clip(min=eps)

    center = pdf_reference.argmax()

    if center == 0 or center == (grid.size - 1):

        warn("PDF reference is monotonic.")

        return full(grid.size, nan)

    left_kl = pdf[:center] * log(pdf[:center] / pdf_reference[:center])

    right_kl = pdf[center:] * log(pdf[center:] / pdf_reference[center:])

    if left_kl.sum() / left_kl.size < minimum_kl:

        left_context_indices = zeros_like(left_kl)

    else:

        left_context_indices = -cumsum((left_kl / left_kl.sum())[::-1])[::-1]

    if right_kl.sum() / right_kl.size < minimum_kl:

        right_context_indices = zeros_like(right_kl)

    else:

        right_context_indices = cumsum(right_kl / right_kl.sum())

    if scale_with_kl:

        left_context_indices *= left_kl.sum() / left_kl.size

        right_context_indices *= right_kl.sum() / right_kl.size

    context_indices = concatenate((left_context_indices, right_context_indices))

    if multiply_distance_from_reference_argmax:

        context_indices *= absolute(grid - grid[center])

    return context_indices
