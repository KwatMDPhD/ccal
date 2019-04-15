from numpy import (
    absolute,
    asarray,
    concatenate,
    cumsum,
    full,
    inf,
    linspace,
    minimum,
    nan,
)
from statsmodels.sandbox.distributions.extras import ACSkewT_gen

from .check_nd_array_for_bad import check_nd_array_for_bad
from .compute_kullback_leibler_divergence_between_2_pdfs import (
    compute_kullback_leibler_divergence_between_2_pdfs,
)
from .fit_skew_t_pdf_on_1d_array import fit_skew_t_pdf_on_1d_array
from .make_coordinates_for_reflection import make_coordinates_for_reflection


def _compute_pdf_context(
    grid, pdf, pdf_reference, multiply_distance_from_reference_argmax
):

    center = pdf_reference.argmax()

    left_kl = compute_kullback_leibler_divergence_between_2_pdfs(
        pdf[:center], pdf_reference[:center]
    )

    right_kl = compute_kullback_leibler_divergence_between_2_pdfs(
        pdf[center:], pdf_reference[center:]
    )

    left_kl[left_kl == inf] = 0

    right_kl[right_kl == inf] = 0

    left_context = -cumsum((left_kl / left_kl.sum())[::-1])[::-1]

    right_context = cumsum(right_kl / right_kl.sum())

    left_context *= left_kl.sum() / left_kl.size

    right_context *= right_kl.sum() / right_kl.size

    context = concatenate((left_context, right_context))

    if multiply_distance_from_reference_argmax:

        context *= absolute(grid - grid[center])

    return context


def compute_1d_array_context(
    _1d_array,
    n_data=None,
    location=None,
    scale=None,
    degree_of_freedom=None,
    shape=None,
    fit_initial_location=None,
    fit_initial_scale=None,
    n_grid=1e3,
    degree_of_freedom_for_tail_reduction=1e8,
    multiply_distance_from_reference_argmax=False,
    global_location=None,
    global_scale=None,
    global_degree_of_freedom=None,
    global_shape=None,
):

    is_bad = check_nd_array_for_bad(_1d_array, raise_for_bad=False)

    _1d_array_good = _1d_array[~is_bad]

    if any(
        parameter is None
        for parameter in (n_data, location, scale, degree_of_freedom, shape)
    ):

        n_data, location, scale, degree_of_freedom, shape = fit_skew_t_pdf_on_1d_array(
            _1d_array_good,
            fit_initial_location=fit_initial_location,
            fit_initial_scale=fit_initial_scale,
        )

    grid = linspace(_1d_array_good.min(), _1d_array_good.max(), n_grid)

    skew_t_model = ACSkewT_gen()

    pdf = skew_t_model.pdf(grid, degree_of_freedom, shape, loc=location, scale=scale)

    shape_pdf_reference = minimum(
        pdf,
        skew_t_model.pdf(
            make_coordinates_for_reflection(grid, grid[pdf.argmax()]),
            degree_of_freedom_for_tail_reduction,
            shape,
            loc=location,
            scale=scale,
        ),
    )

    shape_context = _compute_pdf_context(
        grid, pdf, shape_pdf_reference, multiply_distance_from_reference_argmax
    )

    if any(
        parameter is None
        for parameter in (
            global_location,
            global_scale,
            global_degree_of_freedom,
            global_shape,
        )
    ):

        location_pdf_reference = None

        location_context = None

        context = shape_context

    else:

        location_pdf_reference = minimum(
            pdf,
            skew_t_model.pdf(
                grid,
                global_degree_of_freedom,
                global_shape,
                loc=global_location,
                scale=global_scale,
            ),
        )

        location_context = _compute_pdf_context(
            grid, pdf, location_pdf_reference, multiply_distance_from_reference_argmax
        )

        context = shape_context + location_context

    context_like_array = full(_1d_array.size, nan)

    context_like_array[~is_bad] = context[
        [absolute(grid - value).argmin() for value in _1d_array_good]
    ]

    return {
        "fit": asarray((n_data, location, scale, degree_of_freedom, shape)),
        "grid": grid,
        "pdf": pdf,
        "shape_pdf_reference": shape_pdf_reference,
        "shape_context": shape_context,
        "location_pdf_reference": location_pdf_reference,
        "location_context": location_context,
        "context": context,
        "context_like_array": context_like_array,
    }
