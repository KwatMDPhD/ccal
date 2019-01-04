from warnings import warn

from statsmodels.sandbox.distributions.extras import ACSkewT_gen

from .check_nd_array_for_bad import check_nd_array_for_bad


def fit_skew_t_pdf(
    _1d_array,
    fit_fixed_location=None,
    fit_fixed_scale=None,
    fit_initial_location=None,
    fit_initial_scale=None,
):

    _1d_array = _1d_array[~check_nd_array_for_bad(_1d_array, raise_for_bad=False)]

    keyword_arguments = {}

    guessed_location = _1d_array.mean()

    guessed_scale = _1d_array.std() / 2

    if fit_fixed_location is not None:

        keyword_arguments["floc"] = fit_fixed_location

    if fit_fixed_scale is not None:

        keyword_arguments["fscale"] = fit_fixed_scale

    if fit_initial_location is not None:

        keyword_arguments["loc"] = fit_initial_location

    else:

        keyword_arguments["loc"] = guessed_location

    if fit_initial_scale is not None:

        keyword_arguments["scale"] = fit_initial_scale

    else:

        keyword_arguments["scale"] = guessed_scale

    skew_t_model = ACSkewT_gen()

    degree_of_freedom, shape, location, scale = skew_t_model.fit(
        _1d_array, **keyword_arguments
    )

    if 24 < abs(shape):

        warn("Refitting with scale = (standard deviation / 2) ...")

        keyword_arguments["fscale"] = guessed_scale

        degree_of_freedom, shape, location, scale = skew_t_model.fit(
            _1d_array, **keyword_arguments
        )

        if 24 < abs(shape):

            warn("Refitting with location = mean ...")

            keyword_arguments["floc"] = guessed_location

            degree_of_freedom, shape, location, scale = skew_t_model.fit(
                _1d_array, **keyword_arguments
            )

    return _1d_array.size, location, scale, degree_of_freedom, shape
