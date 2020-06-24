from numpy import asarray, isclose, where

from .information import get_jsd
from .plot import plot_plotly


def _lrc(p):

    assert isclose(p.sum(), 1)

    lc = p.cumsum()

    rc = p[::-1].cumsum()[::-1]

    add = 1e-8

    return lc + add, rc + add


def _hm_1_p_lrc(h_1, m_1, a, plot_process):

    h_1 *= a

    h_1_p = h_1 / h_1.sum()

    m_1_p = m_1 / m_1.sum()

    if plot_process:

        plot_plotly(
            {
                "layout": {"title": {"text": "PDF"}},
                "data": [
                    {"name": "Hit", "y": h_1_p, "mode": "lines"},
                    {"name": "Miss", "y": m_1_p, "mode": "lines"},
                ],
            },
        )

    h_1_p_lc, h_1_p_rc = _lrc(h_1_p)

    m_1_p_lc, m_1_p_rc = _lrc(m_1_p)

    if plot_process:

        plot_plotly(
            {
                "layout": {"title": {"text": "CDF >"}},
                "data": [
                    {"name": "Hit", "y": h_1_p_lc, "mode": "lines"},
                    {"name": "Miss", "y": m_1_p_lc, "mode": "lines"},
                ],
            },
        )

        plot_plotly(
            {
                "layout": {"title": {"text": "< CDF"}},
                "data": [
                    {"name": "Hit", "y": h_1_p_rc, "mode": "lines"},
                    {"name": "Miss", "y": m_1_p_rc, "mode": "lines"},
                ],
            },
        )

    return h_1_p_lc, m_1_p_lc, h_1_p_rc, m_1_p_rc


def _hm_v_p_lrc(h_1, m_1, a, plot_process):

    h_a = h_1 * a

    m_a = m_1 * a

    h_a_p = h_a / h_a.sum()

    m_a_p = m_a / m_a.sum()

    a_p = a / a.sum()

    if plot_process:

        plot_plotly(
            {
                "layout": {"title": {"text": "Magnitude"}},
                "data": [
                    {"name": "Hit", "y": h_a_p, "mode": "lines"},
                    {"name": "Miss", "y": m_a_p, "mode": "lines"},
                    {"name": "All", "y": a_p, "mode": "lines"},
                ],
            },
        )

    h_a_p_lc, h_a_p_rc = _lrc(h_a_p)

    m_a_p_lc, m_a_p_rc = _lrc(m_a_p)

    a_p_lc, a_p_rc = _lrc(a_p)

    if plot_process:

        plot_plotly(
            {
                "layout": {"title": {"text": "CDF >"}},
                "data": [
                    {"name": "Hit", "y": h_a_p_lc, "mode": "lines"},
                    {"name": "Miss", "y": m_a_p_lc, "mode": "lines"},
                    {"name": "All", "y": a_p_lc, "mode": "lines"},
                ],
            },
        )

        plot_plotly(
            {
                "layout": {"title": {"text": "< CDF"}},
                "data": [
                    {"name": "Hit", "y": h_a_p_rc, "mode": "lines"},
                    {"name": "Miss", "y": m_a_p_rc, "mode": "lines"},
                    {"name": "All", "y": a_p_rc, "mode": "lines"},
                ],
            },
        )

    return h_a_p_lc, m_a_p_lc, a_p_lc, h_a_p_rc, m_a_p_rc, a_p_rc


def score_set(
    element_score,
    set_elements,
    method="classic",
    symmetry=False,
    plot_process=False,
    plot=True,
    title="Score Set",
    element_socre_name="Element Score",
    annotation_text_font_size=8,
    annotation_text_width=160,
    annotation_text_yshift=32,
    html_file_path=None,
):

    element_score = element_score.sort_values()

    absolutes = element_score.abs().values

    set_elements = {element: None for element in set_elements}

    h_1 = asarray(
        tuple(element in set_elements for element in element_score.index), dtype=float
    )

    m_1 = 1 - h_1

    if method == "classic":

        h_l, m_l, h_r, m_r = _hm_1_p_lrc(h_1, m_1, absolutes, plot_process)

        l_h, l_m, l = h_l, m_l, h_l - m_l

        r_h, r_m, r = h_r, m_r, h_r - m_r

    elif method == "2020":

        h_l, m_l, _l, h_r, m_r, _r = _hm_v_p_lrc(h_1, m_1, absolutes, plot_process)

        l_h, l_m, l = get_jsd(h_l, m_l, vector_reference=_l)

        r_h, r_m, r = get_jsd(h_r, m_r, vector_reference=_r)

    if plot_process:

        opacity = 0.32

        signal_template = {
            "name": "Signal",
            "mode": "lines",
            "marker": {"color": "#ff1968"},
        }

        plot_plotly(
            {
                "layout": {"title": {"text": "Signal >"}},
                "data": [
                    {"name": "Hit", "y": l_h, "opacity": opacity, "mode": "lines"},
                    {"name": "Miss", "y": l_m, "opacity": opacity, "mode": "lines"},
                    {"y": l, **signal_template},
                ],
            },
        )

        plot_plotly(
            {
                "layout": {"title": {"text": "< Signal"}},
                "data": [
                    {"name": "Hit", "y": r_h, "opacity": opacity, "mode": "lines"},
                    {"name": "Miss", "y": r_m, "opacity": opacity, "mode": "lines"},
                    {"y": r, **signal_template},
                ],
            },
        )

    if symmetry:

        s = r - l

    else:

        s = r

    score = s.sum() / s.size

    if plot:

        y_fraction = 0.16

        layout = {
            "title": {
                "text": "{}<br>Score (method={}, symmetry={}) = {:.2f}".format(
                    title, method, symmetry, score
                ),
                "x": 0.5,
            },
            "xaxis": {"anchor": "y"},
            "yaxis": {"domain": (0, y_fraction), "title": element_socre_name},
            "yaxis2": {"domain": (y_fraction + 0.08, 1)},
            "legend_orientation": "h",
            "legend": {"y": -0.24},
        }

        h_i = where(h_1)[0]

        data = [
            {
                "name": "Element Score ({})".format(element_score.size),
                "y": element_score.values,
                "text": element_score.index,
                "mode": "lines",
                "line": {"width": 0, "color": "#20d8ba"},
                "fill": "tozeroy",
            },
            {
                "name": "Element ({})".format(h_i.size),
                "yaxis": "y2",
                "x": h_i,
                "y": (0,) * h_i.size,
                "text": element_score.index[h_i],
                "mode": "markers",
                "marker": {
                    "symbol": "line-ns-open",
                    "size": 8,
                    "color": "#2e211b",
                    "line": {"width": 1.6},
                },
                "hoverinfo": "x+text",
            },
        ]

        for name, is_, color in (
            ("- Enrichment", s < 0, "#0088ff"),
            ("+ Enrichment", 0 < s, "#ff1968"),
        ):

            data.append(
                {
                    "name": name,
                    "yaxis": "y2",
                    "y": where(is_, s, 0),
                    "mode": "lines",
                    "line": {"width": 0, "color": color},
                    "fill": "tozeroy",
                }
            )

        plot_plotly({"layout": layout, "data": data}, html_file_path=html_file_path)

    return score
