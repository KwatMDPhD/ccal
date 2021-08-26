from multiprocessing import Pool

from numpy import array, full, nan, unique, where
from numpy.random import choice, seed, shuffle
from pandas import DataFrame

from ..array import apply, check_is_extreme
from ..cluster import cluster
from ..constant import RANDOM_SEED, SAMPLE_FRACTION
from ..dictionary import merge
from ..function import ignore_nan_and_apply
from ..plot import plot_plotly
from ..significance import get_margin_of_error, get_p_value_q_value
from ._make_data_annotations import _make_data_annotations
from ._make_target_annotation import _make_target_annotation
from ._process_data import _process_data
from ._process_target import _process_target
from .HEATMAP import HEATMAP
from .LAYOUT import LAYOUT
from .TYPE_COLORSCALE import TYPE_COLORSCALE


def make(
    ta,
    da,
    fu,
    ac=True,
    n_jo=1,
    ra=RANDOM_SEED,
    n_sa=10,
    n_sh=10,
    pl=True,
    n_pl=8,
    tyt="continuous",
    tyd="continuous",
    st=nan,
    title="Function Heat Map",
    pa="",
):

    ro_ = da.index.values

    ta = ta.loc[ta.index.intersection(da.columns)]

    if ac is not None:

        ta.sort_values(ascending=ac, inplace=True)

    co_ = ta.index.values

    da = da.loc[:, co_]

    n_ro = ro_.size

    n_co = co_.size

    taa = ta.values

    daa = da.values

    if callable(fu):

        seed(seed=ra)

        po = Pool(processes=n_jo)

        print("Score ({})...".format(fu.__name__))

        sc_ = array(po.starmap(ignore_nan_and_apply, ([taa, ro, fu] for ro in daa)))

        if 0 < n_sa:

            print("0.95 MoE ({} sample)...".format(n_sa))

            sc_ro_sa = full([n_ro, n_sa], nan)

            n_ch = int(n_co * SAMPLE_FRACTION)

            for ie in range(n_sa):

                ie_ = choice(n_co, size=n_ch, replace=False)

                taar = taa[ie_]

                sc_ro_sa[:, ie] = po.starmap(
                    ignore_nan_and_apply, ([taar, ro, fu] for ro in daa[:, ie_])
                )

            ma_ = array([apply(ro, get_margin_of_error) for ro in sc_ro_sa])

        else:

            ma_ = full(sc_.size, nan)

        if 0 < n_sh:

            print("P-Value and Q-Value ({} shuffle)...".format(n_sh))

            sc_ro_sh = full([n_ro, n_sh], nan)

            taar = taa.copy()

            for ie in range(n_sh):

                shuffle(taar)

                sc_ro_sh[:, ie] = po.starmap(
                    ignore_nan_and_apply, ([taar, ro, fu] for ro in daa)
                )

            pv_, qv_ = get_p_value_q_value(sc_, sc_ro_sh.ravel(), "<>")

        else:

            pv_ = full(sc_.size, nan)

            qv_ = pv_.copy()

        po.terminate()

        fu = DataFrame(
            data=array([sc_, ma_, pv_, qv_]).T,
            index=ro_,
            columns=["Score", "MoE", "P-Value", "Q-Value"],
        )

    else:

        fu = fu.loc[ro_, :]

    fu.sort_values("Score", ascending=False, inplace=True)

    if pa != "":

        fu.to_csv(path_or_buf="{}statistic.tsv".format(pa), sep="\t")

    if pl:

        fua = fu.values

        ro_ = fu.index

        da = da.loc[ro_, :]

        daa = da.values

        if n_pl is not None and n_pl < (n_ro / 2):

            ex_ = check_is_extreme(fua[:, 0], "<>", n_ex=n_pl)

            fua = fua[ex_]

            daa = daa[ex_]

            ro_ = ro_[ex_]

        taap, mit, mat = _process_target(taa, tyt, st)

        daap, mid, mad = _process_data(daa, tyd, st)

        if tyt != "continuous":

            for gr, si in zip(*unique(taap, return_counts=True)):

                if 2 < si:

                    print("Clustering {}...".format(gr))

                    ie_ = where(taap == gr)[0]

                    iec_ = ie_[cluster(daap.T[ie_])[0]]

                    taap[ie_] = taap[iec_]

                    daap[:, ie_] = daap[:, iec_]

                    co_[ie_] = co_[iec_]

        n_ro = daap.shape[0] + 2

        he = 1 / n_ro

        layout = merge(
            {
                "height": max(480, 24 * n_ro),
                "title": {
                    "text": title,
                },
                "yaxis2": {
                    "domain": [1 - he, 1],
                    "showticklabels": False,
                },
                "yaxis": {
                    "domain": [0, 1 - he * 2],
                    "showticklabels": False,
                },
                "annotations": _make_target_annotation(1 - he / 2, ta.name),
            },
            LAYOUT,
        )

        layout["annotations"] += _make_data_annotations(
            1 - he / 2 * 3, True, he, ro_, fua
        )

        if pa != "":

            pa = "{}function_heat_map.html".format(pa)

        plot_plotly(
            {
                "data": [
                    {
                        "yaxis": "y2",
                        "z": taap.reshape([1, -1]),
                        "text": taa.reshape([1, -1]),
                        "x": co_,
                        "zmin": mit,
                        "zmax": mat,
                        "colorscale": TYPE_COLORSCALE[tyt],
                        **HEATMAP,
                    },
                    {
                        "yaxis": "y",
                        "z": daap[::-1],
                        "text": daa[::-1],
                        "y": ro_[::-1],
                        "x": co_,
                        "zmin": mid,
                        "zmax": mad,
                        "colorscale": TYPE_COLORSCALE[tyd],
                        **HEATMAP,
                    },
                ],
                "layout": layout,
            },
            pa=pa,
        )

    return fu
