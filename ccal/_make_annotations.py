def _make_annotations(score_moe_p_value_fdr):

    annotations = score_moe_p_value_fdr.applymap(lambda str_: "{:.2f}".format(str_))

    if "0.95 MoE" in annotations.columns:

        annotations["Score"] = tuple(
            "{} \u00B1 {}".format(score_str, moe_str)
            for score_str, moe_str in zip(
                annotations["Score"], annotations.pop("0.95 MoE")
            )
        )

    return annotations
