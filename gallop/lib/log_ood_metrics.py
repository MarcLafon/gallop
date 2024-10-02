from typing import Type

import pandas as pd
import numpy as np

import gallop.lib as lib

NoneType = Type[None]


def log_ood_metrics(ood_metrics: np.ndarray) -> NoneType:
    table_names = [name_ood for name_ood in ood_metrics.keys()]
    df_res = pd.DataFrame([[""] * len(table_names)], columns=table_names)

    fpr95_avg, auroc_avg, count = 0.0, 0.0, 0
    for ood_name, ood_metric in ood_metrics.items():
        fpr95 = ood_metric["fpr95"]
        auroc = ood_metric["auroc"]
        fpr95_avg += fpr95
        auroc_avg += auroc
        count += 1
        df_res[ood_name] = [f"{np.around(100 * fpr95, 3):.3f} / {np.around(100 * auroc, 3):.3f}&"]
    df_res["Average"] = [f" {np.around(100 * fpr95_avg / count, 3):.3f} / {np.around(100 * auroc_avg / count, 3):.3f}"]

    res = df_res.to_string(index=True)
    lib.LOGGER.info("*** =============")
    lib.LOGGER.info("* " + res.split("\n")[0][4:])
    lib.LOGGER.info("* " + res.split("\n")[1][1:])
