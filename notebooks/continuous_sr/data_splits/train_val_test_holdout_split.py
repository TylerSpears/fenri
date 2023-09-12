# -*- coding: utf-8 -*-
import itertools
import random
from pathlib import Path

import numpy as np
import pandas as pd
from natsort import natsorted

HOLDOUTS = {
    "3t_no_anat_anom": [
        "159744",
        "202719",
        "163129",
        "105216",
        "919966",
        "333330",
        "633847",
        "118831",
        "182840",
        "131419",
        "201414",
        "146634",
        "810843",
        "101915",
        "117930",
        "104012",
        "173637",
        "120212",
        "183337",
        "219231",
    ],
    "3t_anat_anom": ["224022", "122620", "709551", "522434", "168139"],
    "7t_no_anat_anom": [
        "825048",
        "581450",
        "251833",
        "191336",
        # "601127",
        "126426",
        "859671",
        "525541",
        "910241",
        "186949",
        "212419",
        "104416",
        "239136",
        "429040",
        "401422",
        "169747",
        "346137",
        "644246",
        "872764",
        "878776",
    ],
    "7t_anat_anom": ["200210", "360030"],
}


def filter_subjs():
    subjs_table = dict(
        subj_id=list(),
        mag_strength=list(),
        anat_anomoly=list(),
    )

    hcp_3t_path = Path("./HCP_3T_only_QC_tags.csv").resolve()
    subjs_3t = pd.read_csv(hcp_3t_path)

    hcp_7t_path = Path("./HCP_7T_QC_tags.csv").resolve()
    subjs_7t = pd.read_csv(hcp_7t_path)

    # Remove subjects who had particular QC issues during acquisition.
    qc_remove_mask = ~np.asarray([("C" in r) | ("D" in r) for r in subjs_3t.QC_Issue])
    subjs_3t = subjs_3t[qc_remove_mask]
    qc_remove_mask = ~np.asarray([("C" in r) | ("D" in r) for r in subjs_7t.QC_Issue])
    subjs_7t = subjs_7t[qc_remove_mask]

    # Remove all subjects with 7T scans from the 3T list
    mag_7t_remove_mask = ~np.isin(subjs_3t.Subject, subjs_7t.Subject)
    subjs_3t = subjs_3t[mag_7t_remove_mask]
    anat_anomoly_3t_subjs = ["A" in r for r in subjs_3t.QC_Issue]

    subjs_table["subj_id"].extend(list(subjs_3t.Subject.map(str)))
    subjs_table["mag_strength"].extend(
        list(itertools.repeat("3T", len(subjs_3t.Subject)))
    )
    subjs_table["anat_anomoly"].extend(anat_anomoly_3t_subjs)

    anat_anomoly_7t_subjs = ["A" in r for r in subjs_7t.QC_Issue]
    subjs_table["subj_id"].extend(list(subjs_7t.Subject.map(str)))
    subjs_table["mag_strength"].extend(
        list(itertools.repeat("7T", len(subjs_7t.Subject)))
    )
    subjs_table["anat_anomoly"].extend(anat_anomoly_7t_subjs)

    subjs_table = pd.DataFrame.from_dict(subjs_table)
    subjs_table.to_csv("HCP_subj_to_split.csv", index=False)

    return Path("HCP_subj_to_split.csv")


def gen_train_val_test_holdout_split(subjs_table_path, seed=None):
    subjs_table = pd.read_csv(subjs_table_path)

    holdout_subjs = set().union(*[set(HOLDOUTS[k]) for k in HOLDOUTS.keys()])
    subjs_table = subjs_table[~np.isin(subjs_table.subj_id, list(holdout_subjs))]

    subjs_3t_nqca = natsorted(
        subjs_table.loc[
            (subjs_table.mag_strength == "3T") & ~subjs_table.anat_anomoly.astype(bool)
        ].subj_id.tolist()
    )
    subjs_3t_qca = natsorted(
        subjs_table.loc[
            (subjs_table.mag_strength == "3T") & subjs_table.anat_anomoly.astype(bool)
        ].subj_id.tolist()
    )

    if seed is None:
        seed = random.randint(0, np.iinfo(np.int32).max)
    print(seed)
    random.seed(seed)
    gen = np.random.default_rng(seed)

    # Randomly shuffle, then sample without replacement.
    gen.shuffle(subjs_3t_nqca)
    gen.shuffle(subjs_3t_qca)
    sample = {
        "train": {
            "3t_nqca": list(),
            "3t_qca": list(),
        },
        "val": {
            "3t_nqca": list(),
            "3t_qca": list(),
        },
        "test": {
            "3t_nqca": list(),
            "3t_qca": list(),
        },
    }

    n_train_3t_nqca = 35
    n_train_3t_qca = 0
    n_val_3t_nqca = 5
    n_val_3t_qca = 0
    n_test_3t_nqca = 20
    n_test_3t_qca = 5

    reduced_3t_nqca = subjs_3t_nqca.copy()
    reduced_3t_qca = subjs_3t_qca.copy()

    sample["train"]["3t_nqca"] = reduced_3t_nqca[:n_train_3t_nqca]
    reduced_3t_nqca = reduced_3t_nqca[n_train_3t_nqca:]
    sample["val"]["3t_nqca"] = reduced_3t_nqca[:n_val_3t_nqca]
    reduced_3t_nqca = reduced_3t_nqca[n_val_3t_nqca:]
    sample["test"]["3t_nqca"] = reduced_3t_nqca[:n_test_3t_nqca]
    reduced_3t_nqca = reduced_3t_nqca[n_test_3t_nqca:]

    sample["train"]["3t_qca"] = reduced_3t_qca[:n_train_3t_qca]
    reduced_3t_qca = reduced_3t_qca[n_train_3t_qca:]
    sample["val"]["3t_qca"] = reduced_3t_qca[:n_val_3t_qca]
    reduced_3t_qca = reduced_3t_qca[n_val_3t_qca:]
    sample["test"]["3t_qca"] = reduced_3t_qca[:n_test_3t_qca]
    reduced_3t_qca = reduced_3t_qca[n_test_3t_qca:]

    split_table = {
        "subj_id": list(),
        "split": list(),
        "anat_anom": list(),
        "mag_strength": list(),
    }
    for split in ("train", "val", "test"):
        for group in ("3t_nqca", "3t_qca"):
            s = sample[split][group]
            n_s = len(s)
            split_table["subj_id"].extend(sample[split][group])
            split_table["split"].extend(list(itertools.repeat(split, n_s)))
            split_table["anat_anom"].extend(
                list(itertools.repeat(False if "nqca" in group else True, n_s))
            )
            split_table["mag_strength"].extend(list(itertools.repeat("3T", n_s)))

    split_table = pd.DataFrame.from_dict(split_table)

    # Assert that each row has a unique subj id.
    assert len(split_table.subj_id.unique()) == len(split_table.subj_id)

    split_table.to_csv(f"HCP_train-val-test_split_seed_{seed}.csv", index=False)
    return split_table


if __name__ == "__main__":
    if not Path("HCP_subj_to_split.csv").exists():
        subjs_table_path = filter_subjs().resolve()
    else:
        subjs_table_path = Path("HCP_subj_to_split.csv").resolve()

    gen_train_val_test_holdout_split(subjs_table_path)
