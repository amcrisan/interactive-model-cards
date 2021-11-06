"""
rg_utils load helpers methods from python
"""

import pandas as pd
import re
import robustnessgym as rg
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def update_pred(dp, model, dp_only=False):
    """ Updating data panel with model prediction"""

    model.predict_batch(dp, ["sentence"])

    dp = dp.update(
        lambda x: model.predict_batch(x, ["sentence"]),
        batch_size=4,
        is_batched_fn=True,
        pbar=True,
    )

    if dp_only:
        return dp

    labels = pd.Series(["Negative Sentiment", "Positive Sentiment"])
    probs = pd.Series(dp.__dict__["_data"]["probs"][0])

    pred = pd.concat([labels, probs], axis=1)
    pred.columns = ["Label", "Probability"]
    return (dp, pred)


def remove_slice(bench, slice_name="user_data"):
    """ Remove a slice from the rg dev bench"""

    # slices and identifiers are in the same order
    slice_list = []
    slice_identifier = []

    for i in bench.__dict__["_slices"]:
        # look-up the term
        name = str(i.__dict__["_identifier"])
        if not re.search("new_words", name):
            slice_list = slice_list + [i]
            slice_identifier = slice_identifier + [name]

    # metrics put datain a different order
    metrics = {}
    for key in bench.metrics["model"].keys():
        if not re.search("new_words", key):
            metrics[key] = bench.metrics["model"][key]

    # slice table, repeat for sanity  check
    # slice_table = {}
    # for key in bench.__dict__["_slice_table"].keys():
    #    key = str(key)
    #    if not re.search("new_words",key):
    #        slice_table[key] = bench.__dict__["_slice_table"][key]

    bench.__dict__["_slices"] = set(slice_list)
    bench.__dict__["_slice_identifiers"] = set(slice_identifier)
    # bench.__dict__["_slice_table"] = set(slice_identifier)

    bench.metrics["model"] = metrics

    return bench


def add_slice(bench, table, model, slice_name="user_data"):
    """ Adds a custom slice to RG """
    # do it this way or it complains
    dp = rg.DataPanel(
        {
            "sentence": table["sentence"].tolist(),
            "label": table["label"].tolist(),
            "pred": table["pred"].tolist(),
        }
    )

    # dp._identifier = slice_name

    # get prediction
    # add to bench
    # bench.add_slices([dp])
    return dp


def new_bench():
    """ Create new rg dev bench"""
    bench = rg.DevBench()
    bench.add_aggregators(
        {
            # Every model can be associated with custom metric calculation functions
            #'distilbert-base-uncased-finetuned-sst-2-english': {
            "model": {
                # This function uses the predictions we stored earlier to calculate accuracy
                #'accuracy': lambda dp: (dp['label'].round() == dp['pred'].numpy()).mean()
                #'f1' : lambda dp: f1_score(dp['label'].round(),dp['pred'],average='macro',zero_division=1),
                "recall": lambda dp: recall_score(
                    dp["label"].round(), dp["pred"], average="macro", zero_division=1
                ),
                "precision": lambda dp: precision_score(
                    dp["label"].round(), dp["pred"], average="macro", zero_division=1
                ),
                "accuracy": lambda dp: accuracy_score(dp["label"].round(), dp["pred"]),
            }
        }
    )
    return bench


def get_sliceid(slices):
    """ Because RG stores data in a silly way"""
    ids = []
    for slice in list(slices):
        ids = ids + [slice._identifier]

    return ids


def slice_to_df(data):
    """ Convert slice to dataframe"""
    df = pd.DataFrame(
        {
            "sentence": list(data["sentence"]),
            "model label": [int(round(x)) for x in data["label"]],
            "probability": list(data["label"]),
        }
    )

    return df


def metrics_to_dict(metrics, slice_name):
    """ Convert metrics to dataframe"""

    all_metrics = {slice_name: {}}
    all_metrics[slice_name]["metrics"] = metrics[slice_name]
    all_metrics[slice_name]["source"] = "Custom Slice"

    return all_metrics
