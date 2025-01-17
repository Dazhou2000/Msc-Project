import json

import numpy as np

import tsfel


def load_json(json_path):
    """Loads the json file given by filename.

    Parameters
    ----------
    json_path : string
        Json path

    Returns
    -------
    Dict
        Dictionary
    """

    return json.load(open(json_path))


def get_features_by_domain(domain=None, json_path=None):
    """Creates a dictionary with the features settings by domain.

    Parameters
    ----------
    domain : string
        Available domains: "statistical"; "spectral"; "temporal"; "fractal"
        If domain equals None, then the features settings from all domains are returned.
    json_path : string
        Directory of json file. Default: package features.json directory

    Returns
    -------
    Dict
        Dictionary with the features settings
    """

    if json_path is None:
        json_path = tsfel.__path__[0] + "/feature_extraction/features.json"

        if domain not in ["statistical", "temporal", "spectral", "fractal", None]:
            raise SystemExit(
                "No valid domain. Choose: statistical, temporal, spectral, fractal or None (for all feature settings).",
            )

    dict_features = load_json(json_path)
    if domain is None:
        return dict_features
    else:
        if domain == "fractal":
            for k in dict_features[domain]:
                dict_features[domain][k]["use"] = "yes"
        return {domain: dict_features[domain]}



def get_number_features(dict_features):
    """Count the total number of features based on input parameters of each
    feature.

    Parameters
    ----------
    dict_features : dict
        Dictionary with features settings

    Returns
    -------
    int
        Feature vector size
    """
    number_features = 0
    for domain in dict_features:
        for feat in dict_features[domain]:
            if dict_features[domain][feat]["use"] == "no":
                continue
            n_feat = dict_features[domain][feat]["n_features"]

            if isinstance(n_feat, int):
                number_features += n_feat
            else:
                n_feat_param = dict_features[domain][feat]["parameters"][n_feat]
                if isinstance(n_feat_param, int):
                    number_features += n_feat_param
                else:
                    number_features += eval("len(" + n_feat_param + ")")

    return number_features