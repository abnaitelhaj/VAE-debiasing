"""Optional Comet ML experiment tracking."""

import os

import comet_ml


def create_experiment(project_name, params, api_key=None):
    if api_key is None:
        api_key = os.environ.get("COMET_API_KEY", "")
    assert api_key != "", "Set COMET_API_KEY in the environment to enable Comet logging."

    experiment = comet_ml.Experiment(api_key=api_key, project_name=project_name)
    for param, value in params.items():
        experiment.log_parameter(param, value)
    experiment.flush()
    return experiment
