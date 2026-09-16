"""Running MCMC and saving/loading artifacts.

This module separates two stages with very different costs:

    fit()   - slow (seconds), run once in a while
    load()  - instant, run on every valuation

Thanks to that prediction never repeats MCMC, the saved posterior is enough.
It is the same split as in production ML: training offline, serving online.
"""
import json
from pathlib import Path

import jax
import numpy as np
import numpyro.diagnostics as diag
from numpyro.infer import MCMC, NUTS

from kfv.data import FEATURES, Scaler
from kfv.model import model

DEFAULT_ARTIFACTS = Path(__file__).parent.parent.parent / "artifacts"
POSTERIOR_FILE = "posterior.npz"
META_FILE = "meta.json"


def fit(dataset, num_warmup=1000, num_samples=1000, num_chains=4, seed=0):
    """Fits the model to the data with NUTS.

    Takes the dictionary from data.build(). We compute mu_prior_loc here as the
    mean log(price) in training - the model does not do it itself, because it is
    not allowed to know the data.

    Returns the MCMC object rather than the samples alone, because it also
    carries the sampler statistics needed for diagnostics.
    """
    mcmc = MCMC(
        NUTS(model),
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        progress_bar=False,
    )
    mcmc.run(
        jax.random.PRNGKey(seed),
        X=dataset["X"],
        group_idx=dataset["group_idx"],
        n_groups=len(dataset["levels"]),
        mu_prior_loc=float(np.mean(dataset["y"])),
        y=dataset["y"],
    )
    return mcmc


def diagnostics(mcmc):
    """Returns the worst R-hat, the lowest ESS and the number of divergences.

    We look at the worst value across all parameters rather than at an average:
    a single parameter that has not converged invalidates the whole result.
    """
    samples = mcmc.get_samples(group_by_chain=True)

    worst_rhat, worst_rhat_param = 0.0, None
    lowest_ess, lowest_ess_param = np.inf, None

    for name, values in samples.items():
        values = np.asarray(values)
        rhat = float(np.max(diag.gelman_rubin(values)))
        ess = float(np.min(diag.effective_sample_size(values)))
        if rhat > worst_rhat:
            worst_rhat, worst_rhat_param = rhat, name
        if ess < lowest_ess:
            lowest_ess, lowest_ess_param = ess, name

    return {
        "max_rhat": worst_rhat,
        "max_rhat_param": worst_rhat_param,
        "min_ess": lowest_ess,
        "min_ess_param": lowest_ess_param,
        "divergences": int(np.sum(mcmc.get_extra_fields()["diverging"])),
    }


def converged(diag_result, rhat_threshold=1.01, ess_threshold=400):
    """Are the results safe to interpret?

    Three conditions at once - each catches a different kind of failure:
      R-hat       - the chains do not agree with each other,
      ESS         - samples too correlated, estimates unstable,
      divergences - the sampler avoids part of the space, samples are biased.
    """
    return (
        diag_result["max_rhat"] < rhat_threshold
        and diag_result["min_ess"] > ess_threshold
        and diag_result["divergences"] == 0
    )


def save(mcmc, dataset, directory=None, extra=None):
    """Saves the posterior and everything a later valuation needs.

    The posterior alone is not enough. Without `levels` there is no telling which
    effect belongs to Krowodrza; without the Scaler parameters there is no way to
    put the area of a new flat on the same scale; without FEATURES there is no
    telling in which order to lay out the features.

    npz + json instead of a single binary file: the metadata can be inspected in
    any text editor, without any libraries.
    """
    directory = Path(directory) if directory else DEFAULT_ARTIFACTS
    directory.mkdir(parents=True, exist_ok=True)

    # chains are merged into one sample dimension: (chains, draws, ...) -> (S, ...)
    samples = {k: np.asarray(v) for k, v in mcmc.get_samples().items()}
    np.savez_compressed(directory / POSTERIOR_FILE, **samples)

    meta = {
        "features": FEATURES,
        "levels": dataset["levels"],
        "scaler": dataset["scaler"].to_dict(),
        "mu_prior_loc": float(np.mean(dataset["y"])),
        "n_train": int(len(dataset["y"])),
        "diagnostics": diagnostics(mcmc),
    }
    if extra:
        meta.update(extra)

    (directory / META_FILE).write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return directory


def load(directory=None):
    """Loads the artifacts written by save().

    We rebuild the Scaler here as a ready object, so that the caller does not
    have to remember that the JSON holds plain lists rather than numpy arrays.
    """
    directory = Path(directory) if directory else DEFAULT_ARTIFACTS
    posterior_path = directory / POSTERIOR_FILE

    if not posterior_path.is_file():
        raise FileNotFoundError(
            f"No artifacts in {directory}. Fit the model first."
        )

    posterior = dict(np.load(posterior_path))
    meta = json.loads((directory / META_FILE).read_text(encoding="utf-8"))
    meta["scaler"] = Scaler.from_dict(meta["scaler"])
    return {"posterior": posterior, **meta}
