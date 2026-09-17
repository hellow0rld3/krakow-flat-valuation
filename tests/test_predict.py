"""Tests of the prediction module.

We do not load the real artifacts from disk, we build a synthetic posterior with
known values instead. Three reasons:

  1. artifacts/ is in .gitignore, so after a fresh git clone there would be
     nothing to load and the tests would fail for everyone except the author,
  2. no need to run MCMC, so the tests take milliseconds,
  3. since we set alpha and beta ourselves, the result can be worked out on
     paper - we check a specific number, not "something in the right region".
"""
import numpy as np
import pytest

from kfv import predict
from kfv.data import Scaler

S = 200                                   # number of samples in the synthetic posterior
ALPHA = np.array([13.0, 14.0, 15.0])      # three districts at different price levels
LEVELS = ["Cheap", "Middling", "Expensive"]


def artifacts(features, beta, sigma=0.1, sigma_district=0.5):
    """Builds synthetic artifacts with the given feature weights.

    The Scaler is the identity (mean 0, standard deviation 1), so the features
    enter the model exactly as we pass them - without that the expected result
    could not be computed by hand.

    alpha is constant across all the samples, so without market noise the
    prediction is deterministic.
    """
    return {
        "posterior": {
            "alpha": np.tile(ALPHA, (S, 1)),
            "beta": np.tile(np.asarray(beta, dtype=float), (S, 1)),
            "mu_city": np.full(S, ALPHA.mean()),
            "sigma_district": np.full(S, sigma_district),
            "sigma": np.full(S, sigma),
            "nu": np.full(S, 5.0),
        },
        "levels": LEVELS,
        "features": features,
        "scaler": Scaler(mean=np.zeros(len(features)), std=np.ones(len(features))),
    }


def test_feature_order_comes_from_the_artifacts():
    """The feature vector has to follow the features list from the artifacts.

    This is the most important test in the file, because it catches a bug that
    gives no symptom at all: if prediction laid out the features the way it
    imagines them, and the saved model had a different order, the number of
    rooms would be multiplied by the weight of the floor. No exception would be
    raised, the result would look credible and would simply be wrong.

    The setup: the same weights beta = [0, 1, 0], that is the whole effect on
    the second feature. We change only the order of the features list and check
    that the effect moves onto a different quantity.
    """
    flat = dict(district="Cheap", area_m2=50, rooms=3, floor=7)

    # variant A: the second feature is the floor -> we expect alpha + 7
    a = predict.value_flat(
        artifacts(["log_metraz_m2", "pietro", "liczba_pokoi"], [0.0, 1.0, 0.0]),
        **flat)
    # variant B: the second feature is the number of rooms -> we expect alpha + 3
    b = predict.value_flat(
        artifacts(["log_metraz_m2", "liczba_pokoi", "pietro"], [0.0, 1.0, 0.0]),
        **flat)

    assert np.log(a["segment"]["median"]) == pytest.approx(13.0 + 7.0, abs=1e-6)
    assert np.log(b["segment"]["median"]) == pytest.approx(13.0 + 3.0, abs=1e-6)


def test_known_district_uses_the_right_alpha():
    """For district k the prediction has to be based on alpha[k].

    The feature weights are zero, so the only thing that affects the result is
    the district effect. Each of the three has to give exactly its own value
    from ALPHA.
    """
    art = artifacts(["log_metraz_m2", "pietro", "liczba_pokoi"], [0.0, 0.0, 0.0])

    for k, name in enumerate(LEVELS):
        result = predict.value_flat(art, name, area_m2=50, rooms=3, floor=2)
        assert result["known_district"]
        assert np.log(result["segment"]["median"]) == pytest.approx(ALPHA[k], abs=1e-6)


def test_unknown_district_widens_the_interval():
    """A district outside training is not an error, only more uncertainty.

    For a known district alpha is constant in the synthetic posterior, so the
    interval for the segment average has zero width. For an unknown one the
    model draws a new effect from the population distribution
    N(mu, sigma_district), so the interval has to widen clearly.

    This is where the hierarchy pays off in practice: the model answers a
    question it has no data for, and honestly signals that it knows less.
    """
    art = artifacts(["log_metraz_m2", "pietro", "liczba_pokoi"], [0.0, 0.0, 0.0])
    flat = dict(area_m2=50, rooms=3, floor=2)

    known = predict.value_flat(art, "Middling", **flat)
    unknown = predict.value_flat(art, "Wola Justowska", **flat)

    assert not unknown["known_district"]

    width_known = known["segment"]["q95"] - known["segment"]["q5"]
    width_unknown = unknown["segment"]["q95"] - unknown["segment"]["q5"]
    assert width_unknown > width_known


def test_market_noise_widens_the_interval():
    """The interval for one listing has to be wider than for the segment average.

    The first contains both sources of uncertainty (what the model does not know
    plus the spread of the market), the second only the first one. If they came
    out equal, it would mean the include_noise switch does not work and the user
    is given an understated uncertainty.
    """
    art = artifacts(["log_metraz_m2", "pietro", "liczba_pokoi"], [0.0, 0.0, 0.0])
    result = predict.value_flat(art, "Middling", area_m2=50, rooms=3, floor=2)

    width_listing = result["listing"]["q95"] - result["listing"]["q5"]
    width_segment = result["segment"]["q95"] - result["segment"]["q5"]
    assert width_listing > width_segment
