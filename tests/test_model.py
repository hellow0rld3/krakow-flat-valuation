"""Parameter recovery test.

We generate data with known parameters and check whether the model can recover
them. This is the only test that verifies the correctness of the inference
itself - ordinary unit tests will not catch a mistake in a prior, a flipped
sign or wrong district indexing, because the code runs without an exception and
returns plausible-looking numbers.
"""
import jax
import numpy as np
import numpyro.diagnostics as diag
import pytest
from numpyro.handlers import condition
from numpyro.infer import MCMC, NUTS, Predictive

from kfv.model import model

N_OBS = 800          # as many as we really have in the training set
N_GROUPS = 19        # as many as there are districts
SEED = 0

_rng = np.random.default_rng(SEED)

# We center z_district on zero and pass it to the generator that way.
#
# Without it the test would make no sense: alpha[k] = mu + tau * z[k], so the
# data arise with a real mean alpha equal to mu + tau * mean(z). With 19 groups
# mean(z) has a standard deviation of 1/sqrt(19) ~ 0.23, so a random z moves the
# real mean away from the nominal mu by a few tenths of sigma_district. The
# posterior then correctly estimates what actually generated the data - and the
# test would be comparing it against a number that never produced those data.
_z = _rng.normal(size=N_GROUPS)
_z -= _z.mean()

# True values of the parameters. We set them by hand instead of drawing them
# from the prior: the prior is wide and could draw, say, nu=40 or
# sigma_district=1.2, from which recovering the parameters can be hard. The test
# would then pass only now and then, and a flaky test is worse than no test.
TRUTH = {
    "mu_city": 13.5,
    "sigma_district": 0.15,
    "z_district": _z,
    "beta": np.array([0.90, 0.02, -0.01]),
    "sigma": 0.15,
    "nu": 5.0,
}


@pytest.fixture(scope="module")
def posterior():
    """Generates data at TRUTH and fits the model to them.

    scope="module" makes MCMC run once for the whole file rather than before
    every test separately.
    """
    X = _rng.normal(size=(N_OBS, 3))                # features already "standardized"
    group_idx = _rng.integers(0, N_GROUPS, size=N_OBS)

    inputs = dict(X=X, group_idx=group_idx,
                  n_groups=N_GROUPS, mu_prior_loc=TRUTH["mu_city"])

    # condition substitutes the given values instead of drawing them from the
    # priors, so y arises exactly at the parameters from TRUTH. The generator is
    # the same model() function, so there is no risk of it drifting apart from
    # the specification.
    generated = Predictive(condition(model, TRUTH), num_samples=1)(
        jax.random.PRNGKey(SEED), **inputs)
    y = np.asarray(generated["y"][0])

    mcmc = MCMC(NUTS(model), num_warmup=1500, num_samples=2000,
                num_chains=2, progress_bar=False)
    mcmc.run(jax.random.PRNGKey(SEED + 1), y=y, **inputs)
    return mcmc


@pytest.mark.slow
def test_model_recovers_parameters(posterior):
    """All 26 true values have to lie inside the 95% intervals.

    We check coverage, not equality. MCMC is a random method, so an estimate
    will never come out exactly equal to the truth - the question is whether the
    truth lies inside the interval the model considers credible.

    A 95% interval instead of a 90% one gives us a margin: with 26 values
    checked and a 90% interval, about 2.6 of them would fall outside purely by
    chance. The seeds are fixed, so the test is deterministic - it will fail
    only once something really breaks.
    """
    samples = posterior.get_samples()
    outside = []

    for name, truth in TRUTH.items():
        lo, hi = np.quantile(np.asarray(samples[name]), [0.025, 0.975], axis=0)
        for j, value in enumerate(np.atleast_1d(truth)):
            lower, upper = np.atleast_1d(lo)[j], np.atleast_1d(hi)[j]
            if not lower <= value <= upper:
                outside.append(f"{name}[{j}]: {value:.3f} outside [{lower:.3f}, {upper:.3f}]")

    assert not outside, "Parameters outside the interval:\n" + "\n".join(outside)


@pytest.mark.slow
def test_chains_converged(posterior):
    """Without convergence the result of the previous test would mean nothing.

    Chains that have not converged may hit the truth or miss it - either way
    their samples do not come from the posterior. That is why we check parameter
    recovery together with the diagnostics.

    Divergences mean that the sampler systematically avoids part of the
    parameter space, so the samples are biased even with a correct R-hat.
    """
    samples = posterior.get_samples(group_by_chain=True)
    for name, values in samples.items():
        rhat = float(np.max(diag.gelman_rubin(np.asarray(values))))
        assert rhat < 1.01, f"{name}: R-hat = {rhat:.4f}"

    divergences = int(np.sum(posterior.get_extra_fields()["diverging"]))
    assert divergences == 0, f"{divergences} divergences"
