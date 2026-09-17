"""Predictive distribution of the price - valuation based on a saved posterior.

This module never runs MCMC. It takes the ready samples from the artifacts and
does arithmetic on them, so a valuation takes milliseconds no matter how long
the model itself took to fit.

We keep two sources of uncertainty apart, because they answer two different
questions:

  * uncertainty about the average price in a segment - it follows from the fact
    that we know the parameters only up to some accuracy. It shrinks as we
    collect more data.

  * uncertainty about the price of one particular flat - on top of the above
    comes the spread between listings with identical features (finish, view,
    how urgent the sale is). This one does not shrink with more data, because
    it is irreducible market noise.

For valuing a flat the second, wider interval is the right one.
"""
import numpy as np

# Features the user gives "in human terms". The order in the model vector comes
# from the artifacts, not from here - see _feature_vector.
UNKNOWN_DISTRICT = -1


def _feature_vector(area_m2, rooms, floor, features):
    """Builds the raw feature vector in the order stored in the artifacts.

    The order matters: beta[0] is the weight of the first feature on the
    features list. If we laid out the vector the way we imagine it, and the list
    in the artifacts were different, the model would multiply the area by the
    weight of the floor - no error anywhere, just a meaningless result. That is
    why we build a dictionary first and only then lay out the vector according
    to features.
    """
    values = {
        "log_metraz_m2": np.log(area_m2),
        "liczba_pokoi": float(rooms),
        "pietro": float(floor),
    }
    missing = set(features) - set(values)
    if missing:
        raise ValueError(f"Cannot compute features: {sorted(missing)}")
    return np.array([[values[name] for name in features]], dtype=np.float64)


def district_index(name, levels):
    """Returns the index of the district, or UNKNOWN_DISTRICT if we do not know it.

    Note: this is a different path than data.encode, which raises in the same
    situation. There we are talking about a dataset, where an unknown district
    means an error in the data. Here the user knowingly asks about a district
    the model has not seen. The model can answer that, just less confidently.
    """
    return levels.index(name) if name in levels else UNKNOWN_DISTRICT


def district_effects(posterior):
    """alpha[s, k] - the effect of district k in sample s.

    alpha is already in the posterior thanks to numpyro.deterministic in the
    model, so there is no need to recompute it from mu and z.
    """
    return posterior["alpha"]


def predictive_log_price(artifacts, X_raw, group_idx, include_noise=True, seed=0):
    """Predictive samples of log(price), shape (n_samples, n_listings).

    group_idx == UNKNOWN_DISTRICT marks a district outside the training set: we
    then draw a new effect from the population distribution, that is
    alpha_new = mu_city + sigma_district * z,  z ~ N(0, 1).

    This is exactly where the hierarchy pays off: the model says something
    sensible about a district it has never seen, relying only on how much the
    districts of Krakow differ from one another, and honestly widens the
    interval.
    """
    posterior = artifacts["posterior"]
    scaler = artifacts["scaler"]

    X = scaler.transform(np.asarray(X_raw, dtype=np.float64))
    group_idx = np.asarray(group_idx)

    alpha = district_effects(posterior)                # (S, K)
    n_samples = alpha.shape[0]

    alpha_obs = np.empty((n_samples, len(group_idx)), dtype=np.float64)
    known = group_idx >= 0
    if known.any():
        # array indexing: for every listing we take the effect of its own district
        alpha_obs[:, known] = alpha[:, group_idx[known]]
    if (~known).any():
        rng = np.random.default_rng(seed)
        z_new = rng.standard_normal((n_samples, int((~known).sum())))
        alpha_obs[:, ~known] = (
            posterior["mu_city"][:, None]
            + posterior["sigma_district"][:, None] * z_new
        )

    # X @ beta.T gives (N, S), so we transpose to (S, N)
    mu = alpha_obs + (X @ posterior["beta"].T).T

    if not include_noise:
        return mu

    # Market noise: Student's distribution with nu degrees of freedom, scaled by sigma.
    rng = np.random.default_rng(seed + 1)
    nu = posterior["nu"][:, None]
    t = rng.standard_t(np.broadcast_to(nu, mu.shape), size=mu.shape)
    return mu + posterior["sigma"][:, None] * t


def price_summary(log_price_draws, quantiles=(0.05, 0.25, 0.5, 0.75, 0.95)):
    """Turns log(price) samples into statistics in zlotys.

    We apply exp() to every sample separately and only then compute quantiles.
    The other order (quantiles on the log scale, then exp) would give the same
    quantiles, because exp is increasing, but no longer the same mean, which on
    the zloty scale is larger than exp(mean of logs).
    """
    prices = np.exp(log_price_draws)
    q = np.quantile(prices, quantiles, axis=0)
    return {
        "q5": q[0], "q25": q[1], "median": q[2], "q75": q[3], "q95": q[4],
        "mean": prices.mean(axis=0),
    }


def district_premiums(artifacts):
    """Premium or discount of each district against the city average, in percent.

    We compute exp(alpha - mu) - 1, that is by how many percent a flat with the
    same features is more expensive in a given district than on average in
    Krakow. Subtraction on the log scale corresponds to division on the zloty
    scale.

    Returns the mean and a 90% interval for each district - a single number
    without an interval does not say whether the difference is distinguishable
    from zero.
    """
    posterior = artifacts["posterior"]
    relative = 100.0 * (
        np.exp(posterior["alpha"] - posterior["mu_city"][:, None]) - 1.0
    )
    return {
        name: {
            "mean": float(relative[:, k].mean()),
            "q5": float(np.quantile(relative[:, k], 0.05)),
            "q95": float(np.quantile(relative[:, k], 0.95)),
        }
        for k, name in enumerate(artifacts["levels"])
    }


def value_flat(artifacts, district, area_m2, rooms, floor, seed=0):
    """Values a single flat. Returns both kinds of intervals.

    'listing' - the spread of individual offers (this is the valuation of a flat)
    'segment' - uncertainty about the average itself in this market segment
    """
    idx = district_index(district, artifacts["levels"])
    X_raw = _feature_vector(area_m2, rooms, floor, artifacts["features"])

    full = predictive_log_price(artifacts, X_raw, [idx], True, seed)
    average = predictive_log_price(artifacts, X_raw, [idx], False, seed)

    return {
        "district": district,
        "known_district": idx != UNKNOWN_DISTRICT,
        "listing": {k: float(v[0]) for k, v in price_summary(full).items()},
        "segment": {k: float(v[0]) for k, v in price_summary(average).items()},
    }
