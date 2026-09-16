import numpyro
import numpyro.distributions as dist


def model(X, group_idx, n_groups, mu_prior_loc, y=None):
    """Hierarchical hedonic model for flat valuation.

        log(price) ~ StudentT(nu, alpha[district] + X @ beta, sigma)
        alpha[k]   = mu + sigma_district * z[k]

    Arguments:
        X            - standardized features, shape (N, 3)
        group_idx    - district index of every listing, shape (N,)
        n_groups     - number of districts K (passed explicitly instead of being
                       derived from group_idx - the last district may be absent
                       from the test set, K would come out too small and train
                       and test would end up with a different number of parameters)
        mu_prior_loc - center of the prior on mu, i.e. mean log(price) in training
        y            - log(price); None when generating data from the model

    The model knows nothing about files or pandas frames - it takes ready arrays.
    That is what makes it possible to feed it synthetic data in the parameter
    recovery test.
    """
    n_obs, n_features = X.shape

    # city level (hyperpriors)
    mu = numpyro.sample("mu_city", dist.Normal(mu_prior_loc, 1.0))
    sigma_district = numpyro.sample("sigma_district", dist.HalfNormal(0.5))

    # district level, non-centered parameterization
    # z is N(0,1) regardless of sigma_district, which decouples the two levels
    # of the hierarchy and guards against sparse groups and Neal's funnel,
    # even though the centered version does just as well here
    with numpyro.plate("districts", n_groups):
        z = numpyro.sample("z_district", dist.Normal(0.0, 1.0))

    # feature effects (predictors are standardized)
    with numpyro.plate("features", n_features):
        beta = numpyro.sample("beta", dist.Normal(0.0, 1.0))

    # observation noise
    sigma = numpyro.sample("sigma", dist.HalfNormal(0.5))
    # Student instead of a normal distribution: nu is learned from the data and
    # decides how much weight to give extreme listings, instead of cutting them
    # out by hand.
    nu = numpyro.sample("nu", dist.Gamma(2.0, 0.1))

    alpha = mu + sigma_district * z             # (K,)
    # deterministic keeps alpha in the MCMC output even though it is not a
    # sampled parameter - thanks to that prediction does not recompute it.
    numpyro.deterministic("alpha", alpha)

    mu_obs = alpha[group_idx] + X @ beta        # (N,)

    # N comes from X, not from len(y), because y is None when generating data.
    with numpyro.plate("listings", n_obs):
        numpyro.sample("y", dist.StudentT(nu, mu_obs, sigma), obs=y)
