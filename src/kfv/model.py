import numpyro
import numpyro.distributions as dist


def model(X, group_idx, n_groups, mu_prior_loc, y=None):
    """Hierarchiczny model hedoniczny wyceny mieszkan.

        log(cena) ~ StudentT(nu, alpha[dzielnica] + X @ beta, sigma)
        alpha[k]  = mu + sigma_dzielnica * z[k]        

    Argumenty:
        X            - wystandaryzowane cechy, ksztalt (N, 3)
        group_idx    - indeks dzielnicy kazdej oferty, ksztalt (N,)
        n_groups     - liczba dzielnic K (podawana jawnie, a nie liczona
                       z group_idx - w zbiorze testowym moze nie wystapic
                       ostatnia dzielnica i K wyszloby za male, przez co
                       trening i test mialyby rozna liczbe parametrow)
        mu_prior_loc - srodek priora na mu, czyli srednia log(ceny) w treningu
        y            - log(ceny); None przy generowaniu danych z modelu

    Model nie wie nic o plikach ani ramkach pandas - przyjmuje gotowe tablice.
    Dzieki temu mozna go nakarmic danymi syntetycznymi w tescie odzyskiwania
    parametrow.
    """
    n_obs, n_features = X.shape

    # poziom miasta (hiperpriory)
    mu = numpyro.sample("mu_miasto", dist.Normal(mu_prior_loc, 1.0))
    sigma_dzielnica = numpyro.sample("sigma_dzielnica", dist.HalfNormal(0.5))

    # poziom dzielnicy, parametryzacja niecentrowana
    # z ma rozklad N(0,1) NIEZALEZNIE od sigma_dzielnica, co rozprzega oba
    # poziomy hierarchi i zabezpiecza na wypadek rzadkich grup i pojawienia
    # sie lejka Neala, mimo że centrowana radzi sobie równie dobrze
    with numpyro.plate("dzielnice", n_groups):
        z = numpyro.sample("z_dzielnica", dist.Normal(0.0, 1.0))

    # efekty cech (predyktory sa wystandaryzowane)
    with numpyro.plate("cechy", n_features):
        beta = numpyro.sample("beta", dist.Normal(0.0, 1.0))

    # szum obserwacyjny
    sigma = numpyro.sample("sigma", dist.HalfNormal(0.5))
    # Student zamiast rozkladu normalnego: nu uczone z danych decyduje,
    # ile wagi dac ofertom skrajnym, zamiast wycinac je recznie.
    nu = numpyro.sample("nu", dist.Gamma(2.0, 0.1))

    alpha = mu + sigma_dzielnica * z            # (K,)
    # deterministic zapisuje alpha w wynikach MCMC, mimo ze nie jest to
    # parametr probkowany - dzieki temu predykcja nie musi go przeliczac.
    numpyro.deterministic("alpha", alpha)

    mu_obs = alpha[group_idx] + X @ beta        # (N,)

    # N bierzemy z X, a nie z len(y) bo przy generowaniu danych y jest None.
    with numpyro.plate("oferty", n_obs):
        numpyro.sample("y", dist.StudentT(nu, mu_obs, sigma), obs=y)
