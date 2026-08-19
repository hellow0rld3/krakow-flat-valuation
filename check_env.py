"""
Test środowiska: sprawdza, czy JAX, NumPyro i ArviZ działają poprawnie.

Uruchom raz po instalacji:  .venv/bin/python check_env.py

Przy okazji jest to najmniejszy możliwy kompletny przykład NumPyro -
prosty model, który szacuje średnią i odchylenie z 200 liczb.
"""
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS


def model(y=None):
    """Najprostszy model bayesowski: skąd pochodzą te liczby?"""
    mu = numpyro.sample("mu", dist.Normal(0.0, 10.0))
    sigma = numpyro.sample("sigma", dist.HalfNormal(5.0))
    numpyro.sample("obs", dist.Normal(mu, sigma), obs=y)


def main():
    print("=== Wersje ===")
    print(f"  jax      {jax.__version__}")
    print(f"  numpyro  {numpyro.__version__}")
    print(f"  backend  {jax.default_backend()}  (urządzenia: {jax.devices()})")

    # Dane testowe: 200 liczb z rozkładu N(5, 2). Model powinien to odkryć.
    key = jax.random.PRNGKey(0)
    y = 5.0 + 2.0 * jax.random.normal(key, (200,))

    print("\n=== Test NUTS (prawda: mu=5.0, sigma=2.0) ===")
    mcmc = MCMC(NUTS(model), num_warmup=500, num_samples=500, num_chains=1,
                progress_bar=False)
    mcmc.run(jax.random.PRNGKey(1), y=y)
    mcmc.print_summary()

    s = mcmc.get_samples()
    mu_hat, sigma_hat = float(jnp.mean(s["mu"])), float(jnp.mean(s["sigma"]))
    ok = abs(mu_hat - 5.0) < 0.5 and abs(sigma_hat - 2.0) < 0.5

    import arviz as az
    print(f"  arviz    {az.__version__}")

    print(f"\n{'ŚRODOWISKO DZIAŁA' if ok else 'COŚ JEST NIE TAK'}: "
          f"mu={mu_hat:.2f}, sigma={sigma_hat:.2f}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
