"""Test odzyskiwania parametrow (parameter recovery).

Generujemy dane o znanych parametrach i sprawdzamy, czy model potrafi je
odzyskac. To jedyny test, ktory weryfikuje poprawnosc samego wnioskowania -
zwykle testy jednostkowe nie wykryja pomylki w priorze, odwroconego znaku
ani zlego indeksowania dzielnic, bo kod wykona sie bez wyjatku i zwroci
prawdopodobnie wygladajace liczby.
"""
import jax
import numpy as np
import numpyro.diagnostics as diag
import pytest
from numpyro.handlers import condition
from numpyro.infer import MCMC, NUTS, Predictive

from kfv.model import model

N_OBS = 800          # tyle, ile mamy realnie w zbiorze treningowym
N_GROUPS = 19        # tyle, ile jest dzielnic
SEED = 0

_rng = np.random.default_rng(SEED)

# z_dzielnica CENTRUJEMY na zero i tak przekazujemy do generatora.
#
# Bez tego test nie mialby sensu: alpha[k] = mu + tau * z[k], wiec dane
# powstaja przy realnej sredniej alpha rownej mu + tau * mean(z). Przy 19
# grupach mean(z) ma odchylenie 1/sqrt(19) ~ 0.23, wiec losowe z odsuwa realna
# srednia od nominalnego mu o kilka dziesiatych sigma_dzielnica. Posterior
# trafnie szacuje wtedy to, co FAKTYCZNIE wygenerowalo dane - a test
# porownywalby go z liczba, ktora tych danych nigdy nie wyprodukowala.
_z = _rng.normal(size=N_GROUPS)
_z -= _z.mean()

# Prawdziwe wartosci parametrow. Ustalamy je recznie zamiast losowac z priora:
# prior jest szeroki i moglby wylosowac np. nu=40 albo sigma_dzielnica=1.2,
# z ktorych odzyskanie parametrow bywa trudne. Test przechodzilby wtedy raz na
# jakis czas, a test niestabilny jest gorszy niz brak testu.
PRAWDA = {
    "mu_miasto": 13.5,
    "sigma_dzielnica": 0.15,
    "z_dzielnica": _z,
    "beta": np.array([0.90, 0.02, -0.01]),
    "sigma": 0.15,
    "nu": 5.0,
}


@pytest.fixture(scope="module")
def posterior():
    """Generuje dane przy PRAWDA i dopasowuje do nich model.

    scope="module" sprawia, ze MCMC odpala sie raz na caly plik, a nie przed
    kazdym testem osobno.
    """
    X = _rng.normal(size=(N_OBS, 3))                # cechy juz "wystandaryzowane"
    group_idx = _rng.integers(0, N_GROUPS, size=N_OBS)

    wejscie = dict(X=X, group_idx=group_idx,
                   n_groups=N_GROUPS, mu_prior_loc=PRAWDA["mu_miasto"])

    # condition podstawia zadane wartosci zamiast losowac je z priorow, wiec y
    # powstaje DOKLADNIE przy parametrach z PRAWDA. Generatorem jest ta sama
    # funkcja model(), nie ma wiec ryzyka, ze rozjedzie sie ze specyfikacja.
    wygenerowane = Predictive(condition(model, PRAWDA), num_samples=1)(
        jax.random.PRNGKey(SEED), **wejscie)
    y = np.asarray(wygenerowane["y"][0])

    mcmc = MCMC(NUTS(model), num_warmup=1500, num_samples=2000,
                num_chains=2, progress_bar=False)
    mcmc.run(jax.random.PRNGKey(SEED + 1), y=y, **wejscie)
    return mcmc


@pytest.mark.slow
def test_model_odzyskuje_parametry(posterior):
    """Wszystkie 26 prawdziwych wartosci musi lezec w 95% przedzialach.

    Sprawdzamy POKRYCIE, a nie rownosc. MCMC jest metoda losowa, wiec estymata
    nigdy nie wyjdzie dokladnie rowna prawdzie - pytanie brzmi, czy prawda lezy
    w przedziale, ktory model uznaje za wiarygodny.

    Przedzial 95% zamiast 90% daje margines: przy 26 sprawdzanych wartosciach
    i przedziale 90% okolo 2.6 z nich wypadaloby poza przedzial czysto
    przypadkiem. Ziarna sa ustalone na sztywno, wiec test jest deterministyczny
    - zawiedzie dopiero wtedy, gdy naprawde cos sie zepsuje.
    """
    probki = posterior.get_samples()
    poza = []

    for nazwa, prawda in PRAWDA.items():
        lo, hi = np.quantile(np.asarray(probki[nazwa]), [0.025, 0.975], axis=0)
        for j, wartosc in enumerate(np.atleast_1d(prawda)):
            dolna, gorna = np.atleast_1d(lo)[j], np.atleast_1d(hi)[j]
            if not dolna <= wartosc <= gorna:
                poza.append(f"{nazwa}[{j}]: {wartosc:.3f} poza [{dolna:.3f}, {gorna:.3f}]")

    assert not poza, "Parametry poza przedzialem:\n" + "\n".join(poza)


@pytest.mark.slow
def test_lancuchy_zbiegly(posterior):
    """Bez zbieznosci wynik poprzedniego testu nie znaczylby nic.

    Lancuchy, ktore nie zbiegly, moga trafic w prawde albo w nia nie trafic -
    tak czy inaczej ich probki nie pochodza z posteriora. Dlatego odzyskiwanie
    parametrow sprawdzamy RAZEM z diagnostyka.

    Dywergencje oznaczaja, ze sampler systematycznie omija czesc przestrzeni
    parametrow, wiec probki sa obciazone nawet przy poprawnym R-hat.
    """
    probki = posterior.get_samples(group_by_chain=True)
    for nazwa, wartosci in probki.items():
        rhat = float(np.max(diag.gelman_rubin(np.asarray(wartosci))))
        assert rhat < 1.01, f"{nazwa}: R-hat = {rhat:.4f}"

    dywergencje = int(np.sum(posterior.get_extra_fields()["diverging"]))
    assert dywergencje == 0, f"{dywergencje} dywergencji"
