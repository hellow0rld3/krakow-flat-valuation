"""Uruchomienie MCMC i zapis/odczyt artefaktow.

Ten modul rozdziela dwa etapy o zupelnie roznym koszcie:

    fit()   - wolny (sekundy), odpalany raz na jakis czas
    load()  - natychmiastowy, odpalany przy kazdej wycenie

Dzieki temu predykcja nie musi powtarzac MCMC - wystarczy jej zapisany
posterior. To ten sam podzial, co w produkcyjnym ML: trening offline,
serwowanie online.
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


def fit(zbior, num_warmup=1000, num_samples=1000, num_chains=4, seed=0):
    """Dopasowuje model do danych metoda NUTS.

    Argumentem jest slownik z data.build(). mu_prior_loc liczymy tutaj jako
    srednia log(ceny) w treningu - model sam tego nie robi, bo nie wolno mu
    znac danych.

    Zwraca obiekt MCMC (a nie same probki), bo niesie tez statystyki samplera
    potrzebne do diagnostyki.
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
        X=zbior["X"],
        group_idx=zbior["group_idx"],
        n_groups=len(zbior["levels"]),
        mu_prior_loc=float(np.mean(zbior["y"])),
        y=zbior["y"],
    )
    return mcmc


def diagnostics(mcmc):
    """Zwraca najgorszy R-hat, najmniejszy ESS i liczbe dywergencji.

    Patrzymy na NAJGORSZA wartosc w calym zestawie parametrow, a nie na
    srednia: jeden parametr, ktory nie zbiegl, unieważnia caly wynik.
    """
    probki = mcmc.get_samples(group_by_chain=True)

    najgorszy_rhat, parametr_rhat = 0.0, None
    najmniejszy_ess, parametr_ess = np.inf, None

    for nazwa, wartosci in probki.items():
        wartosci = np.asarray(wartosci)
        rhat = float(np.max(diag.gelman_rubin(wartosci)))
        ess = float(np.min(diag.effective_sample_size(wartosci)))
        if rhat > najgorszy_rhat:
            najgorszy_rhat, parametr_rhat = rhat, nazwa
        if ess < najmniejszy_ess:
            najmniejszy_ess, parametr_ess = ess, nazwa

    return {
        "max_rhat": najgorszy_rhat,
        "max_rhat_param": parametr_rhat,
        "min_ess": najmniejszy_ess,
        "min_ess_param": parametr_ess,
        "divergences": int(np.sum(mcmc.get_extra_fields()["diverging"])),
    }


def zbiegly(diag_wynik, prog_rhat=1.01, prog_ess=400):
    """Czy wyniki wolno interpretowac?

    Trzy warunki naraz - kazdy wychwytuje inny rodzaj awarii:
      R-hat      - lancuchy nie zgadzaja sie ze soba,
      ESS        - probki zbyt skorelowane, estymaty niestabilne,
      dywergencje- sampler omija czesc przestrzeni, probki obciazone.
    """
    return (
        diag_wynik["max_rhat"] < prog_rhat
        and diag_wynik["min_ess"] > prog_ess
        and diag_wynik["divergences"] == 0
    )


def save(mcmc, zbior, katalog=None, dodatkowe=None):
    """Zapisuje posterior i wszystko, czego potrzebuje pozniejsza wycena.

    Sam posterior nie wystarczy. Bez `levels` nie wiadomo, ktory efekt dotyczy
    Krowodrzy; bez parametrow Scalera nie da sie przeliczyc metrazu nowego
    mieszkania na te sama skale; bez FEATURES nie wiadomo, w jakiej kolejnosci
    ulozyc cechy.

    npz + json zamiast jednego pliku binarnego: metadane mozna podejrzec
    zwyklym edytorem, bez zadnych bibliotek.
    """
    katalog = Path(katalog) if katalog else DEFAULT_ARTIFACTS
    katalog.mkdir(parents=True, exist_ok=True)

    # laczymy lancuchy w jeden wymiar probek: (chains, draws, ...) -> (S, ...)
    probki = {k: np.asarray(v) for k, v in mcmc.get_samples().items()}
    np.savez_compressed(katalog / POSTERIOR_FILE, **probki)

    meta = {
        "features": FEATURES,
        "levels": zbior["levels"],
        "scaler": zbior["scaler"].to_dict(),
        "mu_prior_loc": float(np.mean(zbior["y"])),
        "n_train": int(len(zbior["y"])),
        "diagnostics": diagnostics(mcmc),
    }
    if dodatkowe:
        meta.update(dodatkowe)

    (katalog / META_FILE).write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return katalog


def load(katalog=None):
    """Wczytuje artefakty zapisane przez save().

    Scaler odtwarzamy tu od razu jako gotowy obiekt, zeby wywolujacy nie musial
    pamietac, ze w JSON-ie leza zwykle listy, a nie tablice numpy.
    """
    katalog = Path(katalog) if katalog else DEFAULT_ARTIFACTS
    sciezka_posterior = katalog / POSTERIOR_FILE

    if not sciezka_posterior.is_file():
        raise FileNotFoundError(
            f"Brak artefaktow w {katalog}. Uruchom najpierw dopasowanie modelu."
        )

    posterior = dict(np.load(sciezka_posterior))
    meta = json.loads((katalog / META_FILE).read_text(encoding="utf-8"))
    meta["scaler"] = Scaler.from_dict(meta["scaler"])
    return {"posterior": posterior, **meta}
