"""Rozklad predykcyjny ceny - wycena na podstawie ZAPISANEGO posteriora.

Ten modul nie uruchamia MCMC. Bierze gotowe probki z artefaktow i robi na nich
arytmetyke, wiec wycena trwa milisekundy niezaleznie od tego, jak dlugo
dopasowywal sie model.

Rozrozniamy dwa zrodla niepewnosci, bo odpowiadaja na dwa rozne pytania:

  * niepewnosc co do SREDNIEJ ceny w segmencie - wynika z tego, ze parametry
    modelu znamy tylko z pewna dokladnoscia. Maleje, gdy zbieramy wiecej danych.

  * niepewnosc co do ceny KONKRETNEGO mieszkania - dochodzi rozrzut miedzy
    ofertami o identycznych cechach (standard wykonczenia, widok, pilnosc
    sprzedazy). NIE maleje wraz z liczba danych, bo to nieredukowalny szum rynku.

Do wyceny mieszkania wlasciwy jest ten drugi, szerszy przedzial.
"""
import numpy as np

# Cechy, ktore uzytkownik podaje "po ludzku". Kolejnosc w wektorze modelu
# bierzemy z artefaktow, nie stad - patrz _wektor_cech.
NIEZNANA_DZIELNICA = -1


def _wektor_cech(metraz_m2, liczba_pokoi, pietro, features):
    """Sklada surowy wektor cech w kolejnosci zapisanej w artefaktach.

    Kolejnosc ma znaczenie: beta[0] to waga pierwszej cechy z listy features.
    Gdybysmy ulozyli wektor wedlug wlasnego wyobrazenia, a lista w artefaktach
    bylaby inna, model pomnozylby metraz przez wage pieter, bez zadnego bledu,
    tylko z bezsensownym wynikiem. Dlatego budujemy slownik i dopiero z niego
    ukladamy wektor wedlug features.
    """
    wartosci = {
        "log_metraz_m2": np.log(metraz_m2),
        "liczba_pokoi": float(liczba_pokoi),
        "pietro": float(pietro),
    }
    brakujace = set(features) - set(wartosci)
    if brakujace:
        raise ValueError(f"Nie umiem policzyc cech: {sorted(brakujace)}")
    return np.array([[wartosci[nazwa] for nazwa in features]], dtype=np.float64)


def indeks_dzielnicy(nazwa, levels):
    """Zwraca indeks dzielnicy albo NIEZNANA_DZIELNICA, jesli jej nie znamy

    Uwaga: to inna sciezka niz data.encode, ktore w takiej sytuacji rzuca
    wyjatek. Tam mowimy o ZBIORZE DANYCH, gdzie nieznana dzielnica oznacza blad
    w danych. Tutaj uzytkownik swiadomie pyta o dzielnice, ktorej model nie
    widzial. Model potrafi na to odpowiedziec, tylko mniej pewnie.
    """
    return levels.index(nazwa) if nazwa in levels else NIEZNANA_DZIELNICA


def efekty_dzielnic(posterior):
    """alpha[s, k] - efekt dzielnicy k w probce s.

    alpha jest juz w posteriorze dzieki numpyro.deterministic w modelu, wiec
    nie trzeba go przeliczac z mu i z.
    """
    return posterior["alpha"]


def predictive_log_price(artefakty, X_raw, group_idx, include_noise=True, seed=0):
    """Probki predykcyjne log(ceny), ksztalt (n_probek, n_ofert).

    group_idx == NIEZNANA_DZIELNICA oznacza dzielnice spoza zbioru treningowego:
    losujemy wtedy NOWY efekt z rozkladu populacyjnego, czyli
    alpha_nowa = mu_miasto + sigma_dzielnica * z,  z ~ N(0, 1).

    Na tym wlasnie polega wartosc hierarchii: model mowi cos sensownego
    o dzielnicy, ktorej nigdy nie widzial, opierajac sie wylacznie na tym, jak
    bardzo dzielnice Krakowa roznia sie miedzy soba i uczciwie poszerza
    przedzial.
    """
    posterior = artefakty["posterior"]
    scaler = artefakty["scaler"]

    X = scaler.transform(np.asarray(X_raw, dtype=np.float64))
    group_idx = np.asarray(group_idx)

    alpha = efekty_dzielnic(posterior)                 # (S, K)
    n_probek = alpha.shape[0]

    alpha_obs = np.empty((n_probek, len(group_idx)), dtype=np.float64)
    znane = group_idx >= 0
    if znane.any():
        # indeksowanie tablica: dla kazdej oferty bierzemy efekt JEJ dzielnicy
        alpha_obs[:, znane] = alpha[:, group_idx[znane]]
    if (~znane).any():
        rng = np.random.default_rng(seed)
        z_nowe = rng.standard_normal((n_probek, int((~znane).sum())))
        alpha_obs[:, ~znane] = (
            posterior["mu_city"][:, None]
            + posterior["sigma_district"][:, None] * z_nowe
        )

    # X @ beta.T daje (N, S), wiec transponujemy do (S, N)
    mu = alpha_obs + (X @ posterior["beta"].T).T

    if not include_noise:
        return mu

    # Szum rynku: rozklad Studenta o nu stopniach swobody, przeskalowany sigma.
    rng = np.random.default_rng(seed + 1)
    nu = posterior["nu"][:, None]
    t = rng.standard_t(np.broadcast_to(nu, mu.shape), size=mu.shape)
    return mu + posterior["sigma"][:, None] * t


def podsumowanie_ceny(log_price_draws, kwantyle=(0.05, 0.25, 0.5, 0.75, 0.95)):
    """Zamienia probki log(ceny) na statystyki w zlotowkach.

    exp() stosujemy do kazdej probki osobno, a dopiero potem liczymy kwantyle.
    Odwrotna kolejnosc (kwantyle na skali log, potem exp) dalaby te same
    kwantyle, bo exp jest rosnaca, ale juz nie srednia, ktora na skali
    zlotowek jest wieksza od exp(sredniej log).
    """
    ceny = np.exp(log_price_draws)
    q = np.quantile(ceny, kwantyle, axis=0)
    return {
        "q5": q[0], "q25": q[1], "mediana": q[2], "q75": q[3], "q95": q[4],
        "srednia": ceny.mean(axis=0),
    }


def premia_dzielnic(artefakty):
    """Premia lub dyskonto kazdej dzielnicy wzgledem sredniej miasta, w procentach.

    Liczymy exp(alpha - mu) - 1, czyli o ile procent drozsze jest mieszkanie
    o TYCH SAMYCH cechach w danej dzielnicy niz przecietnie w Krakowie.
    Odejmowanie na skali log odpowiada dzieleniu na skali zlotowek.

    Zwraca dla kazdej dzielnicy srednia i przedzial 90% - bo sama liczba bez
    przedzialu nie mowi, czy roznica jest odrozniana od zera.
    """
    posterior = artefakty["posterior"]
    wzgledne = 100.0 * (
        np.exp(posterior["alpha"] - posterior["mu_city"][:, None]) - 1.0
    )
    return {
        nazwa: {
            "srednia": float(wzgledne[:, k].mean()),
            "q5": float(np.quantile(wzgledne[:, k], 0.05)),
            "q95": float(np.quantile(wzgledne[:, k], 0.95)),
        }
        for k, nazwa in enumerate(artefakty["levels"])
    }


def wycen(artefakty, dzielnica, metraz_m2, liczba_pokoi, pietro, seed=0):
    """Wycenia jedno mieszkanie. Zwraca oba rodzaje przedzialow.

    'oferta' - rozrzut konkretnych ogloszen (to jest wycena mieszkania)
    'segment' - niepewnosc samej sredniej w tym segmencie rynku
    """
    idx = indeks_dzielnicy(dzielnica, artefakty["levels"])
    X_raw = _wektor_cech(metraz_m2, liczba_pokoi, pietro, artefakty["features"])

    pelne = predictive_log_price(artefakty, X_raw, [idx], True, seed)
    srednia = predictive_log_price(artefakty, X_raw, [idx], False, seed)

    return {
        "dzielnica": dzielnica,
        "znana_dzielnica": idx != NIEZNANA_DZIELNICA,
        "oferta": {k: float(v[0]) for k, v in podsumowanie_ceny(pelne).items()},
        "segment": {k: float(v[0]) for k, v in podsumowanie_ceny(srednia).items()},
    }
