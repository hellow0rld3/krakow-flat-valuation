import numpy as np
import pytest

from kfv import data


@pytest.fixture
def zbiory():
    """Buduje raz zbiory train/test i udostepnia je wszystkim testom.

    Test jest przetwarzany parametrami z treningu (levels, scaler) - dokladnie
    tak, jak zrobi to pozniej predykcja na nowych danych.
    """
    df = data.prepare_dataset(data.load_data())
    train, test = data.split(df)
    tr = data.build(train)
    te = data.build(test, levels=tr["levels"], scaler=tr["scaler"])
    return {"train_df": train, "tr": tr, "test_df": test, "te": te}


def test_wiersze_odpowiadaja_zrodlu(zbiory):
    """X, y i group_idx musza opisywac te same wiersze co ramka zrodlowa.

    Sprawdzamy dwie rzeczy, kazda z innego powodu:

    1. Ksztalty. Rozjazd DLUGOSCI i tak wywalilby pozniej MCMC
       ("Incompatible shapes for broadcasting"), wiec tutaj tylko lapiemy
       go wczesniej i czytelniej. Realna wartosc ma natomiast warunek
       y.shape == (N,): postac (N, 1) nie zglosi bledu, tylko po cichu
       rozjedzie broadcasting przy liczeniu rozkladu predykcyjnego.

    2. Zgodnosc wiersz po wierszu. To jest przypadek, ktorego nie wykryje
       nic innego: tablice o poprawnych ksztaltach, ale z przestawionymi
       wierszami. Zdarza sie, gdy ktos przefiltruje albo posortuje jedno
       zrodlo, a zapomni o pozostalych. Model policzy sie normalnie
       i dopasuje ceny do niewlasciwych mieszkan.

    Dla wybranych wierszy sprawdzamy wiec, ze y zgadza sie z cena z ramki,
    ze group_idx wskazuje na wlasciwa dzielnice, i ze odstandaryzowane X
    odtwarza oryginalne cechy.
    """
    for klucz_df, klucz_zbior in [("train_df", "tr"), ("test_df", "te")]:
        df, zbior = zbiory[klucz_df], zbiory[klucz_zbior]
        n = len(zbior["y"])

        assert zbior["X"].shape == (n, len(data.FEATURES))
        assert zbior["y"].shape == (n,)
        assert zbior["group_idx"].shape == (n,)

        for i in (0, n // 3, n - 1):
            assert zbior["y"][i] == pytest.approx(df[data.TARGET].iloc[i])
            assert zbior["levels"][zbior["group_idx"][i]] == df[data.GROUP].iloc[i]

            odtworzone = zbior["X"][i] * zbior["scaler"].std + zbior["scaler"].mean
            assert odtworzone == pytest.approx(df[data.FEATURES].iloc[i].to_numpy())


def test_group_idx_miesci_sie_w_zakresie(zbiory):
    """Kazdy indeks dzielnicy musi wskazywac na istniejacy poziom 0..K-1.

    Ten test lapie ciche uszkodzenia kodu: nieznana dzielnica zamieniona
    na -1 (co w numpy oznacza OSTATNI element) albo NaN skonwertowany na 0
    (czyli PIERWSZA dzielnica). W obu przypadkach nic nie wybucha, a oferta
    zostaje po cichu przypisana do niewlasciwej dzielnicy.
    """
    tr, te = zbiory["tr"], zbiory["te"]
    k = len(tr["levels"])
    for zbior in (tr, te):
        assert zbior["group_idx"].min() >= 0
        assert zbior["group_idx"].max() < k


def test_standaryzacja_nie_przecieka(zbiory):
    """Parametry standaryzacji moga pochodzic wylacznie ze zbioru treningowego.

    Sprawdzamy dwie rzeczy naraz:

    1. srednia X_train ~ 0 i odchylenie ~ 1  -> parametry faktycznie policzono na treningu,
    2. srednia X_test ODBIEGA od zera        -> parametrow NIE policzono na tescie.

    Punkt 2 jest wlasciwym detektorem wycieku. Gdyby ktos "poprawil" kod tak,
    by scaler dopasowywal sie do zbioru testowego, punkt 1 nadal by przechodzil
    (trening zawsze ma srednia zero), a wszystkie metryki jakosci modelu
    wyszlyby zawyzone - bez zadnego bledu i bez zadnego ostrzezenia.

    Prog 0.01 jest arbitralny, ale bezpieczny: przy poprawnym podziale
    obserwowane odchylenie srednich testowych to okolo 0.25.
    """
    
    tr, te = zbiory["tr"], zbiory["te"]

    assert np.allclose(tr["X"].mean(axis=0), 0.0, atol=1e-8)
    assert np.allclose(tr["X"].std(axis=0), 1.0, atol=1e-8)

    assert np.abs(te["X"].mean(axis=0)).max() > 0.01
