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
    return tr, te


def test_ksztalty_sa_spojne(zbiory):
    """X, y i group_idx musza opisywac te same wiersze.

    Rozjazd liczby wierszy to typowy skutek uboczny filtrowania danych
    w jednym miejscu i zapomnienia o nim w drugim. Model nie zglosi bledu
    tylko po prostu dopasuje ceny do niewlasciwych mieszkan.

    Sprawdzamy tez, ze y ma ksztalt (N,), a nie (N, 1) - ta druga postac
    psuje pozniej broadcasting przy liczeniu rozkladu predykcyjnego.
    """
    for zbior in zbiory:
        n = len(zbior["y"])
        assert zbior["X"].shape == (n, len(data.FEATURES))
        assert zbior["y"].shape == (n,)
        assert zbior["group_idx"].shape == (n,)


def test_group_idx_miesci_sie_w_zakresie(zbiory):
    """Kazdy indeks dzielnicy musi wskazywac na istniejacy poziom 0..K-1.

    Ten test lapie ciche uszkodzenia kodu: nieznana dzielnica zamieniona
    na -1 (co w numpy oznacza OSTATNI element) albo NaN skonwertowany na 0
    (czyli PIERWSZA dzielnica). W obu przypadkach nic nie wybucha, a oferta
    zostaje po cichu przypisana do niewlasciwej dzielnicy.
    """
    tr, te = zbiory
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
    
    tr, te = zbiory

    assert np.allclose(tr["X"].mean(axis=0), 0.0, atol=1e-8)
    assert np.allclose(tr["X"].std(axis=0), 1.0, atol=1e-8)

    assert np.abs(te["X"].mean(axis=0)).max() > 0.01
