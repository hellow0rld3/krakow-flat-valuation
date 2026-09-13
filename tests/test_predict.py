"""Testy modulu predykcji.

Nie wczytujemy prawdziwych artefaktow z dysku, tylko budujemy SZTUCZNY
posterior o znanych wartosciach. Trzy powody:

  1. artifacts/ jest w .gitignore, wiec po swiezym git clone nie byloby czego
     wczytac i testy padalyby u wszystkich oprocz autora,
  2. nie trzeba uruchamiac MCMC, wiec testy trwaja milisekundy,
  3. skoro sami ustalamy alpha i beta, wynik da sie policzyc na kartce -
     sprawdzamy konkretna liczbe, a nie "cos w okolicy".
"""
import numpy as np
import pytest

from kfv import predict
from kfv.data import Scaler

S = 200                                   # liczba probek w sztucznym posteriorze
ALPHA = np.array([13.0, 14.0, 15.0])      # trzy dzielnice o roznych poziomach cen
LEVELS = ["Tania", "Srednia", "Droga"]


def artefakty(features, beta, sigma=0.1, sigma_dzielnica=0.5):
    """Buduje sztuczne artefakty o zadanych wagach cech.

    Scaler jest tozsamosciowy (srednia 0, odchylenie 1), wiec cechy wchodza do
    modelu dokladnie takie, jakie je podamy - bez tego nie dalo by sie policzyc
    oczekiwanego wyniku recznie.

    alpha jest STALE we wszystkich probkach, wiec bez szumu rynku predykcja
    jest deterministyczna.
    """
    return {
        "posterior": {
            "alpha": np.tile(ALPHA, (S, 1)),
            "beta": np.tile(np.asarray(beta, dtype=float), (S, 1)),
            "mu_miasto": np.full(S, ALPHA.mean()),
            "sigma_dzielnica": np.full(S, sigma_dzielnica),
            "sigma": np.full(S, sigma),
            "nu": np.full(S, 5.0),
        },
        "levels": LEVELS,
        "features": features,
        "scaler": Scaler(mean=np.zeros(len(features)), std=np.ones(len(features))),
    }


def test_kolejnosc_cech_pochodzi_z_artefaktow():
    """Wektor cech musi byc ulozony wedlug listy features z ARTEFAKTOW.

    To najwazniejszy test w tym pliku, bo lapie blad, ktory nie daje zadnego
    objawu: gdyby predykcja ukladala cechy wedlug wlasnego wyobrazenia,
    a zapisany model mial inna kolejnosc, liczba pokoi zostalaby pomnozona
    przez wage pieter. Zaden wyjatek nie poleci, wynik bedzie wygladal
    wiarygodnie i bedzie po prostu zly.

    Scenariusz: te same wagi beta = [0, 1, 0], czyli caly efekt na DRUGIEJ
    cesze. Zmieniamy tylko kolejnosc listy features i sprawdzamy, ze efekt
    przenosi sie na inna wielkosc.
    """
    mieszkanie = dict(dzielnica="Tania", metraz_m2=50, liczba_pokoi=3, pietro=7)

    # wariant A: druga cecha to pietro  -> oczekujemy alpha + 7
    a = predict.wycen(
        artefakty(["log_metraz_m2", "pietro", "liczba_pokoi"], [0.0, 1.0, 0.0]),
        **mieszkanie)
    # wariant B: druga cecha to liczba pokoi -> oczekujemy alpha + 3
    b = predict.wycen(
        artefakty(["log_metraz_m2", "liczba_pokoi", "pietro"], [0.0, 1.0, 0.0]),
        **mieszkanie)

    assert np.log(a["segment"]["mediana"]) == pytest.approx(13.0 + 7.0, abs=1e-6)
    assert np.log(b["segment"]["mediana"]) == pytest.approx(13.0 + 3.0, abs=1e-6)


def test_znana_dzielnica_uzywa_wlasciwego_alpha():
    """Dla dzielnicy k predykcja musi opierac sie na alpha[k].

    Wagi cech sa zerowe, wiec jedyne, co wplywa na wynik, to efekt dzielnicy.
    Kazda z trzech musi dac dokladnie swoja wartosc z ALPHA.
    """
    art = artefakty(["log_metraz_m2", "pietro", "liczba_pokoi"], [0.0, 0.0, 0.0])

    for k, nazwa in enumerate(LEVELS):
        wynik = predict.wycen(art, nazwa, metraz_m2=50, liczba_pokoi=3, pietro=2)
        assert wynik["znana_dzielnica"]
        assert np.log(wynik["segment"]["mediana"]) == pytest.approx(ALPHA[k], abs=1e-6)


def test_nieznana_dzielnica_daje_szerszy_przedzial():
    """Dzielnica spoza treningu nie jest bledem, tylko wieksza niepewnoscia.

    Dla znanej dzielnicy alpha jest w sztucznym posteriorze stale, wiec
    przedzial dla sredniej segmentu ma szerokosc zero. Dla nieznanej model
    losuje nowy efekt z rozkladu populacyjnego N(mu, sigma_dzielnica), wiec
    przedzial musi sie wyraznie rozszerzyc.

    Na tym polega praktyczna wartosc hierarchii: model odpowiada na pytanie,
    na ktore nie ma danych, i uczciwie sygnalizuje, ze wie mniej.
    """
    art = artefakty(["log_metraz_m2", "pietro", "liczba_pokoi"], [0.0, 0.0, 0.0])
    mieszkanie = dict(metraz_m2=50, liczba_pokoi=3, pietro=2)

    znana = predict.wycen(art, "Srednia", **mieszkanie)
    nieznana = predict.wycen(art, "Wola Justowska", **mieszkanie)

    assert not nieznana["znana_dzielnica"]

    szer_znana = znana["segment"]["q95"] - znana["segment"]["q5"]
    szer_nieznana = nieznana["segment"]["q95"] - nieznana["segment"]["q5"]
    assert szer_nieznana > szer_znana


def test_szum_rynku_poszerza_przedzial():
    """Przedzial dla konkretnej oferty musi byc szerszy niz dla sredniej segmentu.

    Pierwszy zawiera oba zrodla niepewnosci (niewiedza modelu + rozrzut rynku),
    drugi tylko pierwsze. Gdyby wyszly rowne, znaczyloby to, ze przelacznik
    include_noise nie dziala i uzytkownik dostaje zanizona niepewnosc.
    """
    art = artefakty(["log_metraz_m2", "pietro", "liczba_pokoi"], [0.0, 0.0, 0.0])
    wynik = predict.wycen(art, "Srednia", metraz_m2=50, liczba_pokoi=3, pietro=2)

    szer_oferta = wynik["oferta"]["q95"] - wynik["oferta"]["q5"]
    szer_segment = wynik["segment"]["q95"] - wynik["segment"]["q5"]
    assert szer_oferta > szer_segment
