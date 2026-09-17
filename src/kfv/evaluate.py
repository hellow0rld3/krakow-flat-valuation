"""Ocena modelu na zbiorze testowym.

Liczymy dwie rozne rzeczy:

  BLAD PUNKTOWY (MAPE, RMSE) - o ile srednio mylimy sie co do ceny
  KALIBRACJA                 - czy przedzial 90% faktycznie pokrywa 90% cen

Silnik ma znac wlasna niepewnosc, wiec kalibracja jest kryterium waznym
bardziej niz sama trafnosc. Model, ktory myli sie o 14%, ale uczciwie to
przyznaje, jest uzyteczny. Model, ktory myli sie o 10% i twierdzi, ze myli sie
o 2%, jest niebezpieczny, bo ktos podejmie na jego podstawie decyzje
finansowa, ufajac przedzialowi, ktory nie ma pokrycia w rzeczywistosci.

Modul niczego nie modyfikuje i nie zapisuje, tylko liczy.
"""
import numpy as np

from kfv import predict
from kfv.data import GROUP, TARGET

# Poziomy, dla ktorych sprawdzamy pokrycie. Kilka zamiast jednego, bo model
# moze byc dobrze skalibrowany w srodku rozkladu i zle w ogonach - jeden punkt
# tego nie pokaze.
POZIOMY = (0.50, 0.80, 0.90, 0.95)


def przygotuj_wejscie(artefakty, df):
    """Wyciaga z ramki to, czego potrzebuje predict: surowe cechy, indeksy, prawde.

    Cechy bierzemy w kolejnosci z artefaktow (a nie z data.FEATURES), bo to
    zapisany model dyktuje, ktora pozycja wektora odpowiada ktorej wadze beta.

    Dzielnice mapujemy przez predict.district_index, wiec nieznana dzielnica
    dostanie -1 zamiast wyjatku. W ocenie modelu to poprawne zachowanie:
    chcemy wiedziec, jak radzi sobie takze tam, gdzie nie mial danych.
    """
    X_raw = df[artefakty["features"]].to_numpy(dtype=np.float64)
    group_idx = np.array(
        [predict.district_index(d, artefakty["levels"]) for d in df[GROUP]],
        dtype=np.int32,
    )
    y_log = df[TARGET].to_numpy(dtype=np.float64)
    return X_raw, group_idx, y_log


def bledy_punktowe(ceny_prawdziwe, ceny_przewidziane):
    """Klasyczne metryki trafnosci.

    Podajemy i srednia, i mediane bledu wzglednego, bo rozjazd miedzy nimi sam
    w sobie niesie informacje: duza roznica oznacza, ze rozklad bledow jest
    skosny, czyli kilka nietypowych ofert psuje srednia.
    """
    blad_wzgledny = np.abs(ceny_przewidziane - ceny_prawdziwe) / ceny_prawdziwe
    return {
        "MAPE": float(np.mean(blad_wzgledny)),
        "mediana_bledu": float(np.median(blad_wzgledny)),
        "RMSE_pln": float(np.sqrt(np.mean((ceny_przewidziane - ceny_prawdziwe) ** 2))),
    }


def kalibracja(ceny_probki, ceny_prawdziwe, poziomy=POZIOMY):
    """Dla kazdego poziomu: jaki odsetek prawdziwych cen trafil w przedzial.

    Przedzial o poziomie 0.9 budujemy jako kwantyle 5% i 95% - czyli symetrycznie,
    odcinajac po polowie reszty z kazdej strony.

    Interpretacja:
      pokrycie ~ poziom  -> model uczciwie ocenia wlasna niepewnosc
      pokrycie < poziom  -> model jest zbyt pewny siebie
      pokrycie > poziom  -> model jest zachowawczy, przedzialy za szerokie
    """
    wynik = []
    for poziom in poziomy:
        margines = (1.0 - poziom) / 2.0
        dol, gora = np.quantile(ceny_probki, [margines, 1.0 - margines], axis=0)
        pokryte = (ceny_prawdziwe >= dol) & (ceny_prawdziwe <= gora)
        wynik.append({
            "poziom": poziom,
            "pokrycie": float(np.mean(pokryte)),
            "srednia_szerokosc_pln": float(np.mean(gora - dol)),
        })
    return wynik


def ocen(artefakty, df, poziomy=POZIOMY, seed=0):
    """Pelna ocena na podanej ramce (zwykle zbior testowy).

    Wycena punktowa to mediana rozkladu predykcyjnego, a nie srednia. Na skali
    zlotowek rozklad jest prawoskosny, wiec srednia jest zawyzona przez ogon
    drogich ofert wiec mediana lepiej opisuje typowa cene.
    """
    X_raw, group_idx, y_log = przygotuj_wejscie(artefakty, df)

    probki_log = predict.predictive_log_price(
        artefakty, X_raw, group_idx, include_noise=True, seed=seed)

    ceny_probki = np.exp(probki_log)
    ceny_prawdziwe = np.exp(y_log)
    ceny_przewidziane = np.median(ceny_probki, axis=0)

    return {
        "n": int(len(ceny_prawdziwe)),
        "n_nieznanych_dzielnic": int(np.sum(group_idx < 0)),
        **bledy_punktowe(ceny_prawdziwe, ceny_przewidziane),
        "kalibracja": kalibracja(ceny_probki, ceny_prawdziwe, poziomy),
    }


def bledy_per_dzielnica(artefakty, df, seed=0):
    """Mediana bledu wzglednego w rozbiciu na dzielnice.

    Przydatne do diagnozy: model moze byc dobry srednio, a systematycznie mylic
    sie w jednym segmencie rynku. Srednia po calym zbiorze to ukryje.
    """
    X_raw, group_idx, y_log = przygotuj_wejscie(artefakty, df)
    probki_log = predict.predictive_log_price(
        artefakty, X_raw, group_idx, include_noise=True, seed=seed)

    ceny_prawdziwe = np.exp(y_log)
    ceny_przewidziane = np.median(np.exp(probki_log), axis=0)
    blad = np.abs(ceny_przewidziane - ceny_prawdziwe) / ceny_prawdziwe

    wynik = {}
    for dzielnica in sorted(df[GROUP].unique()):
        maska = (df[GROUP] == dzielnica).to_numpy()
        wynik[dzielnica] = {
            "n": int(maska.sum()),
            "mediana_bledu": float(np.median(blad[maska])),
        }
    return wynik
