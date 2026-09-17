"""Interfejs terminalowy.

    python -m kfv.cli fit
    python -m kfv.cli summary
    python -m kfv.cli predict --dzielnica Krowodrza --metraz 55 --pokoje 3 --pietro 2

Ten modul NIE zawiera logiki - tylko parsuje argumenty, wola funkcje z
pozostalych modulow i formatuje wynik. Dzieki temu cala funkcjonalnosc da sie
przetestowac bez uruchamiania CLI, a pozniejsze dolozenie interfejsu webowego
bedzie dopisaniem drugiego cienkiego adaptera obok, a nie przepisywaniem
projektu.
"""
import argparse

import numpy as np
import numpyro

from kfv import data, evaluate, inference, predict


def _naglowek(tekst):
    print(f"\n{tekst}")
    print("-" * len(tekst))


# --------------------------------------------------------------------------- #
#  fit
# --------------------------------------------------------------------------- #
def cmd_fit(args):
    # Musi byc PRZED pierwszym uzyciem jax, inaczej lancuchy policza sie po
    # kolei zamiast rownolegle (samo importowanie jax jeszcze nie inicjalizuje
    # urzadzen, wiec to miejsce jest bezpieczne).
    numpyro.set_host_device_count(args.chains)

    df = data.prepare_dataset(data.load_data(args.data))
    train, test = data.split(df, seed=args.seed)
    zbior = data.build(train)

    print(f"Dane: {len(train)} treningowych / {len(test)} testowych, "
          f"{len(zbior['levels'])} dzielnic")
    print(f"Probkowanie: {args.chains} lancuchy x {args.samples} probek "
          f"(+{args.warmup} rozgrzewki)...")

    mcmc = inference.fit(zbior, num_warmup=args.warmup, num_samples=args.samples,
                         num_chains=args.chains, seed=args.seed)

    diag = inference.diagnostics(mcmc)
    _naglowek("Diagnostyka zbieznosci")
    print(f"  najgorszy R-hat   {diag['max_rhat']:.4f}  ({diag['max_rhat_param']})")
    print(f"  najmniejszy ESS   {diag['min_ess']:.0f}  ({diag['min_ess_param']})")
    print(f"  dywergencje       {diag['divergences']}")

    if inference.converged(diag):
        print("  -> wyniki mozna interpretowac")
    else:
        print("  -> UWAGA: lancuchy nie zbiegly. Zwieksz --warmup/--samples "
              "przed interpretacja wynikow.")

    if len(test) > 0:
        wynik = evaluate.evaluate_on(artefakty_z_pamieci(mcmc, zbior), test)
        _wypisz_ocene(wynik)
    else:
        wynik = None

    katalog = inference.save(mcmc, zbior, args.artifacts,
                             extra={"ocena_testowa": wynik})
    print(f"\nZapisano artefakty do {katalog}")


def artefakty_z_pamieci(mcmc, zbior):
    """Sklada strukture artefaktow bez zapisu na dysk.

    Potrzebne, bo ocene chcemy wypisac PRZED zapisem - gdyby model nie zbiegl,
    wolimy o tym wiedziec, zanim nadpiszemy poprzednie wyniki.
    """
    return {
        "posterior": {k: np.asarray(v) for k, v in mcmc.get_samples().items()},
        "levels": zbior["levels"],
        "scaler": zbior["scaler"],
        "features": data.FEATURES,
    }


def _wypisz_ocene(wynik):
    _naglowek(f"Ocena na zbiorze testowym ({wynik['n']} ofert)")
    print(f"  MAPE              {100 * wynik['MAPE']:>6.1f}%")
    print(f"  mediana bledu     {100 * wynik['median_error']:>6.1f}%")
    print(f"  RMSE              {wynik['RMSE_pln']:>9,.0f} zl")
    print("\n  Kalibracja (im blizej nominalnego, tym lepiej):")
    print(f"  {'nominalnie':>12}{'empirycznie':>14}{'sr. szerokosc':>17}")
    for k in wynik["calibration"]:
        print(f"  {100 * k['level']:>11.0f}%{100 * k['coverage']:>13.1f}%"
              f"{k['mean_width_pln']:>16,.0f} zl")


# --------------------------------------------------------------------------- #
#  summary
# --------------------------------------------------------------------------- #
def cmd_summary(args):
    art = inference.load(args.artifacts)
    post = art["posterior"]
    scaler = art["scaler"]

    _naglowek("Wplyw cech na cene")
    print("  (beta dotyczy cech wystandaryzowanych: efekt zmiany o 1 odchylenie)")
    for i, cecha in enumerate(art["features"]):
        b = post["beta"][:, i]
        print(f"  {cecha:<16} {b.mean():+.3f}  [{np.quantile(b, 0.05):+.3f}, "
              f"{np.quantile(b, 0.95):+.3f}]   => {100 * (np.exp(b.mean()) - 1):+.1f}% ceny")

    # Elastycznosc: beta liczy sie na cechach podzielonych przez odchylenie,
    # wiec zeby wrocic do "% ceny na % metrazu", trzeba przez nie przemnozyc.
    i_metraz = art["features"].index("log_metraz_m2")
    elastycznosc = post["beta"][:, i_metraz] / scaler.std[i_metraz]
    print(f"\n  Elastycznosc wzgledem metrazu: {elastycznosc.mean():.3f} "
          f"[{np.quantile(elastycznosc, 0.05):.3f}, {np.quantile(elastycznosc, 0.95):.3f}]")
    print("  (wzrost metrazu o 1% zmienia cene o tyle procent)")

    _naglowek("Poziom miasta")
    mu, tau, sig = post["mu_city"], post["sigma_district"], post["sigma"]
    print(f"  mu_miasto        {mu.mean():.3f} log-pln  (~{np.exp(mu.mean()):,.0f} zl)")
    print(f"  sigma_dzielnica  {tau.mean():.3f}  => rozrzut miedzy dzielnicami "
          f"~{100 * (np.exp(tau.mean()) - 1):.0f}%")
    print(f"  sigma            {sig.mean():.3f}  => rozrzut ofert o tych samych "
          f"cechach ~{100 * (np.exp(sig.mean()) - 1):.0f}%")
    print(f"  nu               {post['nu'].mean():.1f}  (male = ciezkie ogony)")

    _naglowek("Dzielnice wzgledem sredniej Krakowa")
    premie = predict.district_premiums(art)
    print(f"  {'dzielnica':<28}{'efekt':>8}{'90% CI':>22}")
    for nazwa, v in sorted(premie.items(), key=lambda x: -x[1]["mean"]):
        ci = f"[{v['q5']:+.1f}%, {v['q95']:+.1f}%]"
        print(f"  {nazwa:<28}{v['mean']:>+7.1f}%{ci:>22}")

    if art.get("ocena_testowa"):
        _wypisz_ocene(art["ocena_testowa"])


# --------------------------------------------------------------------------- #
#  predict
# --------------------------------------------------------------------------- #
def cmd_predict(args):
    art = inference.load(args.artifacts)

    if args.list_dzielnice:
        print("Dzielnice znane modelowi:")
        for nazwa in art["levels"]:
            print(f"  {nazwa}")
        return

    if not args.dzielnica:
        raise SystemExit("Podaj --dzielnica (lista: --list-dzielnice)")

    w = predict.value_flat(art, args.dzielnica, args.metraz, args.pokoje, args.pietro)

    if not w["known_district"]:
        print(f"UWAGA: '{args.dzielnica}' nie wystepuje w danych treningowych.")
        print("       Model uzyje rozkladu populacyjnego dzielnic - przedzial "
              "bedzie szerszy.\n")

    _naglowek(f"{args.dzielnica} | {args.metraz:g} m2 | {args.pokoje} pok. "
              f"| pietro {args.pietro}")
    print(f"  Wycena (mediana)      {w['listing']['median']:>12,.0f} zl"
          f"   ({w['listing']['median'] / args.metraz:,.0f} zl/m2)")
    print(f"  Przedzial 50%         {w['listing']['q25']:>12,.0f} - "
          f"{w['listing']['q75']:,.0f} zl")
    print(f"  Przedzial 90%         {w['listing']['q5']:>12,.0f} - "
          f"{w['listing']['q95']:,.0f} zl")
    print(f"\n  Dla porownania - niepewnosc SREDNIEJ w tym segmencie (90%):")
    print(f"                        {w['segment']['q5']:>12,.0f} - "
          f"{w['segment']['q95']:,.0f} zl")
    print("  Wezszy przedzial to niewiedza modelu; szerszy zawiera dodatkowo")
    print("  naturalny rozrzut miedzy konkretnymi ofertami.")


# --------------------------------------------------------------------------- #
def build_parser():
    p = argparse.ArgumentParser(
        prog="kfv",
        description="Bayesowski silnik wyceny mieszkan w Krakowie")
    p.add_argument("--artifacts", default=None,
                   help="katalog z artefaktami (domyslnie artifacts/)")
    sub = p.add_subparsers(dest="command", required=True)

    f = sub.add_parser("fit", help="dopasowanie modelu metoda MCMC")
    f.add_argument("--data", default=None, help="sciezka do krakow.csv")
    f.add_argument("--warmup", type=int, default=1000)
    f.add_argument("--samples", type=int, default=1000)
    f.add_argument("--chains", type=int, default=4)
    f.add_argument("--seed", type=int, default=0)
    f.set_defaults(func=cmd_fit)

    s = sub.add_parser("summary", help="czego nauczyl sie model")
    s.set_defaults(func=cmd_summary)

    pr = sub.add_parser("predict", help="wycena jednego mieszkania")
    pr.add_argument("--dzielnica", default="")
    pr.add_argument("--metraz", type=float, default=50.0)
    pr.add_argument("--pokoje", type=int, default=2)
    pr.add_argument("--pietro", type=int, default=2)
    pr.add_argument("--list-dzielnice", action="store_true",
                    help="wypisz dzielnice znane modelowi")
    pr.set_defaults(func=cmd_predict)

    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        args.func(args)
    except FileNotFoundError as e:
        # Brak danych albo artefaktow to normalna sytuacja uzytkownika, a nie
        # awaria programu - pokazujemy komunikat zamiast sladu stosu.
        raise SystemExit(str(e))


if __name__ == "__main__":
    main()
