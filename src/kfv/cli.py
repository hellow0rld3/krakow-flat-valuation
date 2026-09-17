"""Terminal interface.

    python -m kfv.cli fit
    python -m kfv.cli summary
    python -m kfv.cli predict --district Krowodrza --area 55 --rooms 3 --floor 2

This module holds no logic - it only parses arguments, calls functions from the
other modules and formats the result. Thanks to that the whole functionality can
be tested without running the CLI, and adding a web interface later will mean
writing a second thin adapter next to this one, not rewriting the project.
"""
import argparse

import numpy as np
import numpyro

from kfv import data, evaluate, inference, predict


def _header(text):
    print(f"\n{text}")
    print("-" * len(text))


# --------------------------------------------------------------------------- #
#  fit
# --------------------------------------------------------------------------- #
def cmd_fit(args):
    # Has to come before the first use of jax, otherwise the chains will run one
    # after another instead of in parallel (importing jax does not initialize
    # the devices yet, so this place is safe).
    numpyro.set_host_device_count(args.chains)

    df = data.prepare_dataset(data.load_data(args.data))
    train, test = data.split(df, seed=args.seed)
    dataset = data.build(train)

    print(f"Data: {len(train)} training / {len(test)} test, "
          f"{len(dataset['levels'])} districts")
    print(f"Sampling: {args.chains} chains x {args.samples} samples "
          f"(+{args.warmup} warmup)...")

    mcmc = inference.fit(dataset, num_warmup=args.warmup, num_samples=args.samples,
                         num_chains=args.chains, seed=args.seed)

    diag = inference.diagnostics(mcmc)
    _header("Convergence diagnostics")
    print(f"  worst R-hat   {diag['max_rhat']:.4f}  ({diag['max_rhat_param']})")
    print(f"  lowest ESS    {diag['min_ess']:.0f}  ({diag['min_ess_param']})")
    print(f"  divergences   {diag['divergences']}")

    if inference.converged(diag):
        print("  -> the results can be interpreted")
    else:
        print("  -> warning: the chains have not converged. Increase "
              "--warmup/--samples before interpreting the results.")

    if len(test) > 0:
        result = evaluate.evaluate_on(artifacts_in_memory(mcmc, dataset), test)
        _print_evaluation(result)
    else:
        result = None

    directory = inference.save(mcmc, dataset, args.artifacts,
                               extra={"test_evaluation": result})
    print(f"\nSaved the artifacts to {directory}")


def artifacts_in_memory(mcmc, dataset):
    """Assembles the artifact structure without writing to disk.

    Needed because we want to print the evaluation before saving - if the model
    had not converged, we would rather know about it before overwriting the
    previous results.
    """
    return {
        "posterior": {k: np.asarray(v) for k, v in mcmc.get_samples().items()},
        "levels": dataset["levels"],
        "scaler": dataset["scaler"],
        "features": data.FEATURES,
    }


def _print_evaluation(result):
    _header(f"Evaluation on the test set ({result['n']} listings)")
    print(f"  MAPE              {100 * result['MAPE']:>6.1f}%")
    print(f"  median error      {100 * result['median_error']:>6.1f}%")
    print(f"  RMSE              {result['RMSE_pln']:>9,.0f} PLN")
    print("\n  Calibration (the closer to nominal, the better):")
    print(f"  {'nominal':>12}{'empirical':>14}{'mean width':>20}")
    for k in result["calibration"]:
        print(f"  {100 * k['level']:>11.0f}%{100 * k['coverage']:>13.1f}%"
              f"{k['mean_width_pln']:>16,.0f} PLN")


# --------------------------------------------------------------------------- #
#  summary
# --------------------------------------------------------------------------- #
def cmd_summary(args):
    art = inference.load(args.artifacts)
    post = art["posterior"]
    scaler = art["scaler"]

    _header("Effect of the features on the price")
    print("  (beta refers to standardized features: the effect of a change by "
          "one standard deviation)")
    for i, feature in enumerate(art["features"]):
        b = post["beta"][:, i]
        print(f"  {feature:<16} {b.mean():+.3f}  [{np.quantile(b, 0.05):+.3f}, "
              f"{np.quantile(b, 0.95):+.3f}]   => {100 * (np.exp(b.mean()) - 1):+.1f}% of the price")

    # Elasticity: beta is computed on features divided by the standard
    # deviation, so to get back to "% of price per % of area" we have to
    # multiply by it.
    i_area = art["features"].index("log_metraz_m2")
    elasticity = post["beta"][:, i_area] / scaler.std[i_area]
    print(f"\n  Elasticity with respect to area: {elasticity.mean():.3f} "
          f"[{np.quantile(elasticity, 0.05):.3f}, {np.quantile(elasticity, 0.95):.3f}]")
    print("  (a 1% increase in area changes the price by that many percent)")

    _header("City level")
    mu, tau, sig = post["mu_city"], post["sigma_district"], post["sigma"]
    print(f"  mu_city          {mu.mean():.3f} log-pln  (~{np.exp(mu.mean()):,.0f} PLN)")
    print(f"  sigma_district   {tau.mean():.3f}  => spread between districts "
          f"~{100 * (np.exp(tau.mean()) - 1):.0f}%")
    print(f"  sigma            {sig.mean():.3f}  => spread of listings with the "
          f"same features ~{100 * (np.exp(sig.mean()) - 1):.0f}%")
    print(f"  nu               {post['nu'].mean():.1f}  (small = heavy tails)")

    _header("Districts against the Krakow average")
    premiums = predict.district_premiums(art)
    print(f"  {'district':<28}{'effect':>8}{'90% CI':>22}")
    for name, v in sorted(premiums.items(), key=lambda x: -x[1]["mean"]):
        ci = f"[{v['q5']:+.1f}%, {v['q95']:+.1f}%]"
        print(f"  {name:<28}{v['mean']:>+7.1f}%{ci:>22}")

    if art.get("test_evaluation"):
        _print_evaluation(art["test_evaluation"])


# --------------------------------------------------------------------------- #
#  predict
# --------------------------------------------------------------------------- #
def cmd_predict(args):
    art = inference.load(args.artifacts)

    if args.list_districts:
        print("Districts known to the model:")
        for name in art["levels"]:
            print(f"  {name}")
        return

    if not args.district:
        raise SystemExit("Give --district (for a list: --list-districts)")

    w = predict.value_flat(art, args.district, args.area, args.rooms, args.floor)

    if not w["known_district"]:
        print(f"Warning: '{args.district}' does not appear in the training data.")
        print("         The model will fall back on the population distribution "
              "of districts - the interval will be wider.\n")

    _header(f"{args.district} | {args.area:g} m2 | {args.rooms} rooms "
            f"| floor {args.floor}")
    print(f"  Valuation (median)    {w['listing']['median']:>12,.0f} PLN"
          f"   ({w['listing']['median'] / args.area:,.0f} PLN/m2)")
    print(f"  50% interval          {w['listing']['q25']:>12,.0f} - "
          f"{w['listing']['q75']:,.0f} PLN")
    print(f"  90% interval          {w['listing']['q5']:>12,.0f} - "
          f"{w['listing']['q95']:,.0f} PLN")
    print("\n  For comparison - uncertainty about the average in this segment (90%):")
    print(f"                        {w['segment']['q5']:>12,.0f} - "
          f"{w['segment']['q95']:,.0f} PLN")
    print("  The narrower interval is what the model does not know; the wider one")
    print("  also covers the natural spread between individual listings.")


# --------------------------------------------------------------------------- #
def build_parser():
    p = argparse.ArgumentParser(
        prog="kfv",
        description="Bayesian flat valuation engine for Krakow")
    p.add_argument("--artifacts", default=None,
                   help="directory with the artifacts (artifacts/ by default)")
    sub = p.add_subparsers(dest="command", required=True)

    f = sub.add_parser("fit", help="fit the model with MCMC")
    f.add_argument("--data", default=None, help="path to krakow.csv")
    f.add_argument("--warmup", type=int, default=1000)
    f.add_argument("--samples", type=int, default=1000)
    f.add_argument("--chains", type=int, default=4)
    f.add_argument("--seed", type=int, default=0)
    f.set_defaults(func=cmd_fit)

    s = sub.add_parser("summary", help="what the model has learned")
    s.set_defaults(func=cmd_summary)

    pr = sub.add_parser("predict", help="valuation of a single flat")
    pr.add_argument("--district", default="")
    pr.add_argument("--area", type=float, default=50.0)
    pr.add_argument("--rooms", type=int, default=2)
    pr.add_argument("--floor", type=int, default=2)
    pr.add_argument("--list-districts", action="store_true",
                    help="list the districts known to the model")
    pr.set_defaults(func=cmd_predict)

    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        args.func(args)
    except FileNotFoundError as e:
        # Missing data or artifacts is a normal situation for the user, not a
        # crash of the program - we show a message instead of a stack trace.
        raise SystemExit(str(e))


if __name__ == "__main__":
    main()
