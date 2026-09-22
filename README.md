# krakow-flat-valuation

A hierarchical Bayesian valuation engine for flats in Kraków. Given a district,
a floor area, a number of rooms and a floor, it returns a full predictive
distribution of the price rather than a single number.

Most valuation models answer "how much is this flat worth?" with one figure and
leave you to guess how much to trust it. This one answers with a distribution,
so the answer carries its own uncertainty. The model is hierarchical: districts
with few listings borrow strength from the rest of the city, which keeps their
estimates stable instead of letting a handful of offers dictate them. Asked
about a district it has never seen, it still answers, drawing the district
effect from the population of Kraków districts and widening the interval to say
how much less it knows.

## Results

804 training listings, 202 test listings, seed 0.

| Metric | Value |
|---|---|
| MAPE | 14.7% |
| Median error | 10.5% |
| RMSE | 186,877 PLN |
| Worst R-hat | 1.0007 |
| Lowest ESS | 552 |
| Divergences | 0 |

Calibration on the test set:

| Nominal | Empirical coverage | Mean interval width |
|---|---|---|
| 50% | 54.0% | 179,518 PLN |
| 80% | 84.2% | 375,524 PLN |
| 90% | 91.1% | 529,407 PLN |
| 95% | 96.0% | 703,740 PLN |

Calibration is the number that matters here. A 90% interval covers 91.1% of the
test prices, so the intervals mean roughly what they claim to mean — which is
what makes the point estimate usable in the first place. Every level sits a
little above nominal, so the model is mildly conservative: its intervals are
slightly wider than they strictly need to be. That is the direction one would
rather err in, but it is a bias, not a perfect result.

## Installation

```bash
git clone https://github.com/hellow0rld3/krakow-flat-valuation.git
cd krakow-flat-valuation
python3 -m venv .venv
source .venv/bin/activate
pip install numpyro pandas pytest
```

Developed on Python 3.13 with NumPyro 0.21, JAX 0.11 and pandas 3.0.

## Usage

```bash
PYTHONPATH=src python -m kfv.cli fit
```

Fits the model to `data/processed/krakow.csv`, prints the convergence
diagnostics and the evaluation on the held-out test set, and writes the result
to `artifacts/` as `posterior.npz` and `meta.json`. Takes about six seconds on a
laptop.

```bash
PYTHONPATH=src python -m kfv.cli summary
```

Shows what the model has learned: the effect of each feature on the price, the
city-level parameters, and a ranking of the districts against the Kraków
average.

```bash
PYTHONPATH=src python -m kfv.cli predict --district Krowodrza --area 55 --rooms 3 --floor 2
```

```
Krowodrza | 55 m2 | 3 rooms | floor 2
-------------------------------------
  Valuation (median)         902,776 PLN   (16,414 PLN/m2)
  50% interval               808,306 - 1,010,145 PLN
  90% interval               652,913 - 1,239,222 PLN

  For comparison - uncertainty about the average in this segment (90%):
                             870,391 - 939,240 PLN
```

The two intervals answer two different questions. The narrow one, 870k–939k,
covers the average price of flats with these features in Krowodrza: it reflects
only what the model does not yet know, and it shrinks as more listings are
collected. The wide one, 653k–1,239k, is the valuation of an individual flat,
and it also contains the spread between listings that share every feature the
model can see — finish, view, how urgently the owner wants to sell. That spread
does not shrink with more data, because no amount of listings makes two flats
identical. When valuing a specific flat, the wide interval is the honest one.

`--list-districts` prints the districts known to the model.

## How the model works

```
log(price) ~ StudentT(nu, alpha[district] + X @ beta, sigma)
alpha[k]    = mu_city + sigma_district * z[k]
```

- The features are log floor area, floor and number of rooms. Price and area
  both enter as logarithms, so beta for the area reads directly as an
  elasticity: a 1% larger flat costs a fixed percentage more, regardless of
  whether it is a studio or a penthouse.
- The likelihood is Student's t rather than normal. Its heavy tails let the
  model treat an unusually cheap or expensive listing as an outlier instead of
  letting it drag the whole fit, and `nu` is estimated from the data — the
  degree of robustness is learned, not chosen by hand.
- `alpha[k]` is the price level of district k. It is neither one number shared
  by all of Kraków (which would erase real differences between districts) nor 19
  unrelated numbers (which would let a district with 11 listings say whatever
  those 11 listings happen to say). Every `alpha[k]` is drawn from a common
  distribution whose spread `sigma_district` is itself estimated, so the data
  decide how much the districts are allowed to differ. That is partial pooling.
- The hierarchy uses the non-centered parameterization: `z ~ N(0, 1)`
  independent of `sigma_district`, which decouples the two levels and avoids
  Neal's funnel. On this dataset the centered version actually samples just as
  well, since no district is small enough to create the funnel. The
  non-centered form is kept as a safeguard for the case where districts get
  thinner — more of them, or a filtered subset.
- Fitting uses NUTS with 4 chains. R-hat, effective sample size and divergences
  are checked automatically after every fit, and the CLI says plainly whether
  the results can be interpreted.

## Data

The dataset was collected with a scraper of my own for OLX and Otodom listings.
`data/processed/krakow.csv` holds 1014 offers across 19 districts, all captured
on 2026-08-05; re-running the scraper extends it. The column names are in
Polish because they come straight from the scraper.

These are asking prices, not transaction prices. The final price agreed after
negotiation is typically lower, so the model values what a flat is listed at,
not what it sells for. Transaction data would be the better target, but it is
not publicly available at this granularity.

## Project structure

```
src/kfv/
    data.py        loading, district encoding, train/test split, standardization
    model.py       the NumPyro model
    inference.py   running MCMC, convergence diagnostics, saving artifacts
    predict.py     predictive distribution of the price
    evaluate.py    point metrics and calibration
    cli.py         terminal interface
tests/
data/processed/    the dataset
artifacts/         the fitted posterior (not in the repo)
```

`cli.py` holds no logic of its own — it parses arguments, calls the other
modules and formats the output. Everything the program does can therefore be
tested without going through the CLI, and adding a web interface later means
writing a second thin adapter next to this one rather than rewriting the
project.

## Tests

```bash
pytest                  # everything
pytest -m "not slow"    # without the parameter recovery test
```

The suite is built around one question: what kind of mistake would still let
the code run and return plausible numbers?

The parameter recovery test is the answer for the model itself. It generates
data from known parameter values using the same `model()` function that is later
fitted, then checks that all 26 true values land inside their 95% posterior
intervals and that the chains converged. A flipped sign, a mistaken prior or
wrong district indexing does not raise an exception — it quietly produces a
different model that still fits something. Nothing but recovery catches that.
It is marked `slow` because it runs MCMC.

The data tests cover the same class of failure one layer down. Shapes and
indices are checked not because a mismatch would crash — it usually would — but
because the cases that do not crash are the dangerous ones: a `y` of shape
`(N, 1)` that silently breaks broadcasting later, rows of `X` and `y` that no
longer line up after a filter, a district index of `-1` that numpy happily reads
as the last district. The standardization test verifies that the scaler is
fitted on training data only, and it does so by asserting that the test means
depart from zero — the reverse check would pass even if the scaler leaked.

The prediction tests run against a synthetic posterior with hand-chosen values
instead of the real artifacts, so they need no MCMC, survive a fresh clone, and
can assert exact numbers. The most important of them shuffles the feature order
in the artifacts and checks that the prediction follows it, since a mismatch
there would multiply the number of rooms by the weight of the floor without any
error at all.

## Limitations

- These are asking prices, not transaction prices.
- Three features only. Year of construction, condition, building type and exact
  location within a district are all absent, and all of them matter.
- The data come from a single day, so the model has no notion of a price trend
  over time.
- 46 listings (4.5%) have the district `nieznana` — an artifact of the scraper,
  which the model currently treats as if it were a real district.
- The smallest district has 11 listings. Partial pooling is what makes such a
  district usable at all, but its estimate is still driven mostly by the city
  prior.
- Validation stops at parameter recovery. Simulation-based calibration and a
  prior-posterior contraction check are not implemented yet.
- All reported numbers come from a single train/test split with seed 0.

## Roadmap

- Packaging: `pyproject.toml`, so that `pip install -e .` replaces `PYTHONPATH=src`.
- Split `--seed` into `--seed` and `--split-seed`, so the split can be held fixed
  while the sampler seed varies.
- Simulation-based calibration and a prior-posterior contraction check.
- More features, once the scraper collects them.

## License

MIT — see [LICENSE](LICENSE).
