# Lab notes

Raw record of the experiments run while building the model. Each entry is the
setup, the measured numbers and the caveats — the material the report will be
written from.

---

## 1. Does a wrong prior break the model?

**2026-08-18. Run on the TFP prototype (`Documents/Bayesian-Engine`), not on this
repository.** The dataset there had 855 training listings; the model is the same
in structure but the implementation and the split differ, so the numbers are not
directly comparable with the ones in the README.

The question: how much does the prior on `mu_city` actually matter? The prior
was deliberately mis-centred on ~268,000 PLN, against a true training mean of
772,502 PLN (log 13.557).

| Variant | Prior on `mu_city` | Posterior `mu_city` | MAPE | 90% coverage |
|---|---|---|---|---|
| A | `N(13.557, 1)` — wide, correct | 13.539 → 758,549 PLN | 13.8% | 90.7% |
| B | `N(12.5, 0.05)` — narrow and wrong | 12.560 → 284,870 PLN | 13.9% | 91.4% |
| C | `N(12.5, 1)` — wide but wrong | 13.541 → 759,796 PLN | 13.7% | 90.7% |

Predictive accuracy is unchanged across all three, including B, where the
posterior for `mu_city` sits at a third of the true value.

Why B still predicts correctly — the internals:

| | Variant A | Variant B |
|---|---|---|
| `mu_city` | 13.539 | 12.560 |
| `sigma_district` | 0.126 | 0.928 |
| mean `z` | −0.007 | +1.072 |
| `alpha[Krowodrza]` | 13.652 | 13.654 |

The model compensates. With `mu` pinned at 12.56, the `z` values move up by
~1.07 and `sigma_district` grows from 0.13 to 0.93; the product reproduces the
same `alpha` to three decimal places. The likelihood only ever sees `alpha`, not
`mu` and `z` separately, so the data do not determine the split of
`alpha_k = mu + tau * z_k` into its parts. This is weak identifiability.

What variant B does destroy is interpretation: `sigma_district = 0.928` reads as
"the districts of Kraków differ by ~153%" (true value ~14%), and the district
ranking, computed as `exp(alpha_k - mu)`, would report Krowodrza at +199%
instead of +11.6%.

**Caveat on validity:** these runs used 2 chains × 400 samples and reached R-hat
1.073 (A) and 1.049 (B), both above the 1.01 threshold used elsewhere in the
project. Enough to compare the mechanism, not enough to quote as a result.

**Conclusion:** A good prior is not necessary for prediction, but it is necessary
for interpretation — and interpretation is what this project is for, since a
statement like "Zwierzyniec is 25% more expensive than the city average" is only
meaningful if `mu` really means the city average. A wide prior costs nothing:
variant C started from a badly wrong location and the data pulled it back. A
narrow one is a strong claim, and when the claim is false the model has no way
to back out of it.

---

## 2. Centered vs non-centered parameterization

**2026-09-05. Run on this repository's model, 1 chain.**

The question: is the non-centered parameterization actually necessary here, or
is it cargo-culted from the textbook?

| 1 chain | Non-centered | Centered |
|---|---|---|
| Divergences | 0 | 0 |
| `n_eff` for `mu_city` | 148 | 1434 |
| `n_eff` for `sigma_district` | 185 | 1231 |

The centered version — the one the standard advice warns against — samples about
eight times more efficiently, and neither version produces a single divergence.

The reason is the amount of data per group. With few observations per group the
likelihood constrains `alpha_k` weakly, the shape of the posterior is dictated by
the prior `Normal(mu, tau)`, and Neal's funnel forms: when `tau` is small all
`alpha` must crowd around `mu`, when it is large they may spread out, and no
single step size handles both regions. With many observations per group the
likelihood pins each `alpha_k` to its own data, the funnel never forms, and the
non-centered form only adds an indirection between `z` and `alpha`. This project
has 804 listings across 19 districts, about 42 per district. That is a lot.

Mechanism described in Betancourt & Girolami, *Hamiltonian Monte Carlo for
Hierarchical Models*.

**Conclusion:** the textbook rule — always prefer the non-centered
parameterization in a hierarchical model — is conditional, and on this dataset
the condition does not hold. The rule is about a geometry that appears when
groups are thin, and with ~42 listings per district the geometry never appears.
Nothing in the output would have revealed this: both versions converge, both
report zero divergences, and only the effective sample size tells them apart.
The rule had to be tested against the data to find out that it did not apply.

**Decision:** keep the non-centered version anyway, as a safeguard for a finer
grouping (estates instead of districts, a handful of listings per group), where
the funnel would appear and the centered version would start to break. The price
is sampling efficiency on a model that already fits in six seconds — cheap
insurance against having to revisit this later.

---

## 3. Mutation audit of the test suite

**2026-09-13. Run on a copy of this repository in a temporary directory; the
repository itself was untouched.**

The question: do the tests actually discriminate, or do they pass by inertia?
For each test, a bug that *should* break it was introduced deliberately.

| Mutation | Test failed? |
|---|---|
| `build`: `y` sorted (rows permuted) | yes |
| `encode`: first row given index −1 | yes |
| `build`: scaler always refitted (leakage) | yes |
| `model`: `alpha = mu + z` (`sigma_district` disconnected) | **no** |
| `test_model`: warmup shortened to 3 steps | yes |
| `predict`: feature order hardcoded | yes |
| `predict`: always `alpha` of the first district | yes |
| `predict`: unknown district without drawing a new effect | yes |
| `predict`: `include_noise` ignored | yes |

Eight of nine caught.

The gap: disconnecting `sigma_district` from the model leaves the parameter
recovery test passing.

| | 95% interval | sd |
|---|---|---|
| Posterior `sigma_district` after the mutation | [0.013, 1.112] | 0.290 |
| Prior `HalfNormal(0.5)` | [0.017, 1.118] | 0.298 |

The posterior is indistinguishable from the prior — the data said nothing about
that parameter — and the test still passes, because the true value (0.15) lies
inside the prior's wide interval. A parameter the model does not use passes a
recovery test as long as its prior is wide enough.

The fix is one assertion, prior-posterior contraction:

```
contraction = 1 - sd(posterior) / sd(prior)
```

Near zero means the data carried no information about the parameter. After the
mutation it would be ≈ 0.03; in the correct model `sigma_district` has sd ~0.03
against a prior sd of 0.30, so contraction ≈ 0.9. Requiring contraction > 0.5 for
every parameter would close the gap.

**Status:** not implemented, deferred to 0.2. The docstring of the recovery test
already states the limitation.

**Conclusion:** a suite that passes is evidence of nothing until it has been
shown that it can fail. Deliberately breaking the code turned nine passing tests
into eight that discriminate and one that was passing for the wrong reason. The
narrower lesson is the more useful one: parameter recovery cannot validate a
parameter the model never uses, because a wide prior already contains the truth.
Whether the data actually moved the posterior is a separate question and needs a
separate assertion.

