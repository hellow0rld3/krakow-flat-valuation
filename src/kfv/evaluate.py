"""Evaluation of the model on a test set.

We measure two different things:

  point error (MAPE, RMSE) - how far off we are on the price on average
  calibration              - whether a 90% interval really covers 90% of prices

The engine is supposed to know its own uncertainty, so calibration matters more
than accuracy alone. A model that is off by 14% but admits it honestly is
useful. A model that is off by 10% and claims it is off by 2% is dangerous,
because someone will make a financial decision on it, trusting an interval that
has no backing in reality.

The module modifies and saves nothing, it only computes.
"""
import numpy as np

from kfv import predict
from kfv.data import GROUP, TARGET

# Levels at which we check the coverage. Several instead of one, because the
# model can be well calibrated in the middle of the distribution and badly in
# the tails - a single point would not show that.
COVERAGE_LEVELS = (0.50, 0.80, 0.90, 0.95)


def prepare_input(artifacts, df):
    """Pulls out of the frame what predict needs: raw features, indices, truth.

    We take the features in the order stored in the artifacts (and not from
    data.FEATURES), because it is the saved model that dictates which position
    of the vector corresponds to which beta weight.

    Districts go through predict.district_index, so an unknown district gets -1
    instead of an exception. When evaluating the model that is the correct
    behaviour: we want to know how it does also where it had no data.
    """
    X_raw = df[artifacts["features"]].to_numpy(dtype=np.float64)
    group_idx = np.array(
        [predict.district_index(d, artifacts["levels"]) for d in df[GROUP]],
        dtype=np.int32,
    )
    y_log = df[TARGET].to_numpy(dtype=np.float64)
    return X_raw, group_idx, y_log


def point_errors(true_prices, predicted_prices):
    """Classic accuracy metrics.

    We report both the mean and the median relative error, because the gap
    between them carries information in itself: a large difference means the
    error distribution is skewed, that is a few unusual listings spoil the mean.
    """
    relative_error = np.abs(predicted_prices - true_prices) / true_prices
    return {
        "MAPE": float(np.mean(relative_error)),
        "median_error": float(np.median(relative_error)),
        "RMSE_pln": float(np.sqrt(np.mean((predicted_prices - true_prices) ** 2))),
    }


def calibration(price_draws, true_prices, coverage_levels=COVERAGE_LEVELS):
    """For every level: what fraction of the true prices landed in the interval.

    An interval at level 0.9 is built from the 5% and 95% quantiles, that is
    symmetrically, cutting off half of the remainder on each side.

    How to read it:
      coverage ~ level  -> the model judges its own uncertainty honestly
      coverage < level  -> the model is overconfident
      coverage > level  -> the model is conservative, intervals too wide
    """
    result = []
    for level in coverage_levels:
        margin = (1.0 - level) / 2.0
        low, high = np.quantile(price_draws, [margin, 1.0 - margin], axis=0)
        covered = (true_prices >= low) & (true_prices <= high)
        result.append({
            "level": level,
            "coverage": float(np.mean(covered)),
            "mean_width_pln": float(np.mean(high - low)),
        })
    return result


def evaluate_on(artifacts, df, coverage_levels=COVERAGE_LEVELS, seed=0):
    """Full evaluation on the given frame (usually the test set).

    The point valuation is the median of the predictive distribution, not the
    mean. On the zloty scale the distribution is right-skewed, so the mean is
    pulled up by the tail of expensive listings and the median describes the
    typical price better.
    """
    X_raw, group_idx, y_log = prepare_input(artifacts, df)

    log_draws = predict.predictive_log_price(
        artifacts, X_raw, group_idx, include_noise=True, seed=seed)

    price_draws = np.exp(log_draws)
    true_prices = np.exp(y_log)
    predicted_prices = np.median(price_draws, axis=0)

    return {
        "n": int(len(true_prices)),
        "n_unknown_districts": int(np.sum(group_idx < 0)),
        **point_errors(true_prices, predicted_prices),
        "calibration": calibration(price_draws, true_prices, coverage_levels),
    }


def errors_by_district(artifacts, df, seed=0):
    """Median relative error broken down by district.

    Useful for diagnosis: the model can be good on average and yet be
    systematically wrong in one segment of the market. A mean over the whole set
    would hide that.
    """
    X_raw, group_idx, y_log = prepare_input(artifacts, df)
    log_draws = predict.predictive_log_price(
        artifacts, X_raw, group_idx, include_noise=True, seed=seed)

    true_prices = np.exp(y_log)
    predicted_prices = np.median(np.exp(log_draws), axis=0)
    error = np.abs(predicted_prices - true_prices) / true_prices

    result = {}
    for district in sorted(df[GROUP].unique()):
        mask = (df[GROUP] == district).to_numpy()
        result[district] = {
            "n": int(mask.sum()),
            "median_error": float(np.median(error[mask])),
        }
    return result
