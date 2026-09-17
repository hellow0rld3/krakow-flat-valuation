import numpy as np
import pytest

from kfv import data


@pytest.fixture
def datasets():
    """Builds the train/test sets once and shares them with all the tests.

    The test set is processed with the parameters from training (levels,
    scaler) - exactly the way prediction on new data will do it later.
    """
    df = data.prepare_dataset(data.load_data())
    train, test = data.split(df)
    tr = data.build(train)
    te = data.build(test, levels=tr["levels"], scaler=tr["scaler"])
    return {"train_df": train, "tr": tr, "test_df": test, "te": te}


def test_rows_match_the_source(datasets):
    """X, y and group_idx have to describe the same rows as the source frame.

    We check two things, each for a different reason:

    1. Shapes. A mismatch in length would blow up MCMC later anyway
       ("Incompatible shapes for broadcasting"), so here we only catch it
       earlier and more readably. The condition that really earns its place is
       y.shape == (N,): the shape (N, 1) raises nothing, it only quietly breaks
       the broadcasting when the predictive distribution is computed.

    2. Row-by-row correspondence. This is the case nothing else would catch:
       arrays of the right shape, but with the rows shuffled. It happens when
       someone filters or sorts one source and forgets about the others. The
       model will fit without complaining and match prices to the wrong flats.

    So for a few rows we check that y agrees with the price in the frame, that
    group_idx points at the right district, and that destandardized X
    reproduces the original features.
    """
    for df_key, dataset_key in [("train_df", "tr"), ("test_df", "te")]:
        df, dataset = datasets[df_key], datasets[dataset_key]
        n = len(dataset["y"])

        assert dataset["X"].shape == (n, len(data.FEATURES))
        assert dataset["y"].shape == (n,)
        assert dataset["group_idx"].shape == (n,)

        for i in (0, n // 3, n - 1):
            assert dataset["y"][i] == pytest.approx(df[data.TARGET].iloc[i])
            assert dataset["levels"][dataset["group_idx"][i]] == df[data.GROUP].iloc[i]

            restored = dataset["X"][i] * dataset["scaler"].std + dataset["scaler"].mean
            assert restored == pytest.approx(df[data.FEATURES].iloc[i].to_numpy())


def test_group_idx_stays_in_range(datasets):
    """Every district index has to point at an existing level 0..K-1.

    This test catches silent damage to the code: an unknown district turned
    into -1 (which in numpy means the last element) or a NaN converted to 0
    (that is, the first district). In both cases nothing blows up and the
    listing is quietly assigned to the wrong district.
    """
    tr, te = datasets["tr"], datasets["te"]
    k = len(tr["levels"])
    for dataset in (tr, te):
        assert dataset["group_idx"].min() >= 0
        assert dataset["group_idx"].max() < k


def test_standardization_does_not_leak(datasets):
    """The standardization parameters may come from the training set only.

    We check two things at once:

    1. mean of X_train ~ 0 and standard deviation ~ 1  -> the parameters really
       were computed on training,
    2. mean of X_test departs from zero                -> they were not
       computed on the test set.

    Point 2 is the actual leak detector. If someone "fixed" the code so that the
    scaler fits itself to the test set, point 1 would still pass (training
    always has mean zero), and every quality metric would come out too
    optimistic - with no error and no warning.

    The 0.01 threshold is arbitrary but safe: with a correct split the observed
    departure of the test means is around 0.25.
    """
    tr, te = datasets["tr"], datasets["te"]

    assert np.allclose(tr["X"].mean(axis=0), 0.0, atol=1e-8)
    assert np.allclose(tr["X"].std(axis=0), 1.0, atol=1e-8)

    assert np.abs(te["X"].mean(axis=0)).max() > 0.01
