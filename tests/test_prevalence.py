import numpy as np
import pytest

from analytics.prevalence import (
    OccurrenceModel, document_window, estimate_alpha, injection_test, main_text, mle_alpha,
    placebo_verdict, trend_placebo,
)


def synthetic_corpus(rng, n, p):
    vocabulary = np.array([f"w{chr(97 + i // 26)}{chr(97 + i % 26)}" for i in range(len(p))])
    return [list(vocabulary[rng.random(len(p)) < p]) for _ in range(n)]


@pytest.fixture
def corpora():
    rng = np.random.default_rng(0)
    p_human = rng.uniform(0.05, 0.4, 200)
    p_ai = np.clip(p_human * rng.uniform(0.6, 1.6, 200), 0.01, 0.9)
    return rng, p_human, p_ai


def test_mle_recovers_known_mixture_share(corpora):
    rng, p_human, p_ai = corpora
    model = OccurrenceModel.fit(synthetic_corpus(rng, 3000, p_human),
                                synthetic_corpus(rng, 3000, p_ai))
    for alpha in (0.0, 0.1, 0.3):
        n_ai = int(alpha * 2000)
        docs = synthetic_corpus(rng, 2000 - n_ai, p_human) + synthetic_corpus(rng, n_ai, p_ai)
        lh, la = model.log_likelihoods(docs)
        assert mle_alpha(la - lh) == pytest.approx(alpha, abs=0.03)


def test_mle_boundaries_and_interval():
    assert mle_alpha(np.full(50, -5.0)) == 0.0
    assert mle_alpha(np.full(50, 5.0)) == 1.0
    estimate = estimate_alpha(np.r_[np.full(80, -8.0), np.full(20, 8.0)], resamples=50)
    assert estimate["alpha"] == pytest.approx(0.2, abs=1e-3)
    assert estimate["ci_low"] <= estimate["alpha"] <= estimate["ci_high"]


def test_injection_test_reports_bias_per_share(corpora):
    rng = np.random.default_rng(1)
    rows = injection_test(rng.normal(-6, 2, 3000), rng.normal(6, 2, 500),
                          alphas=(0.0, 0.1), sample_size=500, repeats=5)
    assert [row["true_alpha"] for row in rows] == [0.0, 0.1]
    assert all(abs(row["bias"]) < 0.02 for row in rows)


def test_prose_window_drops_page_furniture_and_short_documents():
    page = "Home | About | Contact\nAccept cookies\n" + "This sentence has plenty of words in it. " * 40
    assert "cookies" not in main_text(page)
    assert len(document_window(page)) == 150
    assert document_window("Too short to use.") is None


def test_placebo_verdict_and_trend():
    near_zero = [{"period": "2020/10", "alpha": 0.004, "ci_low": 0.0, "ci_high": 0.02}]
    assert placebo_verdict(near_zero)["passed"]
    drifted = near_zero + [{"period": "2021/17", "alpha": 0.08, "ci_low": 0.05, "ci_high": 0.11}]
    assert placebo_verdict(drifted)["periods_above_tolerance"] == ["2021/17"]
    dates = np.arange("2013-01-01", "2021-01-01", 120, dtype="datetime64[D]")
    years = np.arange(len(dates)) * 120 / 365.25
    assert not trend_placebo(dates, 64 - 0.6 * years, resamples=200)["passed"]
    flat = 60 + np.random.default_rng(0).normal(0, 0.5, len(dates))
    assert trend_placebo(dates, flat, resamples=200)["passed"]


def test_confidence_belt_removes_a_constant_false_positive_floor():
    from analytics.prevalence import ConfidenceBelt
    rng = np.random.default_rng(2)
    human = np.r_[rng.normal(-6, 2, 2850), rng.normal(6, 2, 150)]  # 5% of humans look AI
    ai = rng.normal(6, 2, 1000)
    belt = ConfidenceBelt.build(human, ai, sample_size=1000, grid=np.linspace(0, 0.5, 26),
                                repeats=40)
    sample = np.r_[rng.choice(human, 900), rng.choice(ai, 100)]
    corrected = belt.correct(estimate_alpha(sample, resamples=20))
    assert corrected["raw_alpha"] == pytest.approx(0.145, abs=0.03)
    assert corrected["alpha"] == pytest.approx(0.10, abs=0.03)
    assert corrected["ci_low"] <= 0.10 <= corrected["ci_high"]
    assert belt.point(0.01) == 0.0


def test_dictionary_vocabulary_drops_proper_nouns_and_page_furniture(tmp_path):
    from analytics.prevalence import dictionary_vocabulary
    words = tmp_path / "words"
    words.write_text("Amazon\nsturdy\nquietly\ndon't\ntwitter\n")
    vocabulary = dictionary_vocabulary(str(words))
    assert vocabulary == {"sturdy", "quietly", "twitter"}
    assert "twittershare" not in vocabulary


def test_sparse_refit_matches_the_document_model(corpora):
    rng, p_human, p_ai = corpora
    human, ai = synthetic_corpus(rng, 500, p_human), synthetic_corpus(rng, 500, p_ai)
    model = OccurrenceModel.fit(human, ai)
    refit = OccurrenceModel.from_matrices(model.vocabulary, model.matrix(human), model.matrix(ai))
    np.testing.assert_allclose(refit.p_human, model.p_human)
    docs = synthetic_corpus(rng, 50, p_ai)
    lh, la = model.log_likelihoods(docs)
    np.testing.assert_allclose(model.log_ratio_matrix(model.matrix(docs)), la - lh, rtol=1e-9)
    doubled = OccurrenceModel.from_matrices(model.vocabulary, model.matrix(human + human),
                                            model.matrix(ai))
    weighted = OccurrenceModel.from_matrices(model.vocabulary, model.matrix(human),
                                             model.matrix(ai), human_weights=np.full(500, 2))
    np.testing.assert_allclose(weighted.p_human, doubled.p_human)


def test_cluster_resample_draws_whole_clusters():
    from analytics.prevalence import cluster_resample
    values = np.r_[np.zeros(100), np.ones(100)]
    clusters = np.r_[np.zeros(100, int), np.ones(100, int)]
    means = {cluster_resample(values, clusters, np.random.default_rng(seed)).mean()
             for seed in range(40)}
    assert means <= {0.0, 0.5, 1.0} and len(means) > 1
