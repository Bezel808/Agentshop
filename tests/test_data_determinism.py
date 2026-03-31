from manage_datasets import _stable_bucket


def test_stable_bucket_deterministic():
    a = _stable_bucket("asin_123", buckets=2)
    b = _stable_bucket("asin_123", buckets=2)
    assert a == b
    assert a in (0, 1)
