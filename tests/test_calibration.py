from pipeline.calibration import assess_cameras, failed


def camera(ratio):
    return {"focal_ratio": ratio}


def test_ultrawide_is_review_not_failure():
    metrics = {"cameras": [camera(.168)]}
    assert assess_cameras(metrics["cameras"]) == "wide_angle_review"
    assert not failed(metrics)


def test_extreme_focal_is_failure():
    assert assess_cameras([camera(.099)]) == "failed"
    assert assess_cameras([camera(3.01)]) == "failed"
    assert failed({"calibration_status": "failed", "cameras": [camera(.5)]})


def test_typical_estimate_is_not_review():
    assert assess_cameras([camera(.8)]) == "estimated"
