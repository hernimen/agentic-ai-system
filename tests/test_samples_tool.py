from lims_agent.tools.samples import get_sample_data


def test_known_sample():
    result = get_sample_data.invoke({"sample_id": "AZ-8839"})
    assert result["batch"] == "L-2026-X"


def test_unknown_sample():
    assert get_sample_data.invoke({"sample_id": "ZZ-0000"}) == {"error": "Sample ID not found"}
