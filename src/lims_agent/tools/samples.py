from langchain_core.tools import tool

_MOCK_SAMPLES: dict[str, dict] = {
    "AZ-8839": {"batch": "L-2026-X", "type": "Injectable", "pH": 7.1, "status": "In Analysis"},
    "BX-1022": {"batch": "L-2026-Y", "type": "Oral Solution", "pH": 5.5, "status": "Approved"},
}


@tool
def get_sample_data(sample_id: str) -> dict:
    """
    Fetch laboratory sample information such as batch, type, pH, and analysis status
    using a specific sample ID (e.g., 'AZ-8839').
    Use this tool whenever the user asks for details or data regarding a physical sample.
    """
    return _MOCK_SAMPLES.get(sample_id, {"error": "Sample ID not found"})
