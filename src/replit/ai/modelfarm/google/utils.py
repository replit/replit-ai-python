from typing import Any, Dict


def ready_parameters(parameters: Dict[str, Any]) -> Dict[str, Any]:
    """
    Private method to prep parameter dict keys to send to API.

    Args:
        parameters (Dict[str, Any]): Dictionary of parameters.

    Returns:
        Dict[str, any]: New dictionary with keys in correct format for API.
    """
    remap = {
        "max_output_tokens": "max_tokens",
        "candidate_count": "n",
        "stop_sequences": "stop",
    }
    return {remap.get(k, k): v for k, v in parameters.items()}
