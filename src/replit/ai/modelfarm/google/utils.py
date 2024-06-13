from typing import Any, Dict

_PROVIDER_EXTRA_PARAMS = {"context", "examples", "top_k"}


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
    params = {remap.get(k, k): v for k, v in parameters.items()}

    provider_extra_parameters = {
        k: params.pop(k)
        for k in _PROVIDER_EXTRA_PARAMS if k in params
    }
    params["provider_extra_parameters"] = provider_extra_parameters

    return params
