from typing import Dict, Any


def ready_parameters(parameters: Dict[str, Any]) -> Dict[str, Any]:
    """
    Private method to prep parameter dict keys to send to API.

    Args:
        parameters (Dict[str, Any]): Dictionary of parameters.

    Returns:
        Dict[str, any]: New dictionary with keys in correct format for API.
    """
    remap = {
        "max_output_tokens": "maxOutputTokens",
        "top_k": "topK",
        "top_p": "topP",
        "candidate_count": "candidateCount",
        "stop_sequences": "stopSequences",
    }
    return {remap.get(k, k): v for k, v in parameters.items()}
