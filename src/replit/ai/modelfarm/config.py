from dataclasses import dataclass


@dataclass
class Config:
    """Config for the Model Farm API client."""

    rootUrl: str = "https://production-modelfarm.replit.com"
    matadorUrl: str = "https://matador-replit.kody.repl.co/replit"
    audience: str = "modelfarm@replit.com"


_config = Config()


def initialize(rootUrl=None, serverAudience=None):
    """Initializes the global config for the Model Farm API client."""
    if rootUrl:
        _config.rootUrl = rootUrl
    if serverAudience:
        _config.audience = serverAudience


def get_config(token_type: str) -> Config:
    """
    Returns the global config for the Model Farm API client.
    If the token type is L402, the rootUrl is set to the Matador URL for L402 payments.

    Parameters:
        token_type (str): The type of token used, Bearer or L402.

    Returns:
        Config: the global config for the Model Farm API client.
    """
    if token_type == "L402":
        _config.rootUrl = _config.matadorUrl
    return _config
