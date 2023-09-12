from dataclasses import dataclass


@dataclass
class Config:
    """Config for the Model Farm API client."""

    rootUrl: str = "https://production-modelfarm.replit.com"
    audience: str = "modelfarm@replit.com"


_config = Config()


def initialize(rootUrl=None, serverAudience=None):
    """Initializes the global config for the Model Farm API client."""
    if rootUrl:
        _config.rootUrl = rootUrl
    if serverAudience:
        _config.audience = serverAudience


def get_config() -> Config:
    """Returns the global config for the Model Farm API client.

    Returns:
        Config: the global config for the Model Farm API client.
    """
    return _config
