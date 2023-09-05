from dataclasses import dataclass


@dataclass
class Config:
  rootUrl: str = "http://staging-modelfarm.ai.gcp.replit.com"
  audience: str = "modelfarm@replit.com"


_config = Config()


def initialize(rootUrl=None, serverAudience=None):
  if rootUrl:
    _config.rootUrl = rootUrl
  if serverAudience:
    _config.audience = serverAudience


def get_config():
  return _config
