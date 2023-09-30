import json
import os
import time
from typing import Optional

import requests
from replit.ai.modelfarm.config import get_config
from replit.ai.modelfarm.identity.sign import SigningAuthority


class MissingEnvironmentVariable(Exception):
    pass


class ReplitIdentityTokenManager:
    def __init__(self, token_timeout: int = 300):
        """Initializes a new instance of ReplitIdentityTokenManager

        Args:
          token_timeout (int): The timeout in seconds for the token. Default is 300 seconds.
        """
        self.token_timeout = token_timeout
        self.last_update: Optional[float] = None
        self.token: Optional[str] = None
        self.__update_token()

    def get_token(self) -> Optional[str]:
        """Returns the token, updates if the current token has expired.

        Returns:
          str: The token.
        """
        if (
            self.last_update is None
            or self.last_update + self.token_timeout < time.time()
        ):
            self.__update_token()
        return self.token

    def __update_token(self):
        """Updates the token and the last_updated time."""
        self.token = self.get_new_token()
        self.last_update = time.time()

    def get_new_token(self) -> str:
        """Gets the most recent token.

        Returns:
          str: The most recent token.
        """
        if self.__in_deployment():
            return self.get_deployment_token()
        return self.get_interactive_token()

    def get_deployment_token(self) -> str:
        """Fetches deployment token from hostindpid1.

        Returns:
          str: Deployment token.
        """
        response = requests.post(
            "http://localhost:1105/getIdentityToken",
            json={"audience": get_config().audience},
        )
        return json.loads(response.content)["identityToken"]

    @classmethod
    def get_env_var(cls, var: str) -> Optional[str]:
        if var in os.environ:
            return os.environ[var]
        else:
            raise MissingEnvironmentVariable(
                f"Did not find the environment variable: {var}"
            )

    def get_interactive_token(self) -> str:
        """Generates and returns an identity token"

        Returns:
          str: Interactive token.
        """
        gsa = SigningAuthority(
            marshaled_private_key=self.get_env_var("REPL_IDENTITY_KEY"),
            marshaled_identity=self.get_env_var("REPL_IDENTITY"),
            replid=self.get_env_var("REPL_ID"),
        )
        signed_token = gsa.sign(audience=get_config().audience)
        return signed_token

    def __in_deployment(self) -> bool:
        """Determines if in deployment environement.

        Returns:
          bool: True if in the deployment environment, False otherwise.
        """
        return "REPLIT_DEPLOYMENT" in os.environ
