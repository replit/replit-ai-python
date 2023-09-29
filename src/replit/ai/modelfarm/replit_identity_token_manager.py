import json
import os
import time
from typing import Optional
from dotenv import load_dotenv

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
        self.__set_token_type()
        if self.token_type != "L402":
            self.__update_token()

    def __set_token_type(self):
        """Sets the type of token to be used. Bearer if in replit, L402 if not."""
        if self.__in_deployment():
            self.token_type = "Bearer"
        else:
            self.token_type = "L402"

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
        try:
            if self.__in_deployment():
                return self.get_deployment_token()
            return self.get_interactive_token()
        # If the environment variables are not set, they're not in Replit, so use L402
        except MissingEnvironmentVariable:
            print("Using L402 token")
            return self.get_l402_token()

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

    def get_l402_token(self) -> str:
        load_dotenv()
        try:
            l402_token = self.get_env_var("REPLIT_L402_TOKEN")
            l402_preimage = self.get_env_var("REPLIT_L402_PREIMAGE")
            return "L402 " + l402_token + ":" + l402_preimage
        except MissingEnvironmentVariable:
            print("Generating new L402 token")
            res = requests.get("https://matador-replit.kody.repl.co")
            new_l402 = res.headers["WWW-Authenticate"]
            token, invoice = split_L402(new_l402)
            print("\n\n*** ERROR: Missing L402 Token ***")
            print(
                "To run the Replit modelfarm off of Replit in your local environment, you need to complete the following steps:"
            )
            print(
                "1. Pay the following lightning invoice using any lightning wallet like Alby or Phoenix:\n\n",
                invoice,
                "\n",
            )
            print(
                "2. After payment, you will receive a preimage. Replace the placeholder in your .env file with this preimage."
            )
            print(
                "3. This will complete your API key and allow you to run the Replit modelfarm locally by proxying through a Matador server on Replit."
            )
            print(
                "\nNote: This process uses the L402 protocol, a payment protocol for the Lightning Network. Bitcoin, used in this protocol, is a secure and efficient payment mechanism, enabling instant transactions with low fees.\n"
            )
            update_env_file(token, invoice)

    def __in_deployment(self) -> bool:
        """Determines if in deployment environement.

        Returns:
          bool: True if in the deployment environment, False otherwise.
        """
        return "REPLIT_DEPLOYMENT" in os.environ


def split_L402(l402: str) -> tuple[str, str]:
    """Splits L402 token into token and invoice.

    Args:
      l402 (str): L402 token. Formatted as "L402 token=token, invoice=invoice"

    Returns:
      tuple[str, str]: Tuple containing the token and invoice.
    """
    token = l402.split(",")[0].split("=")[1].replace('"', "")
    invoice = l402.split(",")[1].split("=")[1].replace('"', "")
    return token, invoice


def update_env_file(token: str, invoice: str):
    env_file_path = ".env"
    token_key = "REPLIT_L402_TOKEN"
    preimage_key = "REPLIT_L402_PREIMAGE"
    preimage_placeholder = "replace_me_with_preimage_after_paying_this_invoice"

    # Read existing .env file or create an empty one if it doesn't exist
    if os.path.exists(env_file_path):
        with open(env_file_path, "r") as f:
            lines = f.readlines()
    else:
        lines = []

    # Update or append the token and preimage lines
    lines = [
        f"{token_key}={token}\n" if line.startswith(token_key) else line
        for line in lines
    ]
    if not any(line.startswith(token_key) for line in lines):
        lines.append(f"{token_key}={token}\n")

    lines = [
        f"{preimage_key}={preimage_placeholder}\n"
        if line.startswith(preimage_key)
        else line
        for line in lines
    ]
    if not any(line.startswith(preimage_key) for line in lines):
        lines.append(f"{preimage_key}={preimage_placeholder} # {invoice}\n")

    # Write the updated lines back to the .env file
    with open(env_file_path, "w") as f:
        f.writelines(lines)
