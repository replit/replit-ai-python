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


class L402TokenManager:
    def __init__(
        self,
        matador_url: Optional[str] = None,
    ):
        """Initializes a new instance of L402TokenManager

        Args:
          matador_url (str): The Matador URL for out of replit use to generate a new L402 if required. Default is None.
          l402 (str): The L402 API key for out of replit use (paid in bitcoin). Default is None.
        """
        self.matador_url = (
            matador_url if matador_url is not None else get_config().matadorUrl
        )
        self.last_update: Optional[float] = None
        self.token: Optional[str] = self.get_env_var("REPLIT_L402")
        self.token_type = "L402"
        self.__update_token()

    def get_token(self) -> Optional[str]:
        """Returns the L402"""
        return self.token

    def get_token_type(self):
        """Returns the token type"""
        return self.token_type

    @classmethod
    def get_env_var(cls, var: str) -> Optional[str]:
        """Returns the environment variable if it exists."""
        load_dotenv()
        if var in os.environ:
            return os.environ[var]
        else:
            None

    def __update_token(self):
        """Updates the token and the last_updated time."""
        self.token = self.get_L402_token()
        self.last_update = time.time()

    def get_L402_token(self) -> str:
        """Returns L402 if set, otherwise generates a new L402 token."""
        if self.token is not None:
            return self.token

        token, invoice = self.generate_L402()

        printInstructions(invoice)

        preimage = input("Enter preimage: ")

        if not preimage:
            preimage = self.l402_placeholder(token)

        update_dotenv(token, preimage)

        self.token = f"{token}:{preimage}"
        return f"{token}:{preimage}"

    def l402_placeholder(self, token):
        """Returns the placeholder for the preimage and prints instructions."""
        preimage = "replace_me_with_preimage_after_paying_this_invoice"
        print(
            "You did not enter a preimage. Replace the placeholder before setting the REPLIT_L402 environment variable."
        )
        print("Then run the following command:")
        print(f"echo 'REPLIT_L402=\"{token}:{preimage}\"' >> .env && source .env\n")

        raise Exception("Missing L402 Token")
        return preimage

    def generate_L402(self):
        """Generates a new L402 token using the Matador API."""
        res = requests.get(self.matador_url + "/new-L402")
        new_L402 = res.headers["WWW-Authenticate"]
        token, invoice = split_L402(new_L402)
        return token, invoice


def update_dotenv(token, preimage):
    """Updates the .env file with the L402 token."""
    response = input("Do you want to add REPLIT_L402 to the .env file? [Y/n]: ")
    if response.lower() in ["yes", "y", ""]:
        with open(".env", "r") as file:
            lines = file.readlines()

        # Remove the existing REPLIT_L402 line if it exists
        lines = [line for line in lines if not line.startswith("REPLIT_L402")]
        lines.append(f'REPLIT_L402="{token}:{preimage}"\n')

        with open(".env", "w") as file:
            file.writelines(lines)

        print("REPLIT_L402 has been updated in the .env file.")

    else:
        print("REPLIT_L402 has not been updated in the .env file.")


def printInstructions(invoice):
    """Prints instructions for paying the L402 invoice."""
    error_message = "\n\n*** ERROR: Missing L402 Token ***"
    instructions = (
        "To run the Replit modelfarm off of Replit in your local environment, you need to complete the following steps:\n"
        "- Pay the following lightning invoice using any lightning wallet like Alby or Phoenix:\n\n"
        f"{invoice}\n\n"
        "- After payment, enter the preimage below.\n"
    )
    note = "Note: This process uses the L402 protocol, a payment protocol for the Lightning Network. Bitcoin, used in this protocol, is a secure and efficient payment mechanism, enabling instant transactions with low fees.\n"

    print(f"{error_message}\n{instructions}\n{note}")


def split_L402(L402: str) -> tuple[str, str]:
    """Splits L402 token into token and invoice.

    Args:
      L402 (str): L402 token. Formatted as "L402 token=token, invoice=invoice"

    Returns:
      tuple[str, str]: Tuple containing the token and invoice.
    """
    token = L402.split(",")[0].split("=")[1].replace('"', "")
    invoice = L402.split(",")[1].split("=")[1].replace('"', "")
    return token, invoice
