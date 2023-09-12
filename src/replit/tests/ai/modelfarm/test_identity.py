"""Tests for replit.identity."""

import json
import os
import unittest
from unittest.mock import patch
from unittest.mock import Mock
import pyseto
import pytest
from replit.ai.modelfarm.identity import verify
from replit.ai.modelfarm.identity.sign import SigningAuthority
from replit.ai.modelfarm.replit_identity_token_manager import ReplitIdentityTokenManager

from replit.ai.modelfarm import config

PUBLIC_KEY = "on0FkSmEC+ce40V9Vc4QABXSx6TXo+lhp99b6Ka0gro="

# This token should be valid for 100y.
# Generated with:
# ```
# go run ./cmd/goval_keypairgen/ -eternal -sample-token \
#   -identity -gen-prefix dev -gen-id identity -issuer conman \
#   -replid=test -shortlived=false
# ```
# in goval.
IDENTITY_PRIVATE_KEY = "k2.secret.6sHU27WoRIaspIOVaShpuZM33ozpfFyI2THfO8fmSX6xiA_Duh4ac5g76Y5bParclsalaOCTaCs6gZowhYivVQ"  # noqa: E501,B950,S106 # line too long
IDENTITY_TOKEN = "v2.public.Q2dSMFpYTjBJZ1IwWlhOMHDnO17Eg43zucAMSAHnCS4C1wn4QUCCOcr-Pggw5SV1KnbOXq8RcQE5if6pMcbJ6lmRWcdoHq5CV9jqyRrUlwo.R0FFaUJtTnZibTFoYmhLckFuWXlMbkIxWW14cFl5NVJNbVF6VTFkd2VHRnRXbmRrTVd4U1lXMDViRm96YkVKU1ZrNUZVVmRzYTA1VmQzSlRSVlp2VWtaa2RGbFZVa3BSVmtwMlVUQmtRbFpYUmtOYU1qbEdXa1ZrVjJWdFVrUlRWRVpvWld0c01Wa3dhRmRoVjBwSVlrZHdUV0pyTldGWGFrWkRUVEEwZVU5WGVGTk5hbFpSVmpGVk5HUkhTbFpQVm1oc1lXdHdORlJVUW5kaFZrbDZVV3hvYUdKWFVubFVWekZyWlZaUmVVOVZhRnBXVkVaTFZtcENjMlZWTVZkV1ZEQkRUbVpoYkhkMk5EUm9SRkZQTFVKWlJDMURWSEUxYVdJeFQzVlVlamxIWW5WTlFVVnFURFExUVVwclZXNW9kR2hxVFZOVVRtOVZSRVphWDBsaVUyTjFjekoxWW05aVowNU1MV2RRVlRGRmVVOTFUVzlHTGxJd1JrWmhWVXAwVkc1YWFXSlVSbTlaYldSMlVteHdTRlpxU2xCaGExVTU"  # noqa: E501,B950,S106 # line too long


def setup_pub_key():
    return json.dumps({"dev:1": PUBLIC_KEY})


def test_read_public_key_from_env() -> None:
    """Test read_public_key_from_env."""
    with patch.dict(os.environ, {"REPL_PUBKEYS": setup_pub_key()}):
        pubkey = verify.read_public_key_from_env("dev:1", "goval")
        assert isinstance(pubkey, pyseto.versions.v2.V2Public)


def test_signing_authority() -> None:
    """Test SigningAuthority."""
    with patch.dict(os.environ, {"REPL_PUBKEYS": setup_pub_key()}):
        gsa = SigningAuthority(
            marshaled_private_key=IDENTITY_PRIVATE_KEY,
            marshaled_identity=IDENTITY_TOKEN,
            replid="test",
        )
        signed_token = gsa.sign("audience")

        verify.verify_identity_token(
            identity_token=signed_token,
            audience="audience",
        )


def test_verify_identity_token() -> None:
    """Test verify_identity_token."""
    with patch.dict(os.environ, {"REPL_PUBKEYS": setup_pub_key()}):
        verify.verify_identity_token(
            identity_token=IDENTITY_TOKEN,
            audience="test",
        )


def test_get_interactive_token() -> None:
    with patch.dict(
        os.environ,
        {
            "REPL_PUBKEYS": setup_pub_key(),
            "REPL_IDENTITY": IDENTITY_TOKEN,
            "REPL_IDENTITY_KEY": IDENTITY_PRIVATE_KEY,
            "REPL_ID": "test",
        },
    ):
        replit_identity_token_manager = ReplitIdentityTokenManager()
        signed_token = replit_identity_token_manager.get_interactive_token()

        verify.verify_identity_token(
            identity_token=signed_token,
            audience="modelfarm@replit.com",
        )
