from replit.ai.modelfarm.config import initialize, get_config, Config


def test_config_initialization():

    old_config = get_config()
    old_rootUrl = old_config.rootUrl
    old_audience = old_config.audience
    assert old_rootUrl is not None
    assert old_audience is not None

    initialize("https://new-url.com", "new_audience")

    new_config = get_config()

    assert new_config.rootUrl == "https://new-url.com"
    assert new_config.audience == "new_audience"

    # Reset config back to original
    initialize(old_rootUrl, old_audience)
    new_config2 = get_config()
    assert new_config2.rootUrl == old_rootUrl
    assert new_config2.audience == old_audience
