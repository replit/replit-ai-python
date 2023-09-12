from replit.ai.modelfarm import CompletionModel


def test_loadmodel_complete_endpoint():
    model = CompletionModel("loadtesting")
    response = model.complete(["1 + 1 = "])

    assert len(response.responses) == 1
    assert len(response.responses[0].choices) == 1

    choice = response.responses[0].choices[0]

    assert "Content!" in choice.content
