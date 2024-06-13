from replit.ai.modelfarm import Modelfarm


def test_loadmodel_complete_endpoint():
  client = Modelfarm()
  response = client.completions.create(
      model="loadtesting",
      prompt=["1 + 1 = "],
  )

  assert len(response.choices) == 1

  choice = response.choices[0]

  assert "Content!" in choice.text
