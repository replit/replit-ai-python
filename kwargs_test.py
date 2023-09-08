from typing import Dict, Any
def foo(a, b,  parameters: Dict[str, Any]):
  print("a: ", a)
  print("b: ", b)
  print("parameters: ", parameters)

foo(a=1, b=2, parameters={"c": 3, "d": 4})


def foo_kwargs(a, b, **parameters):
  print("a: ", a)
  print("b: ", b)
  print("parameters: ", parameters)
  # print("**parameters: ", f"**parameters")

foo_kwargs(a=1, b=2, c=3, d=4)