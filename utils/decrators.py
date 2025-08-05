from collections.abc import Callable
from typing import Any, Protocol, TypeVar, runtime_checkable

F = TypeVar('F', bound=Callable[..., Any])


@runtime_checkable
class WithStaticVar(Protocol):
	"""
	A protocol that allows a class to have static variables.
	"""

	def __call__(self, *args: Any, **kwds: Any) -> Any: ...

	value: Any


def static_var(**kwargs: Any) -> Callable[[F], F]:
	def decorate(func: F) -> F:
		for key, value in kwargs.items():
			setattr(func, key, value)
		return func

	return decorate


def singleton(cls) -> Callable[..., Any]:
	"""
	A decorator to make a class a singleton.
	"""
	instances = {}

	def get_instance(*args, **kwargs):
		if cls not in instances:
			instances[cls] = cls(*args, **kwargs)
		return instances[cls]

	return get_instance
