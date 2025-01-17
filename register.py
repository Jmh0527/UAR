from typing import Callable, List
from collections import UserDict


class BaseRegistry:
    """
    A generic registry for dynamically managing and accessing objects.
    Subclass this class to create specific registries.
    """
    _registry = {}  

    @classmethod
    def register(cls, name: str) -> Callable:
        """
        Decorator for registering an object under a specific name.

        Args:
            name (str): The name to register the object under.
        """
        def decorator(obj):
            if name in cls._registry:
                raise KeyError(f"'{name}' is already registered.")
            cls._registry[name] = obj
            return obj
        return decorator

    @classmethod
    def __class_getitem__(cls, key: str) -> Callable:
        """
        Access a registered object by name.

        Args:
            key (str): The name of the registered object.
        
        Returns:
            Callable: The registered object.
        """
        if key not in cls._registry:
            raise KeyError(f"'{key}' is not registered.")
        return cls._registry[key]

    @classmethod
    def list_registered(cls) -> List[str]:
        """
        List all registered object names.

        Returns:
            List[str]: A list of registered object names.
        """
        return list(cls._registry.keys())


class NetworkRegistry(BaseRegistry):
    """
    A registry for managing and accessing network models.
    """
    _registry = {}


class TransformRegistry(BaseRegistry):
    """
    A registry for managing and accessing data transformations.
    """
    _registry = {}