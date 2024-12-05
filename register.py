from typing import Callable, List


class BaseRegistry:
    """
    A generic registry for dynamically managing and accessing objects.
    Subclass this class to create specific registries.
    """
    _registry = {}  # 将注册表改为类属性

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
    def __setitem__(cls, key: str, value: Callable):
        """
        Add an object to the registry manually.
        
        Args:
            key (str): The name to register the object under.
            value (Callable): The object to register.
        """
        if key in cls._registry:
            raise KeyError(f"'{key}' already exists.")
        cls._registry[key] = value

    @classmethod
    def __getitem__(cls, key: str) -> Callable:
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
