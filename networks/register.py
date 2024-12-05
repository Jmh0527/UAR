import os 
from collections import UserDict

import torch.nn as nn 


class NetworkRegistry(UserDict):
    """
    A registry to dynamically manage and load network definitions.
    """
    _registry = {}

    @classmethod
    def register(cls, name):
        """
        Decorator to register a network by name.

        Args:
            name (str): The name to register the network under.
        """
        def decorator(network_class):
            cls._registry[name] = network_class
            return network_class
        return decorator
    
    def __setitem__(self, key, value):
        """
        Manually add a network to the registry.
        """
        sefl._registry[key] = value

    def __getitem__(self, key):
        """
        Access registered networks as a dictionary.
        """
        if key not in cls._registry:
            raise KeyError(f"Network '{key}' is not registered.")
        return cls._registry[key] 
    
    @classmethod
    def list_registered(cls):
        """
        List all registered networks
        """
        return list(cls._registry.keys())