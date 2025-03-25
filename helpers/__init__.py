"""HALO"""

# Set default logging handler to avoid logging with logging.lastResort logger.
import logging

from ._constants import REGISTRY_KEYS
from ._settings import settings

# this import needs to come after prior imports to prevent circular import
# from . import data, model, external, utils


# try:
#     import importlib.metadata as importlib_metadata
# except ModuleNotFoundError:
#     import importlib_metadata
# package_name = "halo"
# __version__ = importlib_metadata.version(package_name)

# settings.verbosity = logging.INFO
# test_var = "test"

# Jax sets the root logger, this prevents double output.
halo_logger = logging.getLogger("halo")
halo_logger.propagate = False

__all__ = ["settings", "REGISTRY_KEYS", "data", "model", "external", "utils"]
