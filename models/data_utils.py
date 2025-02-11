import logging
import warnings
from typing import Optional, Union
from uuid import uuid4

import h5py
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import scipy.sparse as sp_sparse
from anndata import AnnData
from anndata._core.sparse_dataset import SparseDataset
from mudata import MuData
from pandas.api.types import CategoricalDtype

from scvi._types import AnnOrMuData
_ADATA_LATENT_UNS_KEY = "_scvi_adata_latent"


def _get_latent_adata_type(adata: AnnData) -> Optional[str]:
    return adata.uns.get(_ADATA_LATENT_UNS_KEY, None)

