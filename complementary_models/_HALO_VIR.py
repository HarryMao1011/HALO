import logging
from typing import List, Optional

from anndata import AnnData
from scipy.sparse import csr_matrix

from .REGISTRY_KEYS import REGISTRY_KEYS
from scvi._compat import Literal
# from scvi._types import LatentDataType
from scvi.data import AnnDataManager
from scvi.data.fields import (
    CategoricalJointObsField,
    CategoricalObsField,
    LayerField,
    NumericalJointObsField,
    NumericalObsField,
    ObsmField,
)
from scvi.model._utils import _init_library_size
from scvi.model.base import UnsupervisedTrainingMixin
from scvi.module import VAE
from scvi.utils import setup_anndata_dsp
from scvi.model import SCVI
from scvi.model.base import ArchesMixin, RNASeqMixin, VAEMixin, BaseModelClass
import torch
logger = logging.getLogger(__name__)
import numpy as np
from ._HALO_VAER import HALOVAER
from typing import Dict, Iterable, List, Optional, Sequence, Union

logger = logging.getLogger(__name__)

class HALOVIR(RNASeqMixin, VAEMixin, ArchesMixin, UnsupervisedTrainingMixin, BaseModelClass):
    def __init__(
        self,
        adata: AnnData,
        n_genes :int,
        n_regions: int,
        n_hidden: int = 128,
        n_latent: int = 10,
        n_layers: int = 1,
        dropout_rate: float = 0.1,
        dispersion: Literal["gene", "gene-batch", "gene-label", "gene-cell"] = "gene",
        gene_likelihood: Literal["zinb", "nb", "poisson"] = "zinb",
        latent_distribution: Literal["normal", "ln"] = "normal",
        **model_kwargs,
    ):
        super(HALOVIR, self).__init__(adata)
        n_cats_per_cov = (
            self.adata_manager.get_state_registry(
                REGISTRY_KEYS.CAT_COVS_KEY
            ).n_cats_per_key
            if REGISTRY_KEYS.CAT_COVS_KEY in self.adata_manager.data_registry
            else None
        )
        n_batch = self.summary_stats.n_batch
        use_size_factor_key = (
            REGISTRY_KEYS.SIZE_FACTOR_KEY in self.adata_manager.data_registry
        )
        library_log_means, library_log_vars = None, None
        if not use_size_factor_key:
            library_log_means, library_log_vars = _init_library_size(
                self.adata_manager, n_batch
            )
        print("n_genes :{}".format(n_genes))
        # n_input = self.summary_stats.n_vars
        n_labels = self.summary_stats.n_labels
        # print("n_input {}, n_labels {}".format(n_input, n_labels))
        self.module = HALOVAER(
            n_input_genes=n_genes,
            n_input_regions = n_regions,
            n_batch=n_batch,
            n_labels=n_labels,
            n_continuous_cov=self.summary_stats.get("n_extra_continuous_covs", 0),
            n_cats_per_cov=n_cats_per_cov,
            n_hidden=n_hidden,
            n_latent=n_latent,
            n_layers=n_layers,
            dropout_rate=dropout_rate,
            dispersion=dispersion,
            gene_likelihood=gene_likelihood,
            latent_distribution=latent_distribution,
            use_size_factor_key=use_size_factor_key,
            library_log_means=library_log_means,
            library_log_vars=library_log_vars,
            **model_kwargs,
        )
        self._model_summary_string = (
            "SCVI Model with the following params: \nn_hidden: {}, n_latent: {}, n_layers: {}, dropout_rate: "
            "{}, dispersion: {}, gene_likelihood: {}, latent_distribution: {}"
        ).format(
            n_hidden,
            n_latent,
            n_layers,
            dropout_rate,
            dispersion,
            gene_likelihood,
            latent_distribution,
        )
        self.init_params_ = self._get_init_params(locals())
    
    # @classmethod
    # @setup_anndata_dsp.dedent
    # def setup_anndata(
    #     cls,
    #     adata: AnnData,
    #     layer: Optional[str] = None,
    #     batch_key: Optional[str] = None,
    #     labels_key: Optional[str] = None,
    #     size_factor_key: Optional[str] = None,
    #     categorical_covariate_keys: Optional[List[str]] = None,
    #     continuous_covariate_keys: Optional[List[str]] = None,
    #     **kwargs,
    # ):
    #     """
    #     %(summary)s.

    #     Parameters
    #     ----------
    #     %(param_layer)s
    #     %(param_batch_key)s
    #     %(param_labels_key)s
    #     %(param_size_factor_key)s
    #     %(param_cat_cov_keys)s
    #     %(param_cont_cov_keys)s
    #     """
    #     setup_method_args = cls._get_setup_method_args(**locals())
    #     anndata_fields = [
    #         LayerField(REGISTRY_KEYS.X_KEY, layer, is_count_data=True),
    #         CategoricalObsField(REGISTRY_KEYS.BATCH_KEY, batch_key),
    #         CategoricalObsField(REGISTRY_KEYS.LABELS_KEY, labels_key),
    #         NumericalObsField(
    #             REGISTRY_KEYS.SIZE_FACTOR_KEY, size_factor_key, required=False
    #         ),
    #         CategoricalJointObsField(
    #             REGISTRY_KEYS.CAT_COVS_KEY, categorical_covariate_keys
    #         ),
    #         NumericalJointObsField(
    #             REGISTRY_KEYS.CONT_COVS_KEY, continuous_covariate_keys
    #         ),
    #     ]
    #     adata_manager = AnnDataManager(
    #         fields=anndata_fields, setup_method_args=setup_method_args
    #     )
    #     adata_manager.register_fields(adata, **kwargs)
    #     cls.register_manager(adata_manager)


    @classmethod
    @setup_anndata_dsp.dedent
    def setup_anndata(
        cls,
        adata: AnnData,
        layer: Optional[str] = None,
        batch_key: Optional[str] = None,
        time_key: Optional[str] = None,
        cell_key:  Optional[str] = None,
        size_factor_key: Optional[str] = None,
        labels_key: Optional[str] = None,
        categorical_covariate_keys: Optional[List[str]] = None,
        continuous_covariate_keys: Optional[List[str]] = None,
        **kwargs,
    ):
        """
        %(summary)s.
        Parameters
        ----------
        %(param_layer)s
        %(param_batch_key)s
        %(param_size_factor_key)s
        %(param_cat_cov_keys)s
        %(param_cont_cov_keys)s
        """
        setup_method_args = cls._get_setup_method_args(**locals())
        anndata_fields = [
            LayerField(REGISTRY_KEYS.X_KEY, layer, is_count_data=True),
            CategoricalObsField(REGISTRY_KEYS.BATCH_KEY, batch_key),
            CategoricalObsField(REGISTRY_KEYS.TIME_KEY, time_key),
            CategoricalObsField(REGISTRY_KEYS.LABELS_KEY, labels_key),

            ## omit the cell types for now
            NumericalObsField(
                REGISTRY_KEYS.SIZE_FACTOR_KEY, size_factor_key, required=False
            ),
            CategoricalJointObsField(
                REGISTRY_KEYS.CAT_COVS_KEY, categorical_covariate_keys
            ),
            NumericalJointObsField(
                REGISTRY_KEYS.CONT_COVS_KEY, continuous_covariate_keys
            ),
        ]
        adata_manager = AnnDataManager(
            fields=anndata_fields, setup_method_args=setup_method_args
        )
        adata_manager.register_fields(adata, **kwargs)
        cls.register_manager(adata_manager)  

    @torch.no_grad()
    def get_latent_representation(
        self,
        adata: Optional[AnnData] = None,
        modality: Literal["joint", "expression", "accessibility"] = "joint",
        indices: Optional[Sequence[int]] = None,
        give_mean: bool = True,
        batch_size: Optional[int] = None,
    ) -> np.ndarray:
        """
        Return the latent representation for each cell.
        Parameters
        ----------
        adata
            AnnData object with equivalent structure to initial AnnData. If `None`, defaults to the
            AnnData object used to initialize the model.
        modality
            Return modality specific or joint latent representation.
        indices
            Indices of cells in adata to use. If `None`, all cells are used.
        give_mean
            Give mean of distribution or sample from it.
        batch_size
            Minibatch size for data loading into model. Defaults to `scvi.settings.batch_size`.
        Returns
        -------
        latent_representation : np.ndarray
            Low-dimensional representation for each cell
        """
        # if not self.is_trained_:
        #     raise RuntimeError("Please train the model first.")

        # keys = {
        # ## integrated RNA
        # "qz_expr_m": "qz_expr_m", "qz_expr_v": "qz_expr_v" , "z_expr":"z_expr",
        # ## integrated ATAC
        # "qz_acc_m": "qz_acc_m", "qz_acc_v": "qz_acc_v",  "z_acc": "z_acc",
        # ## dependent RNA components 
        # "z_expr_dep":"z_expr_dep", "qzm_expr_dep":"qzm_expr_dep", "qzv_expr_dep":"qzv_expr_dep",
        # ## dependent ATAC components
        # "z_acc_dep":"z_acc_dep", "qzm_acc_dep":"qzm_acc_dep", "qzv_acc_dep":"qzv_acc_dep",
        # ## independent/lagging RNA components
        # "z_expr_indep": "z_expr_indep","qzm_expr_indep": "qzm_expr_indep", "qzv_expr_indep":"qzv_expr_indep",
        # ## independent/lagging ATAC components
        # "z_acc_indep": "z_acc_indep", "qzm_acc_indep":"qzm_acc_indep", "qzv_acc_indep":"qzv_acc_indep"
        # }
        

        adata = self._validate_anndata(adata)
        scdl = self._make_data_loader(
            adata=adata, indices=indices, batch_size=batch_size
        )
        latent_expr_dep = []
        latent_atac_dep = []
        latent_expr_indep = []
        latent_atac_indep = []
        latent_expr = []
        latent_atac = []
        times = []
        for tensors in scdl:
            inference_inputs = self.module._get_inference_input(tensors)
            outputs = self.module.inference(**inference_inputs)
      
            z_expr = outputs["z"]
  
            z_acc = outputs["z_acc"]

            z_expr_indep=outputs["z_expr_indep"]
            z_expr_dep = outputs["z_expr_dep"]


            ## independent ATAC
            z_acc_indep=outputs["z_acc_indep"]
            z_acc_dep=outputs["z_acc_dep"]


            time_keys = outputs['time_key']

            # latent += [z.cpu()]
            latent_atac+= [z_acc.cpu()]
            latent_expr += [z_expr.cpu()]

            latent_atac_dep+= [z_acc_dep.cpu()]
            latent_expr_dep += [z_expr_dep.cpu()]          

            latent_atac_indep+= [z_acc_indep.cpu()]
            latent_expr_indep += [z_expr_indep.cpu()]
            times += [time_keys.cpu()]



        return  torch.cat(latent_expr).numpy(), torch.cat(latent_atac).numpy(), \
            torch.cat(latent_expr_dep).numpy(), torch.cat(latent_atac_dep).numpy(), \
                torch.cat(latent_expr_indep).numpy(), torch.cat(latent_atac_indep).numpy(), torch.cat(times).numpy()
      
