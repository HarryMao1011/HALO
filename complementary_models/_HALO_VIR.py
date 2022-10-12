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
import pandas as pd
from scvi._types import Number

from scvi.model._utils import (
    _get_batch_code_from_category,
    scatac_raw_counts_properties,
    scrna_raw_counts_properties,
)

from scvi.utils._docstrings import doc_differential_expression, setup_anndata_dsp
from scipy.sparse import csr_matrix, vstack
from scvi._utils import _doc_params

from functools import partial
from scvi.model.base._utils import _de_core


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
            NumericalObsField(REGISTRY_KEYS.TIME_KEY, time_key),
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
    
    @torch.no_grad()
    def get_accessibility_estimates(
        self,
        adata: Optional[AnnData] = None,
        indices: Sequence[int] = None,
        n_samples_overall: Optional[int] = None,
        region_list: Optional[Sequence[str]] = None,
        transform_batch: Optional[Union[str, int]] = None,
        use_z_mean: bool = True,
        threshold: Optional[float] = None,
        normalize_cells: bool = False,
        normalize_regions: bool = False,
        batch_size: int = 128,
        return_numpy: bool = False,
    ) -> Union[np.ndarray, csr_matrix, pd.DataFrame]:
        """
        Impute the full accessibility matrix.
        Returns a matrix of accessibility probabilities for each cell and genomic region in the input
        (for return matrix A, A[i,j] is the probability that region j is accessible in cell i).
        Parameters
        ----------
        adata
            AnnData object that has been registered with scvi. If `None`, defaults to the
            AnnData object used to initialize the model.
        indices
            Indices of cells in adata to use. If `None`, all cells are used.
        n_samples_overall
            Number of samples to return in total
        region_indices
            Indices of regions to use. if `None`, all regions are used.
        transform_batch
            Batch to condition on.
            If transform_batch is:
            - None, then real observed batch is used
            - int, then batch transform_batch is used
        use_z_mean
            If True (default), use the distribution mean. Otherwise, sample from the distribution.
        threshold
            If provided, values below the threshold are replaced with 0 and a sparse matrix
            is returned instead. This is recommended for very large matrices. Must be between 0 and 1.
        normalize_cells
            Whether to reintroduce library size factors to scale the normalized probabilities.
            This makes the estimates closer to the input, but removes the library size correction.
            False by default.
        normalize_regions
            Whether to reintroduce region factors to scale the normalized probabilities. This makes
            the estimates closer to the input, but removes the region-level bias correction. False by
            default.
        batch_size
            Minibatch size for data loading into model
        """
        adata = self._validate_anndata(adata)
        adata_manager = self.get_anndata_manager(adata, required=True)
        if indices is None:
            indices = np.arange(adata.n_obs)
        if n_samples_overall is not None:
            indices = np.random.choice(indices, n_samples_overall)
        post = self._make_data_loader(
            adata=adata, indices=indices, batch_size=batch_size
        )
        transform_batch = _get_batch_code_from_category(adata_manager, transform_batch)

        if region_list is None:
            region_mask = slice(None)
        else:
            region_mask = [
                region in region_list for region in adata.var_names[self.n_genes :]
            ]

        if threshold is not None and (threshold < 0 or threshold > 1):
            raise ValueError("the provided threshold must be between 0 and 1")

        imputed = []
        for tensors in post:
            get_generative_input_kwargs = dict(transform_batch=transform_batch[0])
            generative_kwargs = dict(use_z_mean=use_z_mean)
            inference_outputs, generative_outputs = self.module.forward(
                tensors=tensors,
                get_generative_input_kwargs=get_generative_input_kwargs,
                generative_kwargs=generative_kwargs,
                compute_loss=False,
            )
            p = generative_outputs["p"].cpu()

            if normalize_cells:
                p *= inference_outputs["libsize_acc"].cpu()
            if normalize_regions:
                p *= torch.sigmoid(self.module.region_factors).cpu()
            if threshold:
                p[p < threshold] = 0
                p = csr_matrix(p.numpy())
            if region_mask is not None:
                p = p[:, region_mask]
            imputed.append(p)

        if threshold:  # imputed is a list of csr_matrix objects
            imputed = vstack(imputed, format="csr")
        else:  # imputed is a list of tensors
            imputed = torch.cat(imputed).numpy()

        if return_numpy:
            return imputed
        elif threshold:
            return pd.DataFrame.sparse.from_spmatrix(
                imputed,
                index=adata.obs_names[indices],
                columns=adata.var_names[self.n_genes :][region_mask],
            )
        else:
            return pd.DataFrame(
                imputed,
                index=adata.obs_names[indices],
                columns=adata.var_names[self.n_genes :][region_mask],
            )

    @torch.no_grad()
    def get_normalized_expression(
        self,
        adata: Optional[AnnData] = None,
        indices: Optional[Sequence[int]] = None,
        n_samples_overall: Optional[int] = None,
        transform_batch: Optional[Sequence[Union[Number, str]]] = None,
        gene_list: Optional[Sequence[str]] = None,
        use_z_mean: bool = True,
        n_samples: int = 1,
        batch_size: Optional[int] = None,
        return_mean: bool = True,
        return_numpy: bool = False,
    ) -> Union[np.ndarray, pd.DataFrame]:
        r"""
        Returns the normalized (decoded) gene expression.
        This is denoted as :math:`\rho_n` in the scVI paper.
        Parameters
        ----------
        adata
            AnnData object with equivalent structure to initial AnnData. If `None`, defaults to the
            AnnData object used to initialize the model.
        indices
            Indices of cells in adata to use. If `None`, all cells are used.
        transform_batch
            Batch to condition on.
            If transform_batch is:
            - None, then real observed batch is used.
            - int, then batch transform_batch is used.
        gene_list
            Return frequencies of expression for a subset of genes.
            This can save memory when working with large datasets and few genes are
            of interest.
        library_size
            Scale the expression frequencies to a common library size.
            This allows gene expression levels to be interpreted on a common scale of relevant
            magnitude. If set to `"latent"`, use the latent libary size.
        use_z_mean
            If True, use the mean of the latent distribution, otherwise sample from it
        n_samples
            Number of posterior samples to use for estimation.
        batch_size
            Minibatch size for data loading into model. Defaults to `scvi.settings.batch_size`.
        return_mean
            Whether to return the mean of the samples.
        Returns
        -------
        If `n_samples` > 1 and `return_mean` is False, then the shape is `(samples, cells, genes)`.
        Otherwise, shape is `(cells, genes)`. In this case, return type is :class:`~pandas.DataFrame` unless `return_numpy` is True.
        """
        adata = self._validate_anndata(adata)
        adata_manager = self.get_anndata_manager(adata, required=True)
        if indices is None:
            indices = np.arange(adata.n_obs)
        if n_samples_overall is not None:
            indices = np.random.choice(indices, n_samples_overall)
        scdl = self._make_data_loader(
            adata=adata, indices=indices, batch_size=batch_size
        )

        transform_batch = _get_batch_code_from_category(adata_manager, transform_batch)

        if gene_list is None:
            gene_mask = slice(None)
        else:
            all_genes = adata.var_names[: self.n_genes]
            gene_mask = [gene in gene_list for gene in all_genes]

        exprs = []
        for tensors in scdl:
            per_batch_exprs = []
            for batch in transform_batch:
                if batch is not None:
                    batch_indices = tensors[REGISTRY_KEYS.BATCH_KEY]
                    tensors[REGISTRY_KEYS.BATCH_KEY] = (
                        torch.ones_like(batch_indices) * batch
                    )
                _, generative_outputs = self.module.forward(
                    tensors=tensors,
                    inference_kwargs=dict(n_samples=n_samples),
                    generative_kwargs=dict(use_z_mean=use_z_mean),
                    compute_loss=False,
                )
                output = generative_outputs["px_scale"]
                output = output[..., gene_mask]
                output = output.cpu().numpy()
                per_batch_exprs.append(output)
            per_batch_exprs = np.stack(
                per_batch_exprs
            )  # shape is (len(transform_batch) x batch_size x n_var)
            exprs += [per_batch_exprs.mean(0)]

        if n_samples > 1:
            # The -2 axis correspond to cells.
            exprs = np.concatenate(exprs, axis=-2)
        else:
            exprs = np.concatenate(exprs, axis=0)
        if n_samples > 1 and return_mean:
            exprs = exprs.mean(0)

        if return_numpy:
            return exprs
        else:
            return pd.DataFrame(
                exprs,
                columns=adata.var_names[: self.n_genes][gene_mask],
                index=adata.obs_names[indices],
            )

    @_doc_params(doc_differential_expression=doc_differential_expression)
    def differential_accessibility(
        self,
        adata: Optional[AnnData] = None,
        groupby: Optional[str] = None,
        group1: Optional[Iterable[str]] = None,
        group2: Optional[str] = None,
        idx1: Optional[Union[Sequence[int], Sequence[bool]]] = None,
        idx2: Optional[Union[Sequence[int], Sequence[bool]]] = None,
        mode: Literal["vanilla", "change"] = "change",
        delta: float = 0.05,
        batch_size: Optional[int] = None,
        all_stats: bool = True,
        batch_correction: bool = False,
        batchid1: Optional[Iterable[str]] = None,
        batchid2: Optional[Iterable[str]] = None,
        fdr_target: float = 0.05,
        silent: bool = False,
        two_sided: bool = True,
        **kwargs,
    ) -> pd.DataFrame:
        r"""
        A unified method for differential accessibility analysis.
        Implements `"vanilla"` DE [Lopez18]_ and `"change"` mode DE [Boyeau19]_.
        Parameters
        ----------
        {doc_differential_expression}
        two_sided
            Whether to perform a two-sided test, or a one-sided test.
        **kwargs
            Keyword args for :meth:`scvi.model.base.DifferentialComputation.get_bayes_factors`
        Returns
        -------
        Differential accessibility DataFrame with the following columns:
        prob_da
            the probability of the region being differentially accessible
        is_da_fdr
            whether the region passes a multiple hypothesis correction procedure with the target_fdr
            threshold
        bayes_factor
            Bayes Factor indicating the level of significance of the analysis
        effect_size
            the effect size, computed as (accessibility in population 2) - (accessibility in population 1)
        emp_effect
            the empirical effect, based on observed detection rates instead of the estimated accessibility
            scores from the PeakVI model
        est_prob1
            the estimated probability of accessibility in population 1
        est_prob2
            the estimated probability of accessibility in population 2
        emp_prob1
            the empirical (observed) probability of accessibility in population 1
        emp_prob2
            the empirical (observed) probability of accessibility in population 2
        """
        adata = self._validate_anndata(adata)
        col_names = adata.var_names[self.n_genes :]
        model_fn = partial(
            self.get_accessibility_estimates, use_z_mean=False, batch_size=batch_size
        )

        # TODO check if change_fn in kwargs and raise error if so
        def change_fn(a, b):
            return a - b

        if two_sided:

            def m1_domain_fn(samples):
                return np.abs(samples) >= delta

        else:

            def m1_domain_fn(samples):
                return samples >= delta

        all_stats_fn = partial(
            scatac_raw_counts_properties,
            var_idx=np.arange(adata.shape[1])[self.n_genes :],
        )

        result = _de_core(
            adata_manager=self.get_anndata_manager(adata, required=True),
            model_fn=model_fn,
            groupby=groupby,
            group1=group1,
            group2=group2,
            idx1=idx1,
            idx2=idx2,
            all_stats=all_stats,
            all_stats_fn=all_stats_fn,
            col_names=col_names,
            mode=mode,
            batchid1=batchid1,
            batchid2=batchid2,
            delta=delta,
            batch_correction=batch_correction,
            fdr=fdr_target,
            change_fn=change_fn,
            m1_domain_fn=m1_domain_fn,
            silent=silent,
            **kwargs,
        )

        # manually change the results DataFrame to fit a PeakVI differential accessibility results
        result = pd.DataFrame(
            {
                "prob_da": result.proba_de,
                "is_da_fdr": result.loc[:, f"is_de_fdr_{fdr_target}"],
                "bayes_factor": result.bayes_factor,
                "effect_size": result.scale2 - result.scale1,
                "emp_effect": result.emp_mean2 - result.emp_mean1,
                "est_prob1": result.scale1,
                "est_prob2": result.scale2,
                "emp_prob1": result.emp_mean1,
                "emp_prob2": result.emp_mean2,
            },
            index=col_names,
        )
        return result

    @_doc_params(doc_differential_expression=doc_differential_expression)
    def differential_expression(
        self,
        adata: Optional[AnnData] = None,
        groupby: Optional[str] = None,
        group1: Optional[Iterable[str]] = None,
        group2: Optional[str] = None,
        idx1: Optional[Union[Sequence[int], Sequence[bool]]] = None,
        idx2: Optional[Union[Sequence[int], Sequence[bool]]] = None,
        mode: Literal["vanilla", "change"] = "change",
        delta: float = 0.25,
        batch_size: Optional[int] = None,
        all_stats: bool = True,
        batch_correction: bool = False,
        batchid1: Optional[Iterable[str]] = None,
        batchid2: Optional[Iterable[str]] = None,
        fdr_target: float = 0.05,
        silent: bool = False,
        **kwargs,
    ) -> pd.DataFrame:
        r"""
        A unified method for differential expression analysis.
        Implements `"vanilla"` DE [Lopez18]_ and `"change"` mode DE [Boyeau19]_.
        Parameters
        ----------
        {doc_differential_expression}
        **kwargs
            Keyword args for :meth:`scvi.model.base.DifferentialComputation.get_bayes_factors`
        Returns
        -------
        Differential expression DataFrame.
        """
        adata = self._validate_anndata(adata)

        col_names = adata.var_names[: self.n_genes]
        model_fn = partial(
            self.get_normalized_expression,
            batch_size=batch_size,
        )
        all_stats_fn = partial(
            scrna_raw_counts_properties,
            var_idx=np.arange(adata.shape[1])[: self.n_genes],
        )
        result = _de_core(
            adata_manager=self.get_anndata_manager(adata, required=True),
            model_fn=model_fn,
            groupby=groupby,
            group1=group1,
            group2=group2,
            idx1=idx1,
            idx2=idx2,
            all_stats=all_stats,
            all_stats_fn=all_stats_fn,
            col_names=col_names,
            mode=mode,
            batchid1=batchid1,
            batchid2=batchid2,
            delta=delta,
            batch_correction=batch_correction,
            fdr=fdr_target,
            silent=silent,
            **kwargs,
        )

        return result
      
