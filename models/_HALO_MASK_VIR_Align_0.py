import logging
from typing import List, Optional

from anndata import AnnData
from scipy.sparse import csr_matrix
import scvi
from .REGISTRY_KEYS import REGISTRY_KEYS
from scvi._compat import Literal
# from scvi.typing import Literal
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
from ._HALO_MASK_VAE_Align import HALOMASKVAE_ALN as HALOMASKVAE
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
from tqdm.auto import tqdm
from scipy.stats import fisher_exact
import tools.adata_interface.core as adi 
import tools.adata_interface.regulators as ri 
from  tools.plots.factor_influence_plot import plot_factor_influence
logger = logging.getLogger(__name__)
import scanpy as sc

class HALOMASKVIR_ALN(RNASeqMixin, VAEMixin, ArchesMixin, UnsupervisedTrainingMixin, BaseModelClass):
    def __init__(
        self,
        adata: AnnData,
        n_genes :int,
        n_regions: int,
        n_hidden: int = 128,
        n_latent: int = 10,
        n_dependent: int = 5,
        n_layers: int = 1,
        dropout_rate: float = 0.1,
        dispersion: Literal["gene", "gene-batch", "gene-label", "gene-cell"] = "gene",
        gene_likelihood: Literal["zinb", "nb", "poisson"] = "zinb",
        latent_distribution: Literal["normal", "ln"] = "normal",
        fine_tune = False,
        **model_kwargs,
    ):
        # super(HALOMASKVIR_ALN, self).__init__(adata)

        super().__init__(adata,
        n_genes,
        n_regions,
        n_hidden,
        n_latent,
        dropout_rate,
        gene_likelihood,
        latent_distribution,
        **model_kwargs,)

        n_cats_per_cov = (
            self.adata_manager.get_state_registry(
                REGISTRY_KEYS.CAT_COVS_KEY
            ).n_cats_per_key
            if REGISTRY_KEYS.CAT_COVS_KEY in self.adata_manager.data_registry
            else None
        )
        n_batch = self.summary_stats.n_batch
        n_batch=0
        use_size_factor_key = (
            REGISTRY_KEYS.SIZE_FACTOR_KEY in self.adata_manager.data_registry
        )
        library_log_means, library_log_vars = None, None
        if not use_size_factor_key:
            library_log_means, library_log_vars = _init_library_size(
                self.adata_manager, n_batch
            )
        # print("n_genes :{}".format(n_genes))
        # n_input = self.summary_stats.n_vars
        n_labels = self.summary_stats.n_labels
        self.fine_tune = fine_tune
        self.n_latent = n_latent
        self.n_genes = n_genes
        # print("fine tune is {}".format(fine_tune))

        # print("n_input {}, n_labels {}".format(n_input, n_labels))
        self.module = HALOMASKVAE(
            n_input_genes=n_genes,
            n_input_regions = n_regions,
            n_batch=n_batch,
            n_labels=n_labels,
            n_latent_dep= n_dependent,
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
            gates_finetune=self.fine_tune,
            **model_kwargs,
        )
        self._model_summary_string = (
            "SCVI Model with the following params: \nn_hidden: {}, n_latent: {}, n_layers: {}, dropout_rate: "
            "{}, dispersion: {}, gene_likelihood: {}, latent_distribution: {}, continuous var: {}, categorical vars: {}"
        ).format(
            n_hidden,
            n_latent,
            n_layers,
            dropout_rate,
            dispersion,
            gene_likelihood,
            latent_distribution,
            n_cats_per_cov,
            self.summary_stats.get("n_extra_continuous_covs", 0)
        )
        self.init_params_ = self._get_init_params(locals())

        ### store the enrichment information
        self.enrichments = dict()
        self.num_endog_features = 0
        self.num_exog_features = 0
        self.features = 0
        self.highly_variable = 0
        self.num_exo_features = 0
    
    def scheduled_train(self, epoch=400, batch_size=16):
        
        recon_epoch = epoch * 0.75
        
        causal_epoch = epoch*0.25

        for i in range(epoch):
            if i < recon_epoch:
                print("start training RNA and ATAC reconstruction ... ")
                self.module.set_train_params(expr_train=True, acc_train=True)
                self.module.set_finetune_params(0)
                self.train(max_epochs=recon_epoch, batch_size=batch_size)
            elif i >= recon_epoch and i < epoch:
                print("start training the causal constraints and reconstruction ...")
                self.module.set_train_params(expr_train=True, acc_train=True)
                self.module.set_finetune_params(2)
                self.train(max_epochs=causal_epoch, batch_size=batch_size)


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
    def get_atac_expr_denoms(
        self,
        adata: Optional[AnnData] = None,
        indices: Optional[Sequence[int]] = None,
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
        rna_denoms = []
        atac_denoms = []
       
        for tensors in scdl:
            inference_inputs = self.module._get_inference_input(tensors)
            outputs = self.module.inference(**inference_inputs)      
            z_expr = outputs["z"]
            z_acc = outputs["z_acc"]
            library_sz =  outputs[""]
            # loading = self.module.z_decoder_accessibility.get_loading_global_weights(z_acc)
            atac_denom = self.module.z_decoder_accessibility._get_softmax_denom(z_acc)
            rna_denom = self.module.decoder._get_softmax_denom(z_expr)
            atac_denoms+= [atac_denom.cpu()]
            rna_denoms += [rna_denom.cpu()]
   
        return  torch.cat(atac_denoms).numpy(), torch.cat(rna_denoms).numpy()


    @torch.no_grad()
    def get_rna_atac_denoms(
        self,
        adata: Optional[AnnData] = None,
        indices: Optional[Sequence[int]] = None,
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
        loadings = []
       
        for tensors in scdl:
            inference_inputs = self.module._get_inference_input(tensors)
            outputs = self.module.inference(**inference_inputs)      
            z_acc = outputs["z_acc"]
            # loading = self.module.z_decoder_accessibility.get_loading_global_weights(z_acc)
            loading = self.module.z_decoder_accessibility.get_loading_global_weights(z_acc)
            loading = np.expand_dims(loading, axis=0)
            loadings.append(loading)
        
        # nploadings =  np.array(loadings)
        # nploadings = nploadings.mean()
        loadings = np.concatenate(loadings, axis=0)
        loadings = np.mean(loadings, axis=0)


        return  loadings



    @torch.no_grad()
    def get_atac_loading(self):
        return self.get_atac_loading_global()

    @torch.no_grad()
    def get_atac_loading_global(
        self
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
        loadings = self.module.z_decoder_accessibility.get_loading_global_weights()
        return  loadings

    ## get the enriched TFs
    @torch.no_grad()
    @adi.wraps_modelfunc(ri.fetch_factor_hits, adi.return_output,
        ['hits_matrix','metadata']) 
    def get_enriched_TFs(self, factor_type = 'motifs', top_quantile = 0.2, *, 
            topic_num, hits_matrix, metadata, loadings, num_exo_features):
        '''
        Get TF enrichments in top peaks associated with a topic. Can be used to
        associate a topic with either motif or ChIP hits from Cistrome's 
        collection of public ChIP-seq data.
        Before running this function, one must run either:
        `mira.tl.get_motif_hits_in_peaks`
        or:
        `mira.tl.get_ChIP_hits_in_peaks`
        Parameters
        ----------
        factor_type : str, 'motifs' or 'chip', default = 'motifs'
            Which factor type to use for enrichment
        top_quantile : float > 0, default = 0.2
            Top quantile of peaks to use to represent topic in fisher exact test.
        topic_num : int > 0
            Topic for which to get enrichments
        
        Examples
        --------
        .. code-block:: python
            >>> mira.tl.get_motif_hits_in_peaks(atac_data, genome_fasta = '~/genome.fa')
            >>> atac_model.get_enriched_TFs(atac_data, topic_num = 10)
        '''

        assert(isinstance(top_quantile, float) and top_quantile > 0 and top_quantile < 1)
        hits_matrix = self._validate_hits_matrix(hits_matrix)
        print(hits_matrix.shape)
        print(metadata)
        num_peaks = loadings.shape[1]
        if num_exo_features == None:
            num_exo_features = num_peaks
        # print("num of exo features {}".format(num_exo_features))
        ## remaped exog_features
        module_idx = self._argsort_peaks(topic_num,  loadings=loadings)[-int(num_exo_features*top_quantile) : ]
        zeros_index = np.where(loadings[topic_num, :] <= 0.1)[0]
        # print("zeros index len {}".format(len(zeros_index)))
        # print("module_idx len before {}".format(len(module_idx)))

        module_idx = np.setdiff1d(module_idx, zeros_index)
        # print("module_idx len after {}".format(len(module_idx)))


        pvals, test_statistics = [], []
        for i in tqdm(range(hits_matrix.shape[0]), 'Finding enrichments'):

            tf_hits = hits_matrix[i,:].indices
            overlap = len(np.intersect1d(tf_hits, module_idx))
            module_only = len(module_idx) - overlap
            tf_only = len(tf_hits) - overlap
            ## check this part of code
            # neither = num_peaks - (overlap + module_only + tf_only)
            ## reset to exo number


            neither = num_exo_features - (overlap + module_only + tf_only)
            if neither < 0:
                neither = 0
            # print("tf_only {}, module_only {}, overlap {}, tf_hits {}".format(tf_only, module_only, overlap, tf_hits))
            # print("neither: {}".format(neither))


            contingency_matrix = np.array([[overlap, module_only], [tf_only, neither]])
            # print("contigency_matrix {}".format(contingency_matrix))
            stat,pval = fisher_exact(contingency_matrix, alternative='greater')
            pvals.append(pval)
            test_statistics.append(stat)

        results = [
            dict(**meta, pval = pval, test_statistic = test_stat)
            for meta, pval, test_stat in zip(metadata, pvals, test_statistics)
        ]
        self.enrichments[(factor_type, topic_num)] = results
        return results, module_idx

        
    def _validate_hits_matrix(self, hits_matrix):
        # assert(isspmatrix(hits_matrix))
        assert(len(hits_matrix.shape) == 2)
        hits_matrix = hits_matrix.tocsr()
        hits_matrix.data = np.ones_like(hits_matrix.data)
        return hits_matrix

    ## run this function after getting the loading matrix
    ## return from the descending order
    def _argsort_peaks(self, latent_num, loadings):
        
        return np.argsort(loadings[latent_num, :])


    def get_coupled_decoupled_genes(self, loadingmatrix):
        """
        
        """
        return NotImplemented


    def get_coupled_decoupled_peaks(self, loadingmatrix):
        """
        
        """
        return NotImplemented    



    def plot_compare_topic_enrichments(self, topic_1, topic_2, factor_type = 'motifs', 
        label_factors = None, hue = None, palette = 'coolwarm', hue_order = None, 
        ax = None, figsize = (8,8), legend_label = '', show_legend = True, fontsize = 13, 
        pval_threshold = (1e-50, 1e-50), na_color = 'lightgrey',
        color = 'grey', label_closeness = 3, max_label_repeats = 3, show_factor_ids = False):
        '''
        It is often useful to contrast topic enrichments in order to
        understand which factors' influence is unique to certain
        cell states. Topics may be enriched for constitutively-active
        transcription factors, so comparing two similar topics to find
        the factors that are unique to each elucidates the dynamic
        aspects of regulation between states.
        This function contrasts the enrichments of two topics.
        Parameters
        ----------
        topic1, topic2 : int
            Which topics to compare.
        factor_type : str, 'motifs' or 'chip', default = 'motifs'
            Which factor type to use for enrichment.
        label_factors : list[str], np.ndarray[str], None; default=None
            List of factors to label. If not provided, will label all
            factors that meet the p-value thresholds.
        hue : dict[str : {str, float}] or None
            If provided, colors the factors on the plot. The keys of the dict
            must be the names of transcription factors, and the values are
            the associated data to map to colors. The values may be 
            categorical, e.g. cluster labels, or scalar, e.g. expression
            values. TFs not provided in the dict are colored as *na_color*.
        palette : str, list[str], or None; default = None
            Palette of plot. Default of None will set `palette` to the style-specific default.
        hue_order : list[str] or None, default = None
            Order to assign hues to features provided by `data`. Works similarly to
            hue_order in seaborn. User must provide list of features corresponding to 
            the order of hue assignment. 
        ax : matplotlib.pyplot.axes, deafult = None
            Provide axes object to function to add streamplot to a subplot composition,
            et cetera. If no axes are provided, they are created internally.
        figsize : tuple(float, float), default = (8,8)
            Size of figure
        legend_label : str, None
            Label for legend.
        show_legend : boolean, default=True
            Show figure legend.
        fontsize : int>0, default=13
            Fontsize of TF labels on plot.
        pval_threshold : tuple[float, float], default=(1e-50, 1e-50)
            Threshold below with TFs will not be labeled on plot. The first and
            second positions relate p-value with respect to topic 1 and topic 2.
        na_color : str, default='lightgrey'
            Color for TFs with no provided *hue*
        color : str, default='grey'
            If *hue* not provided, colors all points on plot this color.
        label_closeness : int>0, default=3
            Closeness of TF labels to points on plot. When *label_closeness* is high,
            labels are forced to be very close to points.
        max_label_repeats : boolean, default=3
            Some TFs have multiple ChIP samples or Motif PWMs. For these factors,
            label the top *max_label_repeats* examples. This prevents clutter when
            many samples for the same TF are close together. The rank of the sample
            for each TF is shown in the label as "<TF name> (<rank>)".
        Returns
        -------
        matplotlib.pyplot.axes
        Examples
        --------
        .. code-block :: python
            >>> label = ['LEF1','HOXC13','MEOX2','DLX3','BACH2','RUNX1', 'SMAD2::SMAD3']
            >>> atac_model.plot_compare_topic_enrichments(23, 17,
            ...     label_factors = label, 
            ...     color = 'lightgrey',
            ...     fontsize=20, label_closeness=5, 
            ... )
        .. image:: /_static/mira.topics.AccessibilityModel.plot_compare_topic_enrichments.svg
            :width: 300
        '''

        m1 = self.get_enrichments(topic_1, factor_type)
        m2 = self.get_enrichments(topic_2, factor_type)        
        
        return plot_factor_influence(m1, m2, ax = ax, label_factors = label_factors,
            pval_threshold = pval_threshold, hue = hue, hue_order = hue_order, 
            palette = palette, legend_label = legend_label, show_legend = show_legend, label_closeness = label_closeness, 
            na_color = na_color, max_label_repeats = max_label_repeats, figsize=figsize,
            axlabels = ('Topic {} Enrichments'.format(str(topic_1)),'Todule {} Enrichments'.format(str(topic_2))), 
            fontsize = fontsize, color = color)    



    def get_enrichments(self, topic_num, factor_type = 'motifs'):
        '''
        Returns TF enrichments for a certain topic.
        Parameters
        ----------
        topic_num : int
            For which topic to return results
        factor_type : str, 'motifs' or 'chip', default = 'motifs'
            Which factor type to use for enrichment
        Returns
        -------
        
        topic_enrichments : list[dict]
            For each record, gives a dict of 
            {'factor_id' : <id>,
            'name' : <name>,
            'parsed_name' : <name used for expression lookup>,
            'pval' : <pval>,
            'test_statistic' : <statistic>}
        Raises
        ------
        KeyError : if *get_enriched_TFs* was not yet run for the given topic.
        '''
        try:
            return self.enrichments[(factor_type, topic_num)]
        except KeyError:
            raise KeyError('User has not gotten enrichments yet for topic {} using factor_type: {}. Run "get_enriched_TFs" function.'\
                .format(str(topic_num), str(factor_type)))


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
        batch_size: int = 32,
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
            # get_generative_input_kwargs = dict(transform_batch=transform_batch[0])
            # generative_kwargs = dict(use_z_mean=use_z_mean)
            inference_outputs, generative_outputs = self.module.forward(
                tensors=tensors,
                # get_generative_input_kwargs=get_generative_input_kwargs,
                # generative_kwargs=generative_kwargs,
                compute_loss=False,
            )
            p = generative_outputs["pa"].cpu()

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
                    # generative_kwargs=dict(use_z_mean=use_z_mean),
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


    def plot_compare_topic_enrichments(self, topic_1, topic_2, factor_type = 'motifs', 
        label_factors = None, hue = None, palette = 'coolwarm', hue_order = None, 
        ax = None, figsize = (8,8), legend_label = '', show_legend = True, fontsize = 13, 
        pval_threshold = (1e-50, 1e-50), na_color = 'lightgrey',
        color = 'grey', label_closeness = 3, max_label_repeats = 3, show_factor_ids = False):
        '''
        It is often useful to contrast topic enrichments in order to
        understand which factors' influence is unique to certain
        cell states. Topics may be enriched for constitutively-active
        transcription factors, so comparing two similar topics to find
        the factors that are unique to each elucidates the dynamic
        aspects of regulation between states.
        This function contrasts the enrichments of two topics.
        Parameters
        ----------
        topic1, topic2 : int
            Which topics to compare.
        factor_type : str, 'motifs' or 'chip', default = 'motifs'
            Which factor type to use for enrichment.
        label_factors : list[str], np.ndarray[str], None; default=None
            List of factors to label. If not provided, will label all
            factors that meet the p-value thresholds.
        hue : dict[str : {str, float}] or None
            If provided, colors the factors on the plot. The keys of the dict
            must be the names of transcription factors, and the values are
            the associated data to map to colors. The values may be 
            categorical, e.g. cluster labels, or scalar, e.g. expression
            values. TFs not provided in the dict are colored as *na_color*.
        palette : str, list[str], or None; default = None
            Palette of plot. Default of None will set `palette` to the style-specific default.
        hue_order : list[str] or None, default = None
            Order to assign hues to features provided by `data`. Works similarly to
            hue_order in seaborn. User must provide list of features corresponding to 
            the order of hue assignment. 
        ax : matplotlib.pyplot.axes, deafult = None
            Provide axes object to function to add streamplot to a subplot composition,
            et cetera. If no axes are provided, they are created internally.
        figsize : tuple(float, float), default = (8,8)
            Size of figure
        legend_label : str, None
            Label for legend.
        show_legend : boolean, default=True
            Show figure legend.
        fontsize : int>0, default=13
            Fontsize of TF labels on plot.
        pval_threshold : tuple[float, float], default=(1e-50, 1e-50)
            Threshold below with TFs will not be labeled on plot. The first and
            second positions relate p-value with respect to topic 1 and topic 2.
        na_color : str, default='lightgrey'
            Color for TFs with no provided *hue*
        color : str, default='grey'
            If *hue* not provided, colors all points on plot this color.
        label_closeness : int>0, default=3
            Closeness of TF labels to points on plot. When *label_closeness* is high,
            labels are forced to be very close to points.
        max_label_repeats : boolean, default=3
            Some TFs have multiple ChIP samples or Motif PWMs. For these factors,
            label the top *max_label_repeats* examples. This prevents clutter when
            many samples for the same TF are close together. The rank of the sample
            for each TF is shown in the label as "<TF name> (<rank>)".
        Returns
        -------
        matplotlib.pyplot.axes
        Examples
        --------
        .. code-block :: python
            >>> label = ['LEF1','HOXC13','MEOX2','DLX3','BACH2','RUNX1', 'SMAD2::SMAD3']
            >>> atac_model.plot_compare_topic_enrichments(23, 17,
            ...     label_factors = label, 
            ...     color = 'lightgrey',
            ...     fontsize=20, label_closeness=5, 
            ... )
        .. image:: /_static/mira.topics.AccessibilityModel.plot_compare_topic_enrichments.svg
            :width: 300
        '''

        m1 = self.get_enrichments(topic_1, factor_type)
        m2 = self.get_enrichments(topic_2, factor_type)        
        
        return plot_factor_influence(m1, m2, ax = ax, label_factors = label_factors,
            pval_threshold = pval_threshold, hue = hue, hue_order = hue_order, 
            palette = palette, legend_label = legend_label, show_legend = show_legend, label_closeness = label_closeness, 
            na_color = na_color, max_label_repeats = max_label_repeats, figsize=figsize,
            axlabels = ('Topic {} Enrichments'.format(str(topic_1)),'Todule {} Enrichments'.format(str(topic_2))), 
            fontsize = fontsize, color = color) 


    def get_rna_loading(self):
        return self.module.get_loadings()         

    def get_rna_decoupled_score(self, rnaloading, rnadata, couple_dim=10):
        
        genes_num = rnaloading.shape[1]
        decouple_scores = []
        couple_scores = []

        for i in range(genes_num):
            couple_latent = rnaloading[:couple_dim, i]
            decouple_latent = rnaloading[couple_dim:,i]
            decouplescore = np.abs(np.sum(decouple_latent[decouple_latent>0], axis=0))
            couplescore = np.abs(np.sum(couple_latent[couple_latent>0], axis=0))
            decouplescore_norm = decouplescore / (couplescore + decouplescore)
            couplescore_norm = couplescore / (couplescore + decouplescore)
            decouple_scores.append(decouplescore_norm)
            couple_scores.append(couplescore_norm)

        couple_scores = np.array(couple_scores)
        decouple_scores = np.array(decouple_scores)
        rnadata.var["decouple_score"] = decouple_scores
        rnadata.var["couple_score"] = couple_scores

        return couple_scores, decouple_scores

    classmethod
    def setup_dataset(multiomic="halo/E18_mouse_Brain/multiomic.h5ad", rna_ann="halo/E18_mouse_Brain/RNA/metadata.tsv"):
        adata_multi = sc.read_h5ad(multiomic)
        adata_multi.obs["batch_id"] = 1
        adata_multi.var["modality"] =adata_multi.var["feature_types"]
        adata_mvi = scvi.data.organize_multiome_anndatas(adata_multi)

        df_meta= pd.read_csv(rna_ann,sep = "\t",index_col=0)
        bins = df_meta.binned.unique()
        times = {}
        index = 0
        for bin in sorted(bins):
            times[bin] = index
            index += 1

        def add_time(row, times):
            timestamp = times[row.binned]
            return timestamp

        df_meta['time_key'] = df_meta.apply(lambda row: add_time(row, times), axis=1)

        newindex = []

        for idx, row in df_meta.iterrows():
            newindex.append(idx+"_paired")

        df_meta['Id'] = newindex    

        df_meta_sub = df_meta[["Id", 'latent_time']]

        df_meta_sub.set_index("Id", inplace=True)
        adata_mvi.obs = adata_mvi.obs.join(df_meta_sub, how="inner")
        sc.pp.filter_genes(adata_mvi, min_cells=int(adata_mvi.shape[0] * 0.01))
        return adata_mvi    

    def _argsort_genes(self, latent_num, loadings):
        
        return np.argsort(loadings[latent_num, :])

    @torch.no_grad()
    def get_top_genes(self, top_num, loadingmatrix, latent_index, rnadata, colname = "gene_short_name"):
        gene_index = self._argsort_genes(latent_index,  loadings=loadingmatrix)[-top_num : ]
        if colname != "index":
            gene_name = rnadata.var[colname][gene_index]
        else:
            gene_name = rnadata.var.index[gene_index]  
        return gene_name.tolist()


    def rank_genes(self, latent_index, loadingmatrix, rnadata, colname="gene_short_name"):
        '''
        Ranks genes according to their activation in module `latent_num`. Sorted from least to most activated.
        Parameters
        ----------
        latent_num : int
            For which latent factors to rank genes
        Returns
        -------
        np.ndarray: sorted array of gene names in order from most suppressed to most activated given the specified module
        Examples
        --------
        Genes are ranked from least to most activated. To get the top genes:
        .. code-block:: python
            >>> rna_model.rank_genes(0)[-10:]
            array(['ESRRG', 'APIP', 'RPGRIP1L', 'TM4SF4', 'DSCAM', 'NRAD1', 'ST3GAL1',
            'LEPR', 'EXOC6', 'SLC44A5'], dtype=object)
        '''
        assert(isinstance(latent_index, int) and latent_index < 2 * self.n_latent and latent_index >= 0)
        gene_index = self._argsort_genes(latent_index,  loadings=loadingmatrix)
        gene_name = rnadata.var[colname][gene_index]

        return gene_name


    def rank_modules(self, gene, rnadata, loadings, colname="gene_short_name"):
        '''
        For a gene, rank how much its expression is activated by each module
        Parameters
        ----------
        gene : str
            Name of gene

        rnadata: adata

        colname: str
            The column name of gene name in rnadata.var    
    
        Raises
        ------
        AssertionError: if **gene** is not in self.genes
        
        Returns
        -------
        list : of format [(topic_num, activation), ...]
        Examples
        --------
        To see the top 5 modules associated with gene "GHRL":
        .. code-block:: python
            >>> rna_model.rank_modules('GHRL')[:5]
            [(14, 3.375548), (22, 2.321417), (1, 2.3068447), (0, 1.780294), (9, 1.3936363)]
        '''
        genenames = rnadata.var[colname].tolist()
        gene_idx = np.argwhere(genenames == gene)[0]

        return list(sorted(zip(range(self.n_latent), np.argsort(loadings)[:, gene_idx].reshape(-1)), key = lambda x : -x[1]))


    @torch.no_grad()
    @adi.wraps_modelfunc(ri.fetch_factor_hits, adi.return_output,
        ['hits_matrix','metadata']) 
    def get_enriched_grouped_TFs(self, factor_type = 'motifs', top_quantile = 0.2, *, 
            group_index,  hits_matrix, metadata, loadings, num_exo_features, group_type="decouple"):
        '''
        Get TF enrichments in top peaks associated with a topic. Can be used to
        associate a topic with either motif or ChIP hits from Cistrome's 
        collection of public ChIP-seq data.
        Before running this function, one must run either:
        `mira.tl.get_motif_hits_in_peaks`
        or:
        `mira.tl.get_ChIP_hits_in_peaks`
        Parameters
        ----------
        factor_type : str, 'motifs' or 'chip', default = 'motifs'
            Which factor type to use for enrichment
        top_quantile : float > 0, default = 0.2
            Top quantile of peaks to use to represent topic in fisher exact test.
        group_indices : an index array of latent factors, eg: range(0,10)
            for which to get enrichments
        group_type: "coupled" or "decoupled"plot_compare_topic_enrichments    
        
        Examples
        --------
        .. code-block:: python
            >>> mira.tl.get_motif_hits_in_peaks(atac_data, genome_fasta = '~/genome.fa')
            >>> atac_model.get_enriched_TFs(atac_data, topic_num = 10)
        '''

        assert(isinstance(top_quantile, float) and top_quantile > 0 and top_quantile < 1)
        hits_matrix = self._validate_hits_matrix(hits_matrix)
        num_peaks = loadings.shape[1]
        if num_exo_features == None:
            num_exo_features = num_peaks
        # print("num of exo features {}".format(num_exo_features))
        ## remaped exog_features
        subloadings = np.sum(loadings[group_index, :], axis=0)
        # module_idx = self._argsort_peaks(topic_num,  loadings=loadings)[-int(num_exo_features*top_quantile) : ]
        module_idx = np.argsort(subloadings)[-int(num_exo_features*top_quantile) : ]
        zeros_index = np.where(subloadings <= 0.1)[0]
        # print("zeros index len {}".format(len(zeros_index)))
        # print("module_idx len before {}".format(len(module_idx)))

        module_idx = np.setdiff1d(module_idx, zeros_index)
        # print("module_idx len after {}".format(len(module_idx)))


        pvals, test_statistics = [], []
        for i in tqdm(range(hits_matrix.shape[0]), 'Finding enrichments'):

            tf_hits = hits_matrix[i,:].indices
            overlap = len(np.intersect1d(tf_hits, module_idx))
            module_only = len(module_idx) - overlap
            tf_only = len(tf_hits) - overlap
            ## check this part of code
            # neither = num_peaks - (overlap + module_only + tf_only)
            ## reset to exo number


            neither = num_exo_features - (overlap + module_only + tf_only)
            if neither < 0:
                neither = 0
            # print("tf_only {}, module_only {}, overlap {}, tf_hits {}".format(tf_only, module_only, overlap, tf_hits))
            # print("neither: {}".format(neither))


            contingency_matrix = np.array([[overlap, module_only], [tf_only, neither]])
            # print("contigency_matrix {}".format(contingency_matrix))
            stat,pval = fisher_exact(contingency_matrix, alternative='greater')
            pvals.append(pval)
            test_statistics.append(stat)

        results = [
            dict(**meta, pval = pval, test_statistic = test_stat)
            for meta, pval, test_stat in zip(metadata, pvals, test_statistics)
        ]
        self.enrichments[(factor_type, group_type)] = results
        return results, module_idx   


    def plot_compare_group_enrichments(self, factor_type = 'motifs', 
        label_factors = None, hue = None, palette = 'coolwarm', hue_order = None, 
        ax = None, figsize = (8,8), legend_label = '', show_legend = True, fontsize = 14, 
        pval_threshold = (1e-50, 1e-50), na_color = 'lightgrey',
        color = 'grey', label_closeness = 3, max_label_repeats = 3, show_factor_ids = False):
        '''
        It is often useful to contrast topic enrichments in order to
        understand which factors' influence is unique to certain
        cell states. Topics may be enriched for constitutively-active
        transcription factors, so comparing two similar topics to find
        the factors that are unique to each elucidates the dynamic
        aspects of regulation between states.
        This function contrasts the enrichments of two topics.
        Parameters
        ----------
        topic1, topic2 : int
            Which topics to compare.
        factor_type : str, 'motifs' or 'chip', default = 'motifs'
            Which factor type to use for enrichment.
        label_factors : list[str], np.ndarray[str], None; default=None
            List of factors to label. If not provided, will label all
            factors that meet the p-value thresholds.
        hue : dict[str : {str, float}] or None
            If provided, colors the factors on the plot. The keys of the dict
            must be the names of transcription factors, and the values are
            the associated data to map to colors. The values may be 
            categorical, e.g. cluster labels, or scalar, e.g. expression
            values. TFs not provided in the dict are colored as *na_color*.
        palette : str, list[str], or None; default = None
            Palette of plot. Default of None will set `palette` to the style-specific default.
        hue_order : list[str] or None, default = None
            Order to assign hues to features provided by `data`. Works similarly to
            hue_order in seaborn. User must provide list of features corresponding to 
            the order of hue assignment. 
        ax : matplotlib.pyplot.axes, deafult = None
            Provide axes object to function to add streamplot to a subplot composition,
            et cetera. If no axes are provided, they are created internally.
        figsize : tuple(float, float), default = (8,8)
            Size of figure
        legend_label : str, None
            Label for legend.
        show_legend : boolean, default=True
            Show figure legend.
        fontsize : int>0, default=13
            Fontsize of TF labels on plot.
        pval_threshold : tuple[float, float], default=(1e-50, 1e-50)
            Threshold below with TFs will not be labeled on plot. The first and
            second positions relate p-value with respect to topic 1 and topic 2.
        na_color : str, default='lightgrey'
            Color for TFs with no provided *hue*
        color : str, default='grey'
            If *hue* not provided, colors all points on plot this color.
        label_closeness : int>0, default=3
            Closeness of TF labels to points on plot. When *label_closeness* is high,
            labels are forced to be very close to points.
        max_label_repeats : boolean, default=3
            Some TFs have multiple ChIP samples or Motif PWMs. For these factors,
            label the top *max_label_repeats* examples. This prevents clutter when
            many samples for the same TF are close together. The rank of the sample
            for each TF is shown in the label as "<TF name> (<rank>)".
        Returns
        -------
        matplotlib.pyplot.axes
        Examples
        --------
        .. code-block :: python
            >>> label = ['LEF1','HOXC13','MEOX2','DLX3','BACH2','RUNX1', 'SMAD2::SMAD3']
            >>> atac_model.plot_compare_topic_enrichments(23, 17,
            ...     label_factors = label, 
            ...     color = 'lightgrey',
            ...     fontsize=20, label_closeness=5, 
            ... )
        .. image:: /_static/mira.topics.AccessibilityModel.plot_compare_topic_enrichments.svg
            :width: 300
        '''

        m1 = self.enrichments[('motifs', 'decoupled')]
        m2 = self.enrichments[('motifs', 'coupled')]
        
        return plot_factor_influence(m1, m2, ax = ax, label_factors = label_factors,
            pval_threshold = pval_threshold, hue = hue, hue_order = hue_order, 
            palette = palette, legend_label = legend_label, show_legend = show_legend, label_closeness = label_closeness, 
            na_color = na_color, max_label_repeats = max_label_repeats, figsize=figsize,
            axlabels = ('Latent {} Enrichments'.format("decoupled"),'Latent {} Enrichments'.format(str("coupled"))), 
            fontsize = fontsize, color = color) 



        


