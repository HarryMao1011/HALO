from typing import Dict, Iterable, Optional

import numpy as np
import torch
from torch.distributions import Normal, Poisson
from torch.distributions import kl_divergence as kld

# from scvi import REGISTRY_KEYS
from scvi._compat import Literal
from scvi.distributions import NegativeBinomial, ZeroInflatedNegativeBinomial
from scvi.module._peakvae import Decoder as DecoderPeakVI
from scvi.module.base import BaseModuleClass, LossRecorder, auto_move_data
from scvi.nn import DecoderSCVI, Encoder, FCLayers
from scvi.module import MULTIVAE
from utils import torch_infer_nonsta_dir

from anndata import AnnData
from typing import Dict, Iterable, List, Optional, Sequence, Union
from scvi.model import MULTIVI 
from complementary_models import REGISTRY_KEYS
# from scvi import REGISTRY_KEYS
from scvi.data import AnnDataManager
from scvi.data.fields import (
    CategoricalJointObsField,
    CategoricalObsField,
    LayerField,
    NumericalJointObsField,
    NumericalObsField,
)
from scvi.utils._docstrings import doc_differential_expression, setup_anndata_dsp
from torch import tensor
# from infer_nonsta_dir import infer_nonsta_dir



class HALOVAE(MULTIVAE):
    """
    Variational auto-encoder model for joint paired + unpaired RNA-seq and ATAC-seq data.
    Parameters
    ----------
    n_input_regions
        Number of input regions.
    n_input_genes
        Number of input genes.
    n_batch
        Number of batches, if 0, no batch correction is performed.
    gene_likelihood
        The distribution to use for gene expression data. One of the following
        * ``'zinb'`` - Zero-Inflated Negative Binomial
        * ``'nb'`` - Negative Binomial
        * ``'poisson'`` - Poisson
    n_hidden
        Number of nodes per hidden layer. If `None`, defaults to square root
        of number of regions.
    n_latent
        Dimensionality of the latent space. If `None`, defaults to square root
        of `n_hidden`.
    n_layers_encoder
        Number of hidden layers used for encoder NN.
    n_layers_decoder
        Number of hidden layers used for decoder NN.
    dropout_rate
        Dropout rate for neural networks
    region_factors
        Include region-specific factors in the model
    use_batch_norm
        One of the following
        * ``'encoder'`` - use batch normalization in the encoder only
        * ``'decoder'`` - use batch normalization in the decoder only
        * ``'none'`` - do not use batch normalization
        * ``'both'`` - use batch normalization in both the encoder and decoder
    use_layer_norm
        One of the following
        * ``'encoder'`` - use layer normalization in the encoder only
        * ``'decoder'`` - use layer normalization in the decoder only
        * ``'none'`` - do not use layer normalization
        * ``'both'`` - use layer normalization in both the encoder and decoder
    latent_distribution
        which latent distribution to use, options are
        * ``'normal'`` - Normal distribution
        * ``'ln'`` - Logistic normal distribution (Normal(0, I) transformed by softmax)
    deeply_inject_covariates
        Whether to deeply inject covariates into all layers of the decoder. If False,
        covariates will only be included in the input layer.
    encode_covariates
        If True, include covariates in the input to the encoder.
    use_size_factor_key
        Use size_factor AnnDataField defined by the user as scaling factor in mean of conditional RNA distribution.
    """
    def __init__(
        self,
        n_input_regions: int = 0,
        n_input_genes: int = 0,
        n_batch: int = 0,
        gene_likelihood: Literal["zinb", "nb", "poisson"] = "zinb",
        n_hidden: Optional[int] = None,
        n_latent: Optional[int] = None,
        n_layers_encoder: int = 2,
        n_layers_decoder: int = 2,
        n_continuous_cov: int = 0,
        n_cats_per_cov: Optional[Iterable[int]] = None,
        dropout_rate: float = 0.1,
        region_factors: bool = True,
        use_batch_norm: Literal["encoder", "decoder", "none", "both"] = "none",
        use_layer_norm: Literal["encoder", "decoder", "none", "both"] = "both",
        latent_distribution: str = "normal",
        deeply_inject_covariates: bool = False,
        encode_covariates: bool = False,
        use_size_factor_key: bool = False,
        alpha : float = 0.3,
        beta_1: float = 1,
        beta_2: float = 1,
        beta_3: float = 1

    ):
    
        super().__init__(
            n_input_regions,
            n_input_genes,
            n_batch,
            gene_likelihood,
            n_hidden,
            n_latent,
            n_layers_encoder,
            n_layers_decoder,
            n_continuous_cov,
            n_cats_per_cov,
            dropout_rate,
            region_factors,
            use_batch_norm,
            use_layer_norm,
            latent_distribution,
            deeply_inject_covariates,
            encode_covariates,
            use_size_factor_key)
                # INIT PARAMS
        
        self.n_input_regions = n_input_regions
        self.n_input_genes = n_input_genes
        self.n_hidden = (
            int(np.sqrt(self.n_input_regions + self.n_input_genes))
            if n_hidden is None
            else n_hidden
        )
        self.n_batch = n_batch

        self.gene_likelihood = gene_likelihood
        self.latent_distribution = latent_distribution

        self.n_latent = int(np.sqrt(self.n_hidden)) if n_latent is None else n_latent
        self.n_layers_encoder = n_layers_encoder
        self.n_layers_decoder = n_layers_decoder
        self.n_cats_per_cov = n_cats_per_cov
        self.n_continuous_cov = n_continuous_cov
        self.dropout_rate = dropout_rate

        self.use_batch_norm_encoder = use_batch_norm in ("encoder", "both")
        self.use_batch_norm_decoder = use_batch_norm in ("decoder", "both")
        self.use_layer_norm_encoder = use_layer_norm in ("encoder", "both")
        self.use_layer_norm_decoder = use_layer_norm in ("decoder", "both")
        self.encode_covariates = encode_covariates
        self.deeply_inject_covariates = deeply_inject_covariates
        self.use_size_factor_key = use_size_factor_key
        self.alpha = alpha
        self.beta_1 = beta_1
        self.beta_2  =beta_2
        self.beta_3 = beta_3

        cat_list = (
            [n_batch] + list(n_cats_per_cov) if n_cats_per_cov is not None else []
        )

        n_input_encoder_acc = (
            self.n_input_regions + n_continuous_cov * encode_covariates
        )
        n_input_encoder_exp = self.n_input_genes + n_continuous_cov * encode_covariates
        encoder_cat_list = cat_list if encode_covariates else None    

        ## non_stationary independent atac encoder
        self.z_encoder_accessibility_indep = Encoder(
            n_input=n_input_encoder_acc,
            n_layers=self.n_layers_encoder,
            n_output=self.n_latent,
            n_hidden=self.n_hidden,
            n_cat_list=encoder_cat_list,
            dropout_rate=self.dropout_rate,
            activation_fn=torch.nn.LeakyReLU,
            distribution=self.latent_distribution,
            var_eps=0,
            use_batch_norm=self.use_batch_norm_encoder,
            use_layer_norm=self.use_layer_norm_encoder,
            return_dist=True,
        )

        ## non_stationary independent expression encoder
        self.z_encoder_expression_indep = Encoder(
            n_input=n_input_encoder_exp,
            n_layers=self.n_layers_encoder,
            n_output=self.n_latent,
            n_hidden=self.n_hidden,
            n_cat_list=encoder_cat_list,
            dropout_rate=self.dropout_rate,
            activation_fn=torch.nn.LeakyReLU,
            distribution=self.latent_distribution,
            var_eps=0,
            use_batch_norm=self.use_batch_norm_encoder,
            use_layer_norm=self.use_layer_norm_encoder,
            return_dist=True,
        )

        ## non_stationary dependent atac encoder
        self.z_encoder_accessibility_dep = Encoder(
            n_input=n_input_encoder_acc,
            n_layers=self.n_layers_encoder,
            n_output=self.n_latent,
            n_hidden=self.n_hidden,
            n_cat_list=encoder_cat_list,
            dropout_rate=self.dropout_rate,
            activation_fn=torch.nn.LeakyReLU,
            distribution=self.latent_distribution,
            var_eps=0,
            use_batch_norm=self.use_batch_norm_encoder,
            use_layer_norm=self.use_layer_norm_encoder,
            return_dist=True,
        )

        ## non_stationary dependent expression encoder
        self.z_encoder_expression_dep = Encoder(
            n_input=n_input_encoder_exp,
            n_layers=self.n_layers_encoder,
            n_output=self.n_latent,
            n_hidden=self.n_hidden,
            n_cat_list=encoder_cat_list,
            dropout_rate=self.dropout_rate,
            activation_fn=torch.nn.LeakyReLU,
            distribution=self.latent_distribution,
            var_eps=0,
            use_batch_norm=self.use_batch_norm_encoder,
            use_layer_norm=self.use_layer_norm_encoder,
            return_dist=True,
        )

        # expression decoder
        self.z_decoder_expression = DecoderSCVI(
            self.n_latent + self.n_continuous_cov,
            n_input_genes,
            n_cat_list=cat_list,
            n_layers=n_layers_decoder,
            n_hidden=self.n_hidden,
            inject_covariates=self.deeply_inject_covariates,
            use_batch_norm=self.use_batch_norm_decoder,
            use_layer_norm=self.use_layer_norm_decoder,
            scale_activation="softplus" if use_size_factor_key else "softmax",
        )

        # accessibility decoder
        self.z_decoder_accessibility = DecoderPeakVI(
            n_input=self.n_latent + self.n_continuous_cov,
            n_output=n_input_regions,
            n_hidden=self.n_hidden,
            n_cat_list=cat_list,
            n_layers=self.n_layers_decoder,
            use_batch_norm=self.use_batch_norm_decoder,
            use_layer_norm=self.use_layer_norm_decoder,
            deep_inject_covariates=self.deeply_inject_covariates,
        )

    def _get_inference_input(self, tensors):
        x = tensors[REGISTRY_KEYS.X_KEY]
        batch_index = tensors[REGISTRY_KEYS.BATCH_KEY]
        time_index = tensors[REGISTRY_KEYS.TIME_KEY]
        # cell_index = tensors[REGISTRY_KEYS.LABELS_KEY]
        cont_covs = tensors.get(REGISTRY_KEYS.CONT_COVS_KEY)
        cat_covs = tensors.get(REGISTRY_KEYS.CAT_COVS_KEY)
        input_dict = dict(
            x=x,
            batch_index=batch_index,
            time_index = time_index,
            # cell_index = cell_index,
            cont_covs=cont_covs,
            cat_covs=cat_covs,
        )
        return input_dict


    @auto_move_data
    def inference(
        self,
        x,
        batch_index,
        cont_covs,
        cat_covs,
        n_samples=1,
    ) -> Dict[str, torch.Tensor]:

        # Get Data and Additional Covs
        x_rna = x[:, : self.n_input_genes]
        x_chr = x[:, self.n_input_genes :]

        x_rna_2 = x_rna.clone()
        x_chr_2 = x_chr.clone()

        mask_expr = x_rna.sum(dim=1) > 0
        mask_acc = x_chr.sum(dim=1) > 0

        if cont_covs is not None and self.encode_covariates:
            encoder_input_expression = torch.cat((x_rna, cont_covs), dim=-1)
            encoder_input_accessibility = torch.cat((x_chr, cont_covs), dim=-1)
        else:
            encoder_input_expression = x_rna
            encoder_input_accessibility = x_chr

        if cont_covs is not None and self.encode_covariates:
            encoder_input_expression_2 = torch.cat((x_rna_2, cont_covs), dim=-1)
            encoder_input_accessibility_2 = torch.cat((x_chr_2, cont_covs), dim=-1)
        else:
            encoder_input_expression_2 = x_rna_2
            encoder_input_accessibility_2 = x_chr_2    

        if cat_covs is not None and self.encode_covariates:
            categorical_input = torch.split(cat_covs, 1, dim=1)
        else:
            categorical_input = tuple()

        # Z Encoders for both dependent (coupled) part
        qz_acc_dep, z_acc_dep = self.z_encoder_accessibility_dep(
            encoder_input_accessibility, batch_index, *categorical_input
        )
        qz_expr_dep, z_expr_dep = self.z_encoder_expression_dep(
            encoder_input_expression, batch_index, *categorical_input
        )
        qzm_acc_dep = qz_acc_dep.loc
        qzm_expr_dep = qz_expr_dep.loc
        qzv_acc_dep = qz_acc_dep.scale**2
        qzv_expr_dep = qz_expr_dep.scale**2


        qz_acc_indep, z_acc_indep = self.z_encoder_accessibility_indep(
            encoder_input_accessibility_2, batch_index, *categorical_input
        )
        qz_expr_indep, z_expr_indep = self.z_encoder_expression_indep(
            encoder_input_expression_2, batch_index, *categorical_input
        )
        qzm_acc_indep = qz_acc_indep.loc
        qzm_expr_indep = qz_expr_indep.loc
        qzv_acc_indep = qz_acc_indep.scale**2
        qzv_expr_indep = qz_expr_indep.scale**2


       # Z Encoders for independent (time-lagging)


        # L encoders
        libsize_expr = self.l_encoder_expression(
            encoder_input_expression, batch_index, *categorical_input
        )
        libsize_acc = self.l_encoder_accessibility(
            encoder_input_accessibility, batch_index, *categorical_input
        )

        # ReFormat Outputs
        if n_samples > 1:
            ## dependent part (coupled)
            untran_za_dep = qz_acc_dep.sample((n_samples,))
            z_acc_dep = self.z_encoder_accessibility_dep.z_transformation(untran_za_dep)
            untran_zr_dep = qz_expr_dep.sample((n_samples,))
            z_expr_dep = self.z_encoder_expression_dep.z_transformation(untran_zr_dep)
            ## independent part (time lagging part)
            untran_za_indep = qz_acc_indep.sample((n_samples,))
            z_acc_indep = self.z_encoder_accessibility_indep.z_transformation(untran_za_indep)
            untran_zr_indep = qz_expr_indep.sample((n_samples,))
            z_expr_indep = self.z_encoder_expression_indep.z_transformation(untran_zr_indep)



            libsize_expr = libsize_expr.unsqueeze(0).expand(
                (n_samples, libsize_expr.size(0), libsize_expr.size(1))
            )
            libsize_acc = libsize_acc.unsqueeze(0).expand(
                (n_samples, libsize_acc.size(0), libsize_acc.size(1))
            )

        ## Integrate the independent part and time lagging part
        # Sample from the average distribution
        qz_acc_m = (qzm_acc_dep + qzm_acc_indep) / 2
        qz_acc_v = (qzv_acc_dep + qzv_acc_indep) / (2**0.5)
        z_acc = Normal(qz_acc_m, qz_acc_v.sqrt()).rsample()

        qz_expr_m = (qzm_expr_dep + qzm_expr_indep) / 2
        qz_expr_v = (qzv_expr_dep + qzv_expr_indep) / (2**0.5)
        z_expr = Normal(qz_expr_m, qz_expr_v.sqrt()).rsample()


        # choose the correct latent representation based on the modality


        outputs = dict(

            ## coupled part
            z_expr_dep=z_expr_dep,
            qzm_expr_dep=qzm_expr_dep,
            qzv_expr_dep=qzv_expr_dep,

            z_acc_dep=z_acc_dep,
            qzm_acc_dep=qzm_acc_dep,
            qzv_acc_dep=qzv_acc_dep,
            
            ## lagging part 
            z_expr_indep=z_expr_indep,
            qzm_expr_indep=qzm_expr_indep,
            qzv_expr_indep=qzv_expr_indep,

            z_acc_indep=z_acc_indep,
            qzm_acc_indep=qzm_acc_indep,
            qzv_acc_indep=qzv_acc_indep,

            ## integrated expression
            qz_expr_m = qz_expr_m,
            qz_expr_v = qz_expr_v,
            z_expr = z_expr,
            
            ## integrated accessibility
            qz_acc_m = qz_acc_m , 
            qz_acc_v = qz_acc_v,
            z_acc = z_acc,

            libsize_expr=libsize_expr,
            libsize_acc=libsize_acc,
        )
        return outputs   


    ### change this function 
    def _get_generative_input(self, tensors, inference_outputs, transform_batch=None):
  
        z_expr = inference_outputs["z_expr"]
        qz_expr_v = inference_outputs["qz_expr_v"]
        qz_expr_m = inference_outputs["qz_expr_m"]
        z_acc = inference_outputs["z_acc"]
        qz_acc_v = inference_outputs["qz_acc_v"]
        qz_acc_m = inference_outputs["qz_acc_m"]

        libsize_expr = inference_outputs["libsize_expr"]
        libsize_acc = inference_outputs["libsize_acc"]

        size_factor_key = REGISTRY_KEYS.SIZE_FACTOR_KEY
        size_factor = (
            torch.log(tensors[size_factor_key])
            if size_factor_key in tensors.keys()
            else None
        )

        batch_index = tensors[REGISTRY_KEYS.BATCH_KEY]
        cont_key = REGISTRY_KEYS.CONT_COVS_KEY
        cont_covs = tensors[cont_key] if cont_key in tensors.keys() else None

        cat_key = REGISTRY_KEYS.CAT_COVS_KEY
        cat_covs = tensors[cat_key] if cat_key in tensors.keys() else None

        if transform_batch is not None:
            batch_index = torch.ones_like(batch_index) * transform_batch

        input_dict = dict(
            z_expr = z_expr,
            qz_expr_v=qz_expr_v,
            qz_expr_m = qz_expr_m,
            
            z_acc=z_acc,
            qz_acc_v=qz_acc_v,
            qz_acc_m=qz_acc_m,

            batch_index=batch_index,
            cont_covs=cont_covs,
            cat_covs=cat_covs,
            libsize_expr=libsize_expr,
            libsize_acc = libsize_acc,
            size_factor=size_factor,

        )
        return input_dict

    @auto_move_data
    def generative(
        self,
        z_expr,
        qzm_expr,
        z_acc,
        qzm_acc,

        batch_index,
        cont_covs=None,
        cat_covs=None,
        libsize_expr=None,
        libsize_acc= None,
        size_factor=None,
        use_z_mean=False,
    ):
        """Runs the generative model."""
        if cat_covs is not None:
            categorical_input = torch.split(cat_covs, 1, dim=1)
        else:
            categorical_input = tuple()

        latent_expr = z_expr if not use_z_mean else qzm_expr
        latent_acc = z_acc if not use_z_mean else qzm_acc
        
        if cont_covs is None:
            decoder_input_expr = latent_expr
            decoder_input_acc = latent_acc

        elif latent_expr.dim() != cont_covs.dim():
            
            decoder_input_expr = torch.cat(
                [latent_expr, cont_covs.unsqueeze(0).expand(latent_expr.size(0), -1, -1)], dim=-1
            )
            decoder_input_acc = torch.cat(
                [latent_acc, cont_covs.unsqueeze(0).expand(latent_acc.size(0), -1, -1)], dim=-1
            )

        else:
            decoder_input_expr = torch.cat([decoder_input_expr, cont_covs], dim=-1)
            decoder_input_acc = torch.cat([decoder_input_acc, cont_covs], dim=-1)

        # Accessibility Decoder
        p = self.z_decoder_accessibility(decoder_input_acc, batch_index, *categorical_input)

        # Expression Decoder
        if not self.use_size_factor_key:
            size_factor = libsize_expr
        px_scale, _, px_rate, px_dropout = self.z_decoder_expression(
            "gene", decoder_input_expr, size_factor, batch_index, *categorical_input
        )

        return dict(
            p=p,
            px_scale=px_scale,
            px_r=torch.exp(self.px_r),
            px_rate=px_rate,
            px_dropout=px_dropout,
        )                

    ## We need to change here
    def loss(
        self, tensors, inference_outputs, generative_outputs, kl_weight: float = 1.0
    ):
        # Get the data
        x = tensors[REGISTRY_KEYS.X_KEY]

        x_rna = x[:, : self.n_input_genes]
        x_chr = x[:, self.n_input_genes :]

        ## get the time
        time = tensors[REGISTRY_KEYS.TIME_KEY]

        ## get the cell type
        ## not considering the cell type at the moment
        # cell_type = tensors[REGISTRY_KEYS.LABELS_KEY]


        mask_expr = x_rna.sum(dim=1) > 0
        mask_acc = x_chr.sum(dim=1) > 0

        # Compute Accessibility loss
        x_accessibility = x[:, self.n_input_genes :]
        p = generative_outputs["p"]
        libsize_acc = inference_outputs["libsize_acc"]
        rl_accessibility = self.get_reconstruction_loss_accessibility(
            x_accessibility, p, libsize_acc
        )

        # Compute Expression loss
        px_rate = generative_outputs["px_rate"]
        px_r = generative_outputs["px_r"]
        px_dropout = generative_outputs["px_dropout"]
        x_expression = x[:, : self.n_input_genes]
        rl_expression = self.get_reconstruction_loss_expression(
            x_expression, px_rate, px_r, px_dropout
        )

        # mix losses to get the correct loss for each cell
        recon_loss = self._mix_modalities(
            rl_accessibility + rl_expression,  # paired
            rl_expression,  # expression
            rl_accessibility,  # accessibility
            mask_expr,
            mask_acc,
        )

    
        # Compute KLD between distributions for paired data
        
        qzm_expr = inference_outputs["qzm_expr"]
        qzv_expr = inference_outputs["qzv_expr"]

        qzm_acc = inference_outputs["qzm_acc"]
        qzv_acc = inference_outputs["qzv_acc"]

        z_expr_dep=inference_outputs['z_expr_dep'],
        qzm_expr_dep=inference_outputs['qzm_expr_dep'],
        qzv_expr_dep=inference_outputs['qzv_expr_dep'],
        
        z_acc_dep=inference_outputs['z_acc_dep'],
        qzm_acc_dep=inference_outputs['qzm_acc_dep'],
        qzv_acc_dep=inference_outputs['qzv_acc_dep'],
        
        ## lagging part 
        z_expr_indep=inference_outputs['z_expr_indep'],
        qzm_expr_indep=inference_outputs['qzm_expr_indep'],
        qzv_expr_indep=inference_outputs['qzv_expr_indep'],
        z_acc_indep= inference_outputs['z_acc_indep'],
        qzm_acc_indep= inference_outputs['qzm_acc_indep'],
        qzv_acc_indep= inference_outputs ['qzv_acc_indep'],


        kl_div_z = kld(
            Normal(qzm_expr, torch.sqrt(qzv_expr)), Normal(0, 1)
        ) + kld(
            Normal(qzm_acc, torch.sqrt(qzv_acc)), Normal(0, 1)
        )

        ## coupled matching KL divergnece, two modalities should carry similar information
        kld_paired = kld(
            Normal(qzm_expr_dep, torch.sqrt(qzv_expr_dep)), Normal(qzm_acc_dep, torch.sqrt(qzv_acc_dep))
        ) + kld(
            Normal(qzm_acc_dep, torch.sqrt(qzv_acc_dep)), Normal(qzm_expr_dep, torch.sqrt(qzv_expr_dep))
        )   

        ## scores of coupled modalities should be both greater or equal than \alpha

        a2rscore_coupled_loss = self.beta_1 * torch.max(self.alpha - torch_infer_nonsta_dir(z_acc_dep, z_expr_dep, time), 0)
        r2ascore_coupled_loss = self.beta_1 * torch.max(self.alpha - torch_infer_nonsta_dir(z_expr_dep, z_acc_dep, time), 0)

        
        ## calculate the lagging (cd-nod nonstationary condtions) constrains

        ## scores of lagging modalities (ATAC --> RNA) should be smaller than \alpha
        a2rscore_lagging = torch_infer_nonsta_dir(z_acc_indep, z_expr_indep, time)
        a2rscore_lagging_loss = self.beta_2 *torch.max(a2rscore_lagging-self.alpha-1e-10, 0)

        ## scores of lagging modalities (ATAC --> RNA) should be smaller than (RNA-->ATAC)
        r2ascore_lagging = torch_infer_nonsta_dir(z_expr_indep, z_acc_indep, time)
        a2r_r2a_score_loss = self.beta_2 * torch.max(a2rscore_lagging-r2ascore_lagging-1e10, 0)

        ## non_stationary loss
        nod_loss = a2rscore_coupled_loss + r2ascore_coupled_loss + a2rscore_lagging_loss + a2r_r2a_score_loss


        # KL WARMUP
        kl_local_for_warmup = kl_div_z + kld_paired
        weighted_kl_local = kl_weight * kl_local_for_warmup

        # PENALTY
        # distance_penalty = kl_weight * torch.pow(z_acc - z_expr, 2).sum(dim=1)

        # TOTAL LOSS
        loss = torch.mean(recon_loss + weighted_kl_local)

        kl_local = dict(kl_divergence_z=kl_div_z)
        kl_global = torch.tensor(0.0)
        return LossRecorder(loss, recon_loss, kl_local, kl_global, nod_loss)
