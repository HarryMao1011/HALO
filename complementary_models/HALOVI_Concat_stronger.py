from typing import Dict, Iterable, Optional
import torch
import numpy as np
import torch
from torch.distributions import Normal, Poisson
from torch.distributions import kl_divergence as kld
from scvi.utils._docstrings import setup_anndata_dsp

from .REGISTRY_KEYS import REGISTRY_KEYS
from scvi._compat import Literal

from scvi.distributions import NegativeBinomial, ZeroInflatedNegativeBinomial
from scvi.module._peakvae import Decoder as DecoderPeakVI
from scvi.module.base import BaseModuleClass, LossRecorder, auto_move_data
from scvi.nn import DecoderSCVI, Encoder, FCLayers
from scvi.module import MULTIVAE
from .parallel_model import MultiVI_Parallel
from anndata import AnnData
from typing import Dict, Iterable, List, Optional, Sequence, Union
from scvi.model import MULTIVI 
from scvi.data import AnnDataManager
from scvi.data.fields import (
    CategoricalJointObsField,
    CategoricalObsField,
    LayerField,
    NumericalJointObsField,
    NumericalObsField,
)
from .utils import torch_infer_nonsta_dir

from scvi.train import AdversarialTrainingPlan, TrainRunner, SemiSupervisedTrainingPlan
from scvi.train._callbacks import SaveBestState
from scvi.dataloaders import DataSplitter

device = 'cuda'
if torch.cuda.is_available():
    device='cuda'
else:
    device='cpu'

class HALOVAECAT2(MULTIVAE):
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
        n_latent_dep: Optional[int] = 5,
        n_latent_indep: Optional[int] = 5,
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
        beta_1: float = 5,
        beta_2: float = 5,
        beta_3: float = 5

    ):
    
        n_latent = n_latent_dep + n_latent_indep
        self.n_latent = n_latent
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
        self.n_latent_dep = n_latent_dep
        self.n_latent_indep = n_latent_indep
        print(n_latent_indep)

        self.n_hidden = (
            int(np.sqrt(self.n_input_regions + self.n_input_genes))
            if n_hidden is None
            else n_hidden
        )
        # self.device = 'cuda'
        # if torch.cuda.is_available():
        #     self.device='cuda'
        # else:
        #     self.device='cpu'
        self.n_batch = n_batch

        self.gene_likelihood = gene_likelihood
        self.latent_distribution = latent_distribution

        self.n_latent = int(np.sqrt(self.n_hidden) / 2) if n_latent is None else n_latent
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
        # print("alpha: {}, beta1: {}, beta2: {}, beta3: {}"\
        #     .format(self.alpha, self.beta_1, self.beta_2, self.beta_3))
        # print("n_latent_dep :{}, n_latent_indep: {}".format(n_latent_dep, n_latent_indep))    

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
            n_output=self.n_latent_indep,
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
            n_output=self.n_latent_indep,
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
            n_output=self.n_latent_dep,
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
            n_output=self.n_latent_dep,
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
        # print("n latent total is {}".format(self.n_latent_total))

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
        # print("try to get time key")

        time_index = tensors[REGISTRY_KEYS.TIME_KEY]
        # print(time_index)
        # cell_index = tensors[REGISTRY_KEYS.LABELS_KEY]
        cont_covs = tensors.get(REGISTRY_KEYS.CONT_COVS_KEY)
        cat_covs = tensors.get(REGISTRY_KEYS.CAT_COVS_KEY)
        input_dict = dict(
            x=x,
            batch_index=batch_index,
            time_index = time_index,
            cont_covs=cont_covs,
            cat_covs=cat_covs,
        )
        return input_dict


    @auto_move_data
    def inference(
        self,
        x,
        time_index,
        batch_index,
        cont_covs,
        cat_covs,
        n_samples=1,
    ) -> Dict[str, torch.Tensor]:

        # Get Data and Additional Covs
        x_rna = x[:, : self.n_input_genes]
        x_chr = x[:, self.n_input_genes :]
        device= 'cuda' if torch.cuda.is_available() else "cpu"
        x_rna_2 = x_rna.clone().detach().requires_grad_(True).to(device)
        x_chr_2 =  x_chr.clone().detach().requires_grad_(True).to(device)

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
        qz_acc_m = torch.cat([qzm_acc_dep , qzm_acc_indep], axis=-1)
        # print("contact qz_acc_m shape {}".format(qz_acc_m.shape))


        qz_acc_v = torch.cat([qzv_acc_dep, qzv_acc_indep], axis=-1)
        # print("contact qz_acc_m shape {}".format(qz_acc_v.shape))

        z_acc = Normal(qz_acc_m, qz_acc_v.sqrt()).rsample()
        # print("z_acc shape {}".format(z_acc.shape))

        qz_expr_m = torch.concat([qzm_expr_dep ,qzm_expr_indep], axis=-1) 
        qz_expr_v = torch.concat([qzv_expr_dep, qzv_expr_indep], axis=-1) 
        z_expr = Normal(qz_expr_m, qz_expr_v.sqrt()).rsample()
        # print("z_expr shape {}".format(z_expr.shape))


        # print("transformed qzv_acc_dep type: {}, value{}".format(type(qzv_acc_dep), qzv_acc_dep))

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
            time_key = time_index,

            libsize_expr=libsize_expr,
            libsize_acc=libsize_acc,

            z=z_expr,
            qz_m=qz_expr_m,
            qz_v=qz_expr_v
        )
        return outputs   


    ### change this function 
    def _get_generative_input(self, tensors, inference_outputs, transform_batch=None):
  
        
        qzm_expr = inference_outputs["qz_expr_m"]
        qzv_expr = inference_outputs["qz_expr_v"]
        z_expr = inference_outputs["qz_expr_v"]

        qzm_acc = inference_outputs["qz_acc_m"]
        qzv_acc = inference_outputs["qz_acc_v"]
        z_acc = inference_outputs['z_acc']


        z_expr_dep=inference_outputs['z_expr_dep']
        qzm_expr_dep=inference_outputs['qzm_expr_dep']
        qzv_expr_dep=inference_outputs['qzv_expr_dep']
        
        z_acc_dep=inference_outputs['z_acc_dep']
        qzm_acc_dep=inference_outputs['qzm_acc_dep']
        qzv_acc_dep=inference_outputs['qzv_acc_dep']
        
        ## lagging part 
        z_expr_indep=inference_outputs['z_expr_indep']
        qzm_expr_indep=inference_outputs['qzm_expr_indep']
        qzv_expr_indep=inference_outputs['qzv_expr_indep']
        z_acc_indep= inference_outputs['z_acc_indep']
        qzm_acc_indep= inference_outputs['qzm_acc_indep']
        qzv_acc_indep= inference_outputs ['qzv_acc_indep']
        
        time_key = inference_outputs['time_key']

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
        time_key = tensors[REGISTRY_KEYS.TIME_KEY]

        cat_key = REGISTRY_KEYS.CAT_COVS_KEY
        cat_covs = tensors[cat_key] if cat_key in tensors.keys() else None

        if transform_batch is not None:
            batch_index = torch.ones_like(batch_index) * transform_batch

        input_dict = dict(
            z_expr = z_expr,
            # qz_expr_v=qzv_acc,
            qzm_expr = qzm_expr,
            
            z_acc=z_acc,
            # qz_acc_v=qzv_acc,
            qzm_acc=qzm_acc,
            time_key = time_key,

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
        time_key,
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
        global device 
        # Get the data
        x = tensors[REGISTRY_KEYS.X_KEY]

        x_rna = x[:, : self.n_input_genes]
        x_chr = x[:, self.n_input_genes :]

        ## get the time

        # time = inference_outputs["time"]

        ## get the cell type
        ## not using the cell type at the moment
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
        
        qzm_expr = inference_outputs["qz_expr_m"]
        qzv_expr = inference_outputs["qz_expr_v"]

        qzm_acc = inference_outputs["qz_acc_m"]
        qzv_acc = inference_outputs["qz_acc_v"]

        z_expr_dep=inference_outputs['z_expr_dep']
        qzm_expr_dep=inference_outputs['qzm_expr_dep']
        qzv_expr_dep=inference_outputs['qzv_expr_dep']
        
        z_acc_dep=inference_outputs['z_acc_dep']
        qzm_acc_dep=inference_outputs['qzm_acc_dep']
        qzv_acc_dep=inference_outputs['qzv_acc_dep']
        # print("loss qzv_acc_dep type: {}".format(type(qzv_acc_dep)) )       
        ## lagging part 
        z_expr_indep=inference_outputs['z_expr_indep']
        qzm_expr_indep=inference_outputs['qzm_expr_indep']
        qzv_expr_indep=inference_outputs['qzv_expr_indep']
        z_acc_indep= inference_outputs['z_acc_indep']
        qzm_acc_indep= inference_outputs['qzm_acc_indep']
        qzv_acc_indep= inference_outputs ['qzv_acc_indep']
        time = inference_outputs['time_key']

        kl_div_z = 100* kld(
            Normal(qzm_expr, torch.sqrt(qzv_expr)), Normal(0, 1)
        ) + 100 *kld(
            Normal(qzm_acc, torch.sqrt(qzv_acc)), Normal(0, 1)
        )

        kld_paired = kld(
            Normal(qzm_expr_dep, qzv_expr_dep.sqrt()), Normal(qzm_acc_dep, qzv_acc_dep.sqrt())
        ) + 100*kld(
            Normal(qzm_acc_dep, qzv_acc_dep.sqrt()), Normal(qzm_expr_dep, qzv_expr_dep.sqrt()))

        kld_paired = torch.cat([kld_paired, torch.zeros_like(qzm_acc_indep, dtype=torch.float64).to(device)], axis=-1) 
        # print(kld_paired.shape, kl_div_z.shape)    
        # )  - kld(Normal(qzm_acc_dep, qzv_acc_dep.sqrt()), Normal(qzm_acc_indep, qzv_acc_indep.sqrt()))\
        #     - kld(Normal(qzm_acc_indep, qzv_acc_indep.sqrt()), Normal(qzm_acc_dep, qzv_acc_dep.sqrt())) \
        # -kld(Normal(qzm_expr_dep, qzv_expr_dep.sqrt()), Normal(qzm_expr_indep, qzv_expr_indep.sqrt()))\
        #     -kld(Normal(qzm_expr_indep, qzv_expr_indep.sqrt()), Normal(qzm_expr_dep, qzv_expr_dep.sqrt()))

        ## scores of coupled modalities should be both greater or equal than \alpha
        # print("shapes are {}".format(z_acc_dep.shape, z_expr_dep.shape))
        # print(z_acc_dep.get_device(), z_expr_dep.get_device(), time.get_device())
        a2rscore_coupled, _, _ = torch_infer_nonsta_dir(z_acc_dep, z_expr_dep, time)
        r2ascore_coupled, _, _ = torch_infer_nonsta_dir(z_expr_dep, z_acc_dep, time)
        
        # print("coupled  ATAC->RNA {}, RNA->ATAC {}".format(a2rscore_coupled, r2ascore_coupled))
        self.alpha=0.01

        a2rscore_coupled_loss = torch.maximum(self.alpha - a2rscore_coupled + 1e-3, torch.tensor(0))
        r2ascore_coupled_loss = torch.maximum(self.alpha - r2ascore_coupled + 1e-3, torch.tensor(0))

        # a2rscore_coupled_loss = -self.beta_1 *  a2rscore_coupled 
        # r2ascore_coupled_loss = -self.beta_1 *  r2ascore_coupled 

        # print("coupled loss {}, {}". format(a2rscore_coupled_loss, r2ascore_coupled_loss))

        
        ## calculate the lagging (cd-nod nonstationary condtions) constrains

        ## scores of lagging modalities (ATAC --> RNA) should be smaller than \alpha
        # z_acc_indep = torch.tensor(z_acc_indep)
        # z_expr_indep = torch.tensor(z_expr_indep)
        

        a2rscore_lagging, _, _ = torch_infer_nonsta_dir(z_acc_indep, z_expr_indep, time)
        r2ascore_lagging, _, _ = torch_infer_nonsta_dir(z_expr_indep, z_acc_indep, time)
        a2r_r2a_score_loss =  torch.maximum(a2rscore_lagging-r2ascore_lagging+1e-4, torch.tensor(0))

        # print("Lagging ATAC->RNA score {}, RNA->ATAC {}". format(a2rscore_lagging, r2ascore_lagging))
        # print("a2r_r2a_score_loss loss {}".format(a2r_r2a_score_loss))


        ## non_stationary loss
        # print("a2rscore_coupled_loss type {} /n,r2ascore_coupled_loss type ".format(type(a2rscore_coupled_loss), ))
        # print("a2rscore_coupled_loss: {} , r2ascore_coupled_loss: {} /n a2rscore_lagging_loss: {},  a2r_r2a_score_loss: {}"\
            # .format(a2rscore_coupled_loss, r2ascore_coupled_loss, a2rscore_lagging_loss, a2r_r2a_score_loss))
        # print("independent distance ATAC-RNA {}".format(a2rscore_lagging-r2ascore_lagging))

        # nod_loss = -1 *self.beta_1 *(-1 * a2rscore_coupled.to(torch.float64) -1 * r2ascore_coupled.to(torch.float64) \
        #     -  r2ascore_coupled.to(torch.float64) +  a2rscore_lagging.to(torch.float64)\
        #     + a2r_r2a_score_loss.to(torch.float64))

        nod_weight = 0

        self.beta_2 = 5e5
        self.beta_3 = 1e8
        self.beta_1 = 1e6
        nod_loss =   self.beta_1 *  a2rscore_lagging.to(torch.float64)  + self.beta_3 * a2r_r2a_score_loss + \
        self.beta_2 * a2rscore_coupled_loss + self.beta_2 * a2rscore_coupled_loss + self.beta_2*r2ascore_coupled_loss

        # nod_loss =  nod_loss - self.beta_2 * r2ascore_lagging.to(torch.float64)-self.beta_2 * a2rscore_coupled.to(torch.float64) -self.beta_2*r2ascore_coupled


        # nod_loss = a2rscore_coupled_loss.to(torch.float64) + r2ascore_coupled_loss.to(torch.float64) \
        #     + a2rscore_lagging_loss.to(torch.float64) + a2r_r2a_score_loss.to(torch.float64)

        # KL WARMUP
        # print("kld_paired shape {}".format(kld_paired.shape))

        # print("kld {}, nod_loss {}".format(kl_div_z.mean(), nod_loss))
        # # kl_div_z =kl_div_z + nod_loss * torch.ones_like(kl_div_z)
        # print("after changes {}".format(kl_div_z.mean()))

        kl_local_for_warmup = kl_div_z + kld_paired


        weighted_kl_local = kl_weight * kl_local_for_warmup

        # print("kld_paired shape {}".format(kld_paired.shape))
        # print("weighted_kl_local shape {}".format(weighted_kl_local.shape))
        # print("before reconstructon loss {}, nod_loss {}".format(recon_loss.mean(), nod_loss))
        # print("nod loss {}".format(nod_loss.shape))
        # print(type(nod_loss))
        weight = 1
        recon_loss = weight* recon_loss + nod_weight *nod_loss * torch.ones_like(recon_loss)
        print("n_indep: {} after reconstructon loss {}, beta:{}".format(self.n_latent_indep, recon_loss.mean(), self.beta_1))

        # PENALTY
        # distance_penalty = kl_weight * torch.pow(z_acc - z_expr, 2).sum(dim=1)

        # TOTAL LOSS
        # print(weighted_kl_local)
        # recon_loss = recon_loss 

        loss = torch.mean(recon_loss.unsqueeze(1) + weighted_kl_local)
        # loss = torch.mean(recon_loss.unsqueeze(1))

        # loss = torch.mean(recon_loss.unsqueeze(1) + weighted_kl_local) 

        # print("plus recon_loss {} {}, kl_divergence {}, {}".format(recon_loss.mean(), recon_loss.shape, weighted_kl_local.mean(), weighted_kl_local.shape))

        kl_local = dict(kl_divergence_z=kl_div_z)
        kl_global = torch.tensor(0)
        # print("independent distance ATAC-RNA {}, nod_loss {}".format(a2rscore_lagging-r2ascore_lagging, nod_loss))

        # print("loss : {}, recon_loss: {}".format(loss, recon_loss))
        return LossRecorder(loss, recon_loss, kl_local, kl_global)



class HALOVICAT2(MultiVI_Parallel):

    def __init__(
        self,
        adata: AnnData,
        n_genes: int,
        n_regions: int,
        n_hidden: Optional[int] = None,
        n_latent_dep: Optional[int] = 5,
        n_latent_indep: Optional[int] = 5,
        n_layers_encoder: int = 2,
        n_layers_decoder: int = 2,
        dropout_rate: float = 0.1,
        region_factors: bool = True,
        gene_likelihood: Literal["zinb", "nb", "poisson"] = "zinb",
        use_batch_norm: Literal["encoder", "decoder", "none", "both"] = "none",
        use_layer_norm: Literal["encoder", "decoder", "none", "both"] = "both",
        latent_distribution: Literal["normal", "ln"] = "normal",
        deeply_inject_covariates: bool = False,
        encode_covariates: bool = False,
        fully_paired: bool = False,
        alpha: float = 0.01,
        beta_1: float=1e4 ,
        beta_2: float=1e4,
        beta_3: float=1e4,

        **model_kwargs,
    ):
        n_latent = None
        super().__init__(adata,
        n_genes,
        n_regions,
        n_hidden,
        n_latent,
        n_layers_encoder,
        n_layers_decoder,
        dropout_rate,
        region_factors,
        gene_likelihood,
        use_batch_norm,
        use_layer_norm,
        latent_distribution,
        deeply_inject_covariates,
        encode_covariates,
        fully_paired,
        **model_kwargs,)
        
        self.alpha = alpha
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.beta_3 = beta_3
        
        n_cats_per_cov = (
            self.adata_manager.get_state_registry(
                REGISTRY_KEYS.CAT_COVS_KEY
            ).n_cats_per_key
            if REGISTRY_KEYS.CAT_COVS_KEY in self.adata_manager.data_registry
            else []
        )

        use_size_factor_key = (
            REGISTRY_KEYS.SIZE_FACTOR_KEY in self.adata_manager.data_registry
        )

        print("time key in registry : {}".format(REGISTRY_KEYS.TIME_KEY in self.adata_manager.data_registry ))
        print("cell type key in registry: {}".format(REGISTRY_KEYS.LABELS_KEY in self.adata_manager.data_registry))

        
        self.module = HALOVAECAT2(
            n_input_genes=n_genes,
            n_input_regions=n_regions,
            n_batch=self.summary_stats.n_batch,
            n_hidden=n_hidden,
            n_latent_dep=n_latent_dep,
            n_latent_indep=n_latent_indep,
            n_layers_encoder=n_layers_encoder,
            n_layers_decoder=n_layers_decoder,
            n_continuous_cov=self.summary_stats.get("n_extra_continuous_covs", 0),
            n_cats_per_cov=n_cats_per_cov,
            dropout_rate=dropout_rate,
            region_factors=region_factors,
            gene_likelihood=gene_likelihood,
            use_batch_norm=use_batch_norm,
            use_layer_norm=use_layer_norm,
            use_size_factor_key=use_size_factor_key,
            latent_distribution=latent_distribution,
            deeply_inject_covariates=deeply_inject_covariates,
            encode_covariates=encode_covariates,
            alpha=self.alpha,
            beta_1= self.beta_1,
            beta_2=self.beta_2,
            beta_3=self.beta_3,
            **model_kwargs,
        )


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
        if not self.is_trained_:
            raise RuntimeError("Please train the model first.")

        keys = {
        ## integrated RNA
        "qz_expr_m": "qz_expr_m", "qz_expr_v": "qz_expr_v" , "z_expr":"z_expr",
        ## integrated ATAC
        "qz_acc_m": "qz_acc_m", "qz_acc_v": "qz_acc_v",  "z_acc": "z_acc",
        ## dependent RNA components 
        "z_expr_dep":"z_expr_dep", "qzm_expr_dep":"qzm_expr_dep", "qzv_expr_dep":"qzv_expr_dep",
        ## dependent ATAC components
        "z_acc_dep":"z_acc_dep", "qzm_acc_dep":"qzm_acc_dep", "qzv_acc_dep":"qzv_acc_dep",
        ## independent/lagging RNA components
        "z_expr_indep": "z_expr_indep","qzm_expr_indep": "qzm_expr_indep", "qzv_expr_indep":"qzv_expr_indep",
        ## independent/lagging ATAC components
        "z_acc_indep": "z_acc_indep", "qzm_acc_indep":"qzm_acc_indep", "qzv_acc_indep":"qzv_acc_indep"
        }
        

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
            # qz_m = outputs[keys["qz_m"]]
            # qz_v = outputs[keys["qz_v"]]
            # z = outputs[keys["z"]]
            
            ## integrated RNA expression
            qzm_expr = outputs[keys["qz_expr_m"]]
            qzv_expr = outputs[keys["qz_expr_v"]]
            z_expr = outputs[keys["z_expr"]]
            
            ## integrated ATAC 
            qzm_acc = outputs[keys["qz_acc_m"]]
            qzv_acc = outputs[keys["qz_acc_v"]]
            z_acc = outputs[keys["z_acc"]]
            
            ## dependent RNA
            qzm_expr_dep = outputs[keys["qzm_expr_dep"]]
            qzv_expr_dep = outputs[keys["qzv_expr_dep"]]
            # print("loss qzv_expr_dep type {}".format(type(qzv_expr_dep)))
            z_expr_dep = outputs[keys["z_expr_dep"]]
            
            ## dependent ATAC
            z_acc_dep=outputs[keys["z_acc_dep"]]
            qzm_acc_dep=outputs[keys['qzm_acc_dep']]
            qzv_acc_dep=outputs[keys['qzv_acc_dep']]

            ## independent RNA
            z_expr_indep=outputs[keys["z_expr_indep"]]
            qzm_expr_indep=outputs[keys['qzm_expr_indep']]
            qzv_expr_indep=outputs[keys['qzv_expr_indep']]

            ## independent ATAC
            z_acc_indep=outputs[keys["z_acc_indep"]]
            qzm_acc_indep=outputs[keys['qzm_acc_indep']]
            qzv_acc_indep=outputs[keys['qzv_acc_indep']]

            time_keys = outputs['time_key']


            if give_mean:
                # does each model need to have this latent distribution param?
                if self.module.latent_distribution == "ln":
                    # samples = Normal(qz_m, qz_v.sqrt()).sample([1])
                    # z = torch.nn.functional.softmax(samples, dim=-1)
                    # z = z.mean(dim=0)

                    samples_expr = Normal(qzm_expr, qzv_expr.sqrt()).sample([1])
                    z_expr = torch.nn.functional.softmax(samples_expr, dim=-1)
                    z_expr = z_expr.mean(dim=0)

                    samples_atac = Normal(qzm_acc, qzv_acc.sqrt()).sample([1])
                    z_acc = torch.nn.functional.softmax(samples_atac, dim=-1)
                    z_acc = z_acc.mean(dim=0)

                    sample_atac_indep =  Normal(qzm_acc_indep, qzv_acc_indep.sqrt()).sample([1])
                    z_acc_indep = torch.nn.functional.softmax(sample_atac_indep, dim=-1)
                    z_acc_indep = z_acc_indep.mean(dim=0)

                    sample_expr_indep =  Normal(qzm_expr_indep, qzv_expr_indep.sqrt()).sample([1])
                    z_expr_indep = torch.nn.functional.softmax(sample_expr_indep, dim=-1)
                    z_expr_indep = z_expr_indep.mean(dim=0)

                    sample_atac_dep =  Normal(qzm_acc_dep, qzv_acc_dep.sqrt()).sample([1])
                    z_acc_dep = torch.nn.functional.softmax(sample_atac_dep, dim=-1)
                    z_acc_dep = z_acc_dep.mean(dim=0)

                    sample_expr_dep =  Normal(qzm_expr_dep, qzv_expr_dep.sqrt()).sample([1])
                    z_expr_indep = torch.nn.functional.softmax(sample_expr_dep, dim=-1)
                    z_expr_indep = z_expr_indep.mean(dim=0)


                else:
                    z_acc = qzm_acc
                    z_expr = qzm_expr
                    z_acc_indep = qzm_acc_indep
                    z_expr_indep = qzm_expr_indep
                    z_acc_dep = qzm_acc_dep
                    z_expr_dep = qzm_expr_dep

            
            # latent += [z.cpu()]
            latent_atac+= [z_acc.cpu()]
            latent_expr += [z_expr.cpu()]

            latent_atac_dep+= [z_acc_dep.cpu()]
            latent_expr_dep += [z_expr_dep.cpu()]          

            latent_atac_indep+= [z_acc_indep.cpu()]
            latent_expr_indep += [z_expr_indep.cpu()]
            times += [time_keys.cpu()]



        return torch.cat(latent_atac).numpy(), torch.cat(latent_expr).numpy(), \
            torch.cat(latent_atac_dep).numpy(), torch.cat(latent_expr_dep).numpy(), \
                torch.cat(latent_atac_indep).numpy(), torch.cat(latent_expr_indep).numpy(), torch.cat(times).numpy()

    def train(
        self,
        max_epochs: int = 500,
        lr: float = 1e-4,
        use_gpu: Optional[Union[str, int, bool]] = None,
        train_size: float = 0.9,
        validation_size: Optional[float] = None,
        batch_size: int = 128,
        weight_decay: float = 1e-3,
        eps: float = 1e-08,
        early_stopping: bool = True,
        save_best: bool = True,
        check_val_every_n_epoch: Optional[int] = None,
        n_steps_kl_warmup: Optional[int] = None,
        n_epochs_kl_warmup: Optional[int] = 50,
        adversarial_mixing: bool = False,
        plan_kwargs: Optional[dict] = None,
        **kwargs,
    ):
        """
        Trains the model using amortized variational inference.
        Parameters
        ----------
        max_epochs
            Number of passes through the dataset.
        lr
            Learning rate for optimization.
        use_gpu
            Use default GPU if available (if None or True), or index of GPU to use (if int),
            or name of GPU (if str), or use CPU (if False).
        train_size
            Size of training set in the range [0.0, 1.0].
        validation_size
            Size of the test set. If `None`, defaults to 1 - `train_size`. If
            `train_size + validation_size < 1`, the remaining cells belong to a test set.
        batch_size
            Minibatch size to use during training.
        weight_decay
            weight decay regularization term for optimization
        eps
            Optimizer eps
        early_stopping
            Whether to perform early stopping with respect to the validation set.
        save_best
            Save the best model state with respect to the validation loss, or use the final
            state in the training procedure
        check_val_every_n_epoch
            Check val every n train epochs. By default, val is not checked, unless `early_stopping` is `True`.
            If so, val is checked every epoch.
        n_steps_kl_warmup
            Number of training steps (minibatches) to scale weight on KL divergences from 0 to 1.
            Only activated when `n_epochs_kl_warmup` is set to None. If `None`, defaults
            to `floor(0.75 * adata.n_obs)`.
        n_epochs_kl_warmup
            Number of epochs to scale weight on KL divergences from 0 to 1.
            Overrides `n_steps_kl_warmup` when both are not `None`.
        adversarial_mixing
            Whether to use adversarial training to penalize the model for umbalanced mixing of modalities.
        plan_kwargs
            Keyword args for :class:`~scvi.train.TrainingPlan`. Keyword arguments passed to
            `train()` will overwrite values present in `plan_kwargs`, when appropriate.
        **kwargs
            Other keyword args for :class:`~scvi.train.Trainer`.
        """
        update_dict = dict(
            lr=lr,
            adversarial_classifier=adversarial_mixing,
            weight_decay=weight_decay,
            eps=eps,
            n_epochs_kl_warmup=n_epochs_kl_warmup,
            n_steps_kl_warmup=n_steps_kl_warmup,
            optimizer="AdamW",
            scale_adversarial_loss=1,
        )
        if plan_kwargs is not None:
            plan_kwargs.update(update_dict)
        else:
            plan_kwargs = update_dict

        if save_best:
            if "callbacks" not in kwargs.keys():
                kwargs["callbacks"] = []
            kwargs["callbacks"].append(
                SaveBestState(monitor="reconstruction_loss_validation")
            )

        data_splitter = DataSplitter(
            self.adata_manager,
            train_size=train_size,
            validation_size=validation_size,
            batch_size=batch_size,
            use_gpu=use_gpu,
        )
        training_plan = AdversarialTrainingPlan(self.module, **plan_kwargs)
        runner = TrainRunner(
            self,
            training_plan=training_plan,
            data_splitter=data_splitter,
            max_epochs=max_epochs,
            use_gpu=use_gpu,
            early_stopping=early_stopping,
            check_val_every_n_epoch=check_val_every_n_epoch,
            early_stopping_monitor="reconstruction_loss_validation",
            early_stopping_patience=50,
            **kwargs,
        )
        return runner()            

    