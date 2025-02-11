"""Main module."""
from typing import Callable, Iterable, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import logsumexp
from torch.distributions import Normal
from torch.distributions import kl_divergence as kl

from .REGISTRY_KEYS import REGISTRY_KEYS
import logging
from typing import List, Optional

from anndata import AnnData
from scipy.sparse import csr_matrix

from scvi._compat import Literal
from scvi.distributions import NegativeBinomial, Poisson, ZeroInflatedNegativeBinomial
from scvi.module.base import  LossRecorder, auto_move_data, BaseModuleClass
# from scvi.nn import DecoderSCVI, Encoder, LinearDecoderSCVI, one_hot
from scvi.nn import DecoderSCVI, Encoder, one_hot
from ._base_components import NeuralDecoderRNA as LinearDecoderSCVI
# from scvi.nn import NeuralDecoderRNA as LinearDecoderSCVI
# from scvi.module._peakvae import NeuralGateDecoder as GateDecoder 
from scvi.module._peakvae import Decoder as DecoderPeakVI
from .__peak_vae import NeuralGateDecoder as GateDecoder
import torch.nn as nn

from scvi.module import VAE 
from torch.distributions import kl_divergence as kld
from .utils import torch_infer_nonsta_dir

torch.backends.cudnn.benchmark = True


class MLP(nn.Module):
    """
    A multilayer perceptron with ReLU activations and optional BatchNorm.
    """

    def __init__(self, sizes, heads=None, batch_norm=True, final_act=None):
        super(MLP, self).__init__()
        self.heads = heads
        if heads is not None:
            sizes[-1] = sizes[-1] * heads

        layers = []
        for s in range(len(sizes) - 1):
            layers += [
                nn.Linear(sizes[s], sizes[s + 1]),
                nn.BatchNorm1d(sizes[s + 1])
                if batch_norm and s < len(sizes) - 2
                else None,
                nn.ReLU()
                if s < len(sizes) - 2
                else None
            ]
        if final_act is None:
            pass
        elif final_act == "relu":
            layers += [nn.ReLU()]
        elif final_act == "sigmoid":
            layers += [nn.Sigmoid()]
        elif final_act == "softmax":
            layers += [nn.Softmax(dim=-1)]
        else:
            raise ValueError("final_act not recognized")

        layers = [l for l in layers if l is not None]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        out = self.network(x)

        if self.heads is not None:
            out = out.view(*out.shape[:-1], -1, self.heads)
        return out


# HALOVAER model
class HALOMASKVAE_ALN(BaseModuleClass):
    """
    Variational auto-encoder model.

    This is an implementation of the scVI model described in [Lopez18]_

    Parameters
    ----------
    n_input
        Number of input genes
    n_batch
        Number of batches, if 0, no batch correction is performed.
    n_labels
        Number of labels
    n_hidden
        Number of nodes per hidden layer
    n_latent
        Dimensionality of the latent space
    n_layers
        Number of hidden layers used for encoder and decoder NNs
    n_continuous_cov
        Number of continuous covarites
    n_cats_per_cov
        Number of categories for each extra categorical covariate
    dropout_rate
        Dropout rate for neural networks
    dispersion
        One of the following

        * ``'gene'`` - dispersion parameter of NB is constant per gene across cells
        * ``'gene-batch'`` - dispersion can differ between different batches
        * ``'gene-label'`` - dispersion can differ between different labels
        * ``'gene-cell'`` - dispersion can differ for every gene in every cell
    log_variational
        Log(data+1) prior to encoding for numerical stability. Not normalization.
    gene_likelihood
        One of

        * ``'nb'`` - Negative binomial distribution
        * ``'zinb'`` - Zero-inflated negative binomial distribution
        * ``'poisson'`` - Poisson distribution
    latent_distribution
        One of

        * ``'normal'`` - Isotropic normal
        * ``'ln'`` - Logistic normal with normal params N(0, 1)
    encode_covariates
        Whether to concatenate covariates to expression in encoder
    deeply_inject_covariates
        Whether to concatenate covariates into output of hidden layers in encoder/decoder. This option
        only applies when `n_layers` > 1. The covariates are concatenated to the input of subsequent hidden layers.
    use_layer_norm
        Whether to use layer norm in layers
    use_size_factor_key
        Use size_factor AnnDataField defined by the user as scaling factor in mean of conditional distribution.
        Takes priority over `use_observed_lib_size`.
    use_observed_lib_size
        Use observed library size for RNA as scaling factor in mean of conditional distribution
    library_log_means
        1 x n_batch array of means of the log library sizes. Parameterizes prior on library size if
        not using observed library size.
    library_log_vars
        1 x n_batch array of variances of the log library sizes. Parameterizes prior on library size if
        not using observed library size.
    var_activation
        Callable used to ensure positivity of the variational distributions' variance.
        When `None`, defaults to `torch.exp`.
    """

    def __init__(
        self,
        n_input_genes: int,
        n_input_regions:int,
        n_batch: int = 0,
        n_labels: int = 0,
        n_hidden: int = 128,
        n_latent: int = 10,
        ## coupled latent variables number
        n_latent_dep: int = 5 , 
        n_layers: int = 1,
        ####
        n_layers_encoder: int = 2,
        n_layers_decoder: int = 2,
        ####
        n_continuous_cov: int = 0,
        n_cats_per_cov: Optional[Iterable[int]] = None,
        dropout_rate: float = 0.1,
        region_factors: bool = True,
        dispersion: str = "gene",
        log_variational: bool = True,
        gene_likelihood: Literal["zinb", "nb", "poisson"] = "zinb",
        latent_distribution: str = "normal",
        encode_covariates: bool = True,
        deeply_inject_covariates: bool = True,
        use_batch_norm: Literal["encoder", "decoder", "none", "both"] = "both",
        use_layer_norm: Literal["encoder", "decoder", "none", "both"] = "none",
        use_size_factor_key: bool = False,
        use_observed_lib_size: bool = True,

        library_log_means: Optional[np.ndarray] = None,
        library_log_vars: Optional[np.ndarray] = None,
        var_activation: Optional[Callable] = None,
        expr_train: Optional[bool] = True,
        acc_train: Optional[bool] = False,
        finetune: Optional[int] = 0,
        alpha: Optional[float] = 0.002,
        beta1: Optional[float] = 1e6,
        beta2: Optional[float] = 5e5,
        beta3: Optional[float] = 1e8,
        gates_finetune:Optional[bool] = False


    ):
        super().__init__()
    
        self.n_input_regions = n_input_regions
        self.region_factors = None
        if region_factors:
            self.region_factors = torch.nn.Parameter(torch.zeros(self.n_input_regions))        

        self.n_input_genes = n_input_genes
        self.dispersion = dispersion
        self.n_latent = n_latent
        self.n_latent_dep = n_latent_dep
        self.n_latent_indep = n_latent - n_latent_dep
        self.log_variational = log_variational
        self.gene_likelihood = gene_likelihood
        # Automatically deactivate if useless
        self.n_batch = n_batch
        self.n_labels = n_labels
  
        self.alpha = alpha

        self.latent_distribution = latent_distribution
        self.encode_covariates = encode_covariates
        self.gates_finetune = gates_finetune

        ##### add hidden_common numbers
        self.n_hidden_common = (
            int(np.sqrt(self.n_input_regions + self.n_input_genes))
            if n_hidden is None
            else n_hidden
        )
        #### add latent common numbers
        self.n_latent_common = int(np.sqrt(self.n_hidden)) if n_latent is None else n_latent
        #### add region encoder, decoder
        self.n_layers_encoder = n_layers_encoder
        self.n_layers_decoder = n_layers_decoder

        self.use_size_factor_key = use_size_factor_key
        self.use_observed_lib_size = use_size_factor_key or use_observed_lib_size

        self.expr_train = expr_train
        self.acc_train = acc_train
        self.both_train = False
        if self.expr_train and self.acc_train:
            self.both = True
        self.finetune = gates_finetune    
        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.beta3 = beta3

        if not self.use_observed_lib_size:
            if library_log_means is None or library_log_means is None:
                raise ValueError(
                    "If not using observed_lib_size, "
                    "must provide library_log_means and library_log_vars."
                )

            self.register_buffer(
                "library_log_means", torch.from_numpy(library_log_means).float()
            )
            self.register_buffer(
                "library_log_vars", torch.from_numpy(library_log_vars).float()
            )

        if self.dispersion == "gene":
            self.px_r = torch.nn.Parameter(torch.randn(n_input_genes))
        elif self.dispersion == "gene-batch":
            self.px_r = torch.nn.Parameter(torch.randn(n_input_genes, n_batch))
        elif self.dispersion == "gene-label":
            self.px_r = torch.nn.Parameter(torch.randn(n_input_genes, n_labels))
        elif self.dispersion == "gene-cell":
            pass
        else:
            raise ValueError(
                "dispersion must be one of ['gene', 'gene-batch',"
                " 'gene-label', 'gene-cell'], but input was "
                "{}.format(self.dispersion)"
            )

        use_batch_norm_encoder = use_batch_norm == "encoder" or use_batch_norm == "both"
        use_batch_norm_decoder = use_batch_norm == "decoder" or use_batch_norm == "both"
        use_layer_norm_encoder = use_layer_norm == "encoder" or use_layer_norm == "both"
        use_layer_norm_decoder = use_layer_norm == "decoder" or use_layer_norm == "both"
        self.use_batch_norm_linear = use_batch_norm_decoder
        # z encoder goes from the n_input-dimensional data to an n_latent-d
        # latent space representation
        n_input_encoder = n_input_genes + n_continuous_cov * encode_covariates
        cat_list = [n_batch] + list([] if n_cats_per_cov is None else n_cats_per_cov)
        encoder_cat_list = cat_list if encode_covariates else None
        # print("n_input_encoder: {}".format(n_input_encoder))
        self.z_encoder = Encoder(
            n_input_encoder,
            n_latent,
            n_cat_list=encoder_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            distribution=latent_distribution,
            inject_covariates=deeply_inject_covariates,
            use_batch_norm=use_batch_norm_encoder,
            use_layer_norm=use_layer_norm_encoder,
            var_activation=var_activation,
            return_dist=True,
        )
        ## set up input encoder:
        n_input_encoder_acc = (
            self.n_input_regions + n_continuous_cov * encode_covariates
        )
        self.z_encoder_accessibility = Encoder(
            n_input=n_input_encoder_acc,
            n_layers=self.n_layers_encoder,
            n_output=self.n_latent,
            n_hidden=self.n_hidden_common,
            n_cat_list=encoder_cat_list,
            dropout_rate=dropout_rate,
            activation_fn=torch.nn.LeakyReLU,
            distribution=self.latent_distribution,
            var_eps=0,
            use_batch_norm=use_batch_norm_encoder,
            use_layer_norm=use_layer_norm_encoder,
            return_dist=True,
        )
   
        self.z_decoder_accessibility = GateDecoder(
            n_input=self.n_latent + n_continuous_cov,
            n_output=n_input_regions,
            n_hidden_global=self.n_hidden_common,
            n_cat_list=cat_list,
            n_layers=self.n_layers_decoder,
            use_batch_norm=use_batch_norm_decoder,
            use_layer_norm=use_layer_norm_decoder,
            deep_inject_covariates=deeply_inject_covariates,
            fine_tune = self.gates_finetune
        )

        
        # l encoder goes from n_input-dimensional data to 1-d library size
        self.l_encoder = Encoder(
            n_input_encoder,
            1,
            n_layers=1,
            n_cat_list=encoder_cat_list,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            inject_covariates=deeply_inject_covariates,
            use_batch_norm=use_batch_norm_encoder,
            use_layer_norm=use_layer_norm_encoder,
            var_activation=var_activation,
            return_dist=True,
        )
        ### ATAC library size encoder
        ### ATAC library size encoder

        self.l_encoder_accessibility = DecoderPeakVI(
            n_input=n_input_encoder_acc,
            n_output=1,
            n_hidden=self.n_hidden_common,
            n_cat_list=encoder_cat_list,
            n_layers=self.n_layers_encoder,
            use_batch_norm=use_batch_norm_encoder,
            use_layer_norm=use_layer_norm_encoder,
            deep_inject_covariates=deeply_inject_covariates,
        )

        # decoder goes from n_latent-dimensional space to n_input-d data
        n_input_decoder = n_latent + n_continuous_cov
        # self.decoder = DecoderSCVI(
        #     n_input_decoder,
        #     n_input_genes,
        #     n_cat_list=cat_list,
        #     n_layers=n_layers,
        #     n_hidden=n_hidden,
        #     inject_covariates=deeply_inject_covariates,
        #     use_batch_norm=use_batch_norm_decoder,
        #     use_layer_norm=use_layer_norm_decoder,
        #     scale_activation="softplus" if use_size_factor_key else "softmax",
        # )

        self.decoder = LinearDecoderSCVI(
            n_input_decoder,
            n_input_genes,
            n_cat_list=cat_list,
            use_batch_norm=use_batch_norm_decoder,
            bias=True,
        )
        
        self.decouple_aligner = MLP([self.n_latent_indep, 50, 100, 100, self.n_latent_indep],final_act="sigmoid")
        self.couple_aligner = MLP([self.n_latent_dep, 50, 100, 100, self.n_latent_dep], final_act="sigmoid")
        

    @torch.no_grad()
    def get_loadings(self) -> np.ndarray:
        """Extract per-gene weights (for each Z, shape is genes by dim(Z)) in the linear decoder."""
        # This is BW, where B is diag(b) batch norm, W is weight matrix
        if self.use_batch_norm_linear is True:
            w = self.decoder.factor_regressor.fc_layers[0][0].weight
            bn = self.decoder.factor_regressor.fc_layers[0][1]
            sigma = torch.sqrt(bn.running_var + bn.eps)
            gamma = bn.weight
            b = gamma / sigma
            bI = torch.diag(b)
            loadings = torch.matmul(bI, w)
        else:
            loadings = self.decoder.factor_regressor.fc_layers[0][0].weight
        loadings = loadings.detach().cpu().numpy()
        if self.n_batch > 1:
            loadings = loadings[:, : -self.n_batch]

        return loadings    

    def _get_inference_input(self, tensors):
        x = tensors[REGISTRY_KEYS.X_KEY]
        batch_index = tensors[REGISTRY_KEYS.BATCH_KEY]
        time_index = tensors[REGISTRY_KEYS.TIME_KEY]


        cont_key = REGISTRY_KEYS.CONT_COVS_KEY
        cont_covs = tensors[cont_key] if cont_key in tensors.keys() else None

        cat_key = REGISTRY_KEYS.CAT_COVS_KEY
        cat_covs = tensors[cat_key] if cat_key in tensors.keys() else None

        input_dict = dict(
            x=x, batch_index=batch_index, cont_covs=cont_covs, cat_covs=cat_covs, time_index = time_index
        )
        return input_dict

    def _get_generative_input(self, tensors, inference_outputs):
        """
        RNA generative input
        
        """
        # if self.gates_finetune:
        #     with torch.no_grad():
        #         z = inference_outputs["z"]
        #         library = inference_outputs["library"]
        #         batch_index = tensors[REGISTRY_KEYS.BATCH_KEY]
        #         y = tensors[REGISTRY_KEYS.LABELS_KEY]
                

        #         cont_key = REGISTRY_KEYS.CONT_COVS_KEY
        #         cont_covs = tensors[cont_key] if cont_key in tensors.keys() else None

        #         cat_key = REGISTRY_KEYS.CAT_COVS_KEY
        #         cat_covs = tensors[cat_key] if cat_key in tensors.keys() else None


        #         size_factor_key = REGISTRY_KEYS.SIZE_FACTOR_KEY
        #         size_factor = (
        #             torch.log(tensors[size_factor_key])
        #             if size_factor_key in tensors.keys()
        #             else None
        #         )

        #         """
                
        #         ATAC genreative input
                
                
        #         """
        #         z_acc = inference_outputs["z_acc"]

        #         qz_acc = inference_outputs["qz_acc"]

        #         libsize_acc = inference_outputs["libsize_acc"]


        # else:

        z = inference_outputs["z"]
        library = inference_outputs["library"]
        batch_index = tensors[REGISTRY_KEYS.BATCH_KEY]
        y = tensors[REGISTRY_KEYS.LABELS_KEY]
        

        cont_key = REGISTRY_KEYS.CONT_COVS_KEY
        cont_covs = tensors[cont_key] if cont_key in tensors.keys() else None

        cat_key = REGISTRY_KEYS.CAT_COVS_KEY
        cat_covs = tensors[cat_key] if cat_key in tensors.keys() else None


        size_factor_key = REGISTRY_KEYS.SIZE_FACTOR_KEY
        size_factor = (
            torch.log(tensors[size_factor_key])
            if size_factor_key in tensors.keys()
            else None
        )


        """
        
        ATAC genreative input
        
        
        """
        z_acc = inference_outputs["z_acc"]

        qz_acc = inference_outputs["qz_acc"]

        libsize_acc = inference_outputs["libsize_acc"]

        input_dict = dict(
            z=z,
            library=library,
            batch_index=batch_index,
            y=y,
            cont_covs=cont_covs,
            cat_covs=cat_covs,
            size_factor=size_factor,
            z_acc =z_acc, 
            qz_acc = qz_acc,
            libsize_acc = libsize_acc
        )
        return input_dict

    def _compute_local_library_params(self, batch_index):
        """
        Computes local library parameters.

        Compute two tensors of shape (batch_index.shape[0], 1) where each
        element corresponds to the mean and variances, respectively, of the
        log library sizes in the batch the cell corresponds to.
        """
        n_batch = self.library_log_means.shape[1]
        local_library_log_means = F.linear(
            one_hot(batch_index, n_batch), self.library_log_means
        )
        local_library_log_vars = F.linear(
            one_hot(batch_index, n_batch), self.library_log_vars
        )
        return local_library_log_means, local_library_log_vars

    @auto_move_data
    def inference(self, x, batch_index, cont_covs=None, cat_covs=None, time_index=None, n_samples=1):
        """
        High level inference method.

        Runs the inference (encoder) model.
        """
        x_rna = x[:, :self.n_input_genes]
        x_ = x_rna
        x_chr = x[:, self.n_input_genes :]

        """
        
        RNA inference part
        
        """     
        if self.use_observed_lib_size:
            library = torch.log(x_rna.sum(1)).unsqueeze(1)
        if self.log_variational:
            x_ = torch.log(1 + x_)
        # print("x_ shape {}".format(x_.shape))
        if cont_covs is not None and self.encode_covariates:
            encoder_input = torch.cat((x_, cont_covs), dim=-1)
        else:
            encoder_input = x_
        if cat_covs is not None and self.encode_covariates:
            categorical_input = torch.split(cat_covs, 1, dim=1)
        else:
            categorical_input = tuple()
        # print("encoder shape {}".format(encoder_input.shape)) 
   
        qz, z = self.z_encoder(encoder_input, batch_index, *categorical_input)
        ql = None

        if not self.use_observed_lib_size:
            ql, library_encoded = self.l_encoder(
                encoder_input, batch_index, *categorical_input
            )
            library = library_encoded

        if n_samples > 1:
            untran_z = qz.sample((n_samples,))
            z = self.z_encoder.z_transformation(untran_z)
            if self.use_observed_lib_size:
                library = library.unsqueeze(0).expand(
                    (n_samples, library.size(0), library.size(1))
                )
            else:
                library = ql.sample((n_samples,))
        """
        
        ATAC inference part
        
        """

        if cont_covs is not None and self.encode_covariates:
            encoder_input_accessibility = torch.cat((x_chr, cont_covs), dim=-1)
        else:
            encoder_input_accessibility = x_chr


        if cat_covs is not None and self.encode_covariates:
            categorical_input = torch.split(cat_covs, 1, dim=1)
        else:
            categorical_input = tuple()

        libsize_acc = self.l_encoder_accessibility(
            encoder_input_accessibility, batch_index, *categorical_input
        )

        qz_acc, z_acc = self.z_encoder_accessibility(
            encoder_input_accessibility, batch_index, *categorical_input
        )

        if n_samples > 1:
            untran_za = qz_acc.sample((n_samples,))
            z_acc = self.z_encoder_accessibility.z_transformation(untran_za)
            libsize_acc = libsize_acc.unsqueeze(0).expand(
                (n_samples, libsize_acc.size(0), libsize_acc.size(1))
            )

        qzm_acc = qz_acc.loc
        qzm_expr = qz.loc
        qzv_acc = qz_acc.scale**2
        qzv_expr = qz.scale**2
        ## define coupled components
        qzm_acc_dep = qzm_acc[:, :self.n_latent_dep]
        qzm_expr_dep = qzm_expr[:, :self.n_latent_dep]
        qzv_acc_dep = qzv_acc[:, :self.n_latent_dep]
        qzv_expr_dep = qzv_expr[:, :self.n_latent_dep]

        ## define decoupled components
        qzm_acc_indep = qzm_acc[:, self.n_latent_dep:]
        qzm_expr_indep = qzm_expr[:, self.n_latent_dep:]
        qzv_acc_indep = qzv_acc[:, self.n_latent_dep:]
        qzv_expr_indep = qzv_expr[:, self.n_latent_dep:]
        
        ## build the alignment and predicting from latent atac to latent gene
        
        pred_expr_dep_m =  self.couple_aligner(qzm_acc_dep)
        pred_expr_indep_m =  self.decouple_aligner(qzm_acc_indep)
        
        z_expr_dep = z[:, :self.n_latent_dep]
        z_expr_indep = z[:, self.n_latent_dep:]

        z_acc_dep = z_acc[:, :self.n_latent_dep]
        z_acc_indep = z_acc[:, self.n_latent_dep:]
        
        ## predict gene expression latent space from ATAC


        outputs = dict(z=z, 
        qz=qz, 
        ql=ql, 
        library=library,
        qz_acc = qz_acc,
        z_acc = z_acc,
        libsize_acc=libsize_acc,

        ### add coupled and decoupled components

        z_expr_dep=z_expr_dep,
        z_acc_dep=z_acc_dep,
        
        qzm_expr_dep=qzm_expr_dep,
        qzv_expr_dep=qzv_expr_dep,
        qzm_acc_dep=qzm_acc_dep,
        qzv_acc_dep=qzv_acc_dep,


        ## lagging part 
        z_expr_indep=z_expr_indep,
        z_acc_indep=z_acc_indep,
        qzm_expr_indep=qzm_expr_indep,
        qzv_expr_indep=qzv_expr_indep,
        qzm_acc_indep=qzm_acc_indep,
        qzv_acc_indep=qzv_acc_indep,
        
        ## aligned part
        pred_expr_dep_m =  pred_expr_dep_m,
        pred_expr_indep_m =  pred_expr_indep_m,
                       
        ## time                            
        time_key = time_index

        )
        return outputs

    @auto_move_data
    def generative(
        self,
        z,
        library,
        batch_index,

        z_acc, 
        qz_acc,
        libsize_acc,

        cont_covs=None,
        cat_covs=None,
        size_factor=None,
        y=None,
        transform_batch=None,
    ):
        """Runs the generative model."""
        # TODO: refactor forward function to not rely on y
        # Likelihood distribution
        if cont_covs is None:
            decoder_input = z
        elif z.dim() != cont_covs.dim():
            decoder_input = torch.cat(
                [z, cont_covs.unsqueeze(0).expand(z.size(0), -1, -1)], dim=-1
            )
        else:
            decoder_input = torch.cat([z, cont_covs], dim=-1)

        if cat_covs is not None:
            categorical_input = torch.split(cat_covs, 1, dim=1)
        else:
            categorical_input = tuple()

        if transform_batch is not None:
            batch_index = torch.ones_like(batch_index) * transform_batch

        if not self.use_size_factor_key:
            size_factor = library

        px_scale, px_r, px_rate, px_dropout = self.decoder(
            self.dispersion,
            decoder_input,
            size_factor,
            batch_index,
            *categorical_input,
            y,
        )
        if self.dispersion == "gene-label":
            px_r = F.linear(
                one_hot(y, self.n_labels), self.px_r
            )  # px_r gets transposed - last dimension is nb genes
        elif self.dispersion == "gene-batch":
            px_r = F.linear(one_hot(batch_index, self.n_batch), self.px_r)
        elif self.dispersion == "gene":
            px_r = self.px_r

        px_r = torch.exp(px_r)

        if self.gene_likelihood == "zinb":
            px = ZeroInflatedNegativeBinomial(
                mu=px_rate,
                theta=px_r,
                zi_logits=px_dropout,
                scale=px_scale,
            )
        elif self.gene_likelihood == "nb":
            px = NegativeBinomial(mu=px_rate, theta=px_r, scale=px_scale)
        elif self.gene_likelihood == "poisson":
            px = Poisson(px_rate, scale=px_scale)

        # Priors
        if self.use_observed_lib_size:
            pl = None
        else:
            (
                local_library_log_means,
                local_library_log_vars,
            ) = self._compute_local_library_params(batch_index)
            pl = Normal(local_library_log_means, local_library_log_vars.sqrt())
        pz = Normal(torch.zeros_like(z), torch.ones_like(z))

        """
        Runs the generative model for ATAC data
        
        """

        if cat_covs is not None:
            categorical_input = torch.split(cat_covs, 1, dim=1)
        else:
            categorical_input = tuple()

        latent_acc = z_acc 
        
        if cont_covs is None:
            decoder_input_acc = latent_acc

        elif latent_acc.dim() != cont_covs.dim():
            

            decoder_input_acc = torch.cat(
                [latent_acc, cont_covs.unsqueeze(0).expand(latent_acc.size(0), -1, -1)], dim=-1
            )

        else:
            decoder_input_acc = torch.cat([decoder_input_acc, cont_covs], dim=-1)

        # Accessibility Decoder
        pa = self.z_decoder_accessibility(decoder_input_acc, batch_index, *categorical_input)

        return dict(
            px=px,
            pl=pl,
            pz=pz,
            pa = pa, 
            px_scale=px_scale,
            px_rate = px_rate,
            px_r = px_r
        )

    def set_gates_finetune(self, gatesfine):
        self.gates_finetune = gatesfine
        ## set the gates finetune parameters
        self.z_decoder_accessibility.set_finetune(gatesfine)

    def set_train_params(self, expr_train, acc_train):

        self.expr_train = expr_train
        self.acc_train = acc_train
        self.both_train = False
        if self.expr_train and self.acc_train:
            self.both = True

    def set_finetune_params(self, finetune):
        """
        0 : no fine tune, focus on reconstruction loss
        1: only fine tune, focus on the delta causal values
        2: fine and reconst loss both are executed
        """
        self.finetune = finetune

    def set_scale_params(self, beta1, beta2, beta3, alpha=0.02):
        self.beta_1 = beta1
        self.beta_2 = beta2
        self.beta_3 = beta3
        self.alpha = alpha

    def get_scale_params(self):
        return [self.beta1, self.beta2, self.beta3, self.alpha]

        
    def _get_dict_sum(self, dictionary):
        total = 0.0
        for value in dictionary.values():
            total += value
        return total

    def loss(
        self,
        tensors,
        inference_outputs,
        generative_outputs,
        kl_weight: float = 1.0,
    ):
        xwhole = tensors[REGISTRY_KEYS.X_KEY]
        x = xwhole[:, :self.n_input_genes]
        x_accessibility = xwhole[:, self.n_input_genes:]
        """
        compute ATAC loss
        
        """
        pa = generative_outputs["pa"]    
        z_acc =inference_outputs['z_acc']
        qz_acc = inference_outputs['qz_acc']
        libsize_acc = inference_outputs['libsize_acc']

        qzm_acc = qz_acc.loc
        qzv_acc = qz_acc.scale**2
        
        kl_divergence_acc = kld(
            Normal(qzm_acc, torch.sqrt(qzv_acc)), Normal(0, 1)
        ).sum(dim=1)
        
        
        # print("x_accessibility shape {}, pa shape {}, libsize_acc {}".format\
        # (x_accessibility.shape, pa.shape, libsize_acc.shape))
        rl_accessibility = self.get_reconstruction_loss_accessibility(
            x_accessibility, pa, libsize_acc
        )
        # print("atac recon shape {}".format(rl_accessibility.shape))
        """
        compute the RNA loss
        
        """
        kl_divergence_z = kl(inference_outputs["qz"], generative_outputs["pz"]).sum(
            dim=1
        )
        if not self.use_observed_lib_size:
            kl_divergence_l = kl(
                inference_outputs["ql"],
                generative_outputs["pl"],
            ).sum(dim=1)
        else:
            kl_divergence_l = 0.0
    
        reconst_loss = 0
        weighted_kl_local = 0

        ## get the aligned part
        qzm_expr_dep=inference_outputs['qzm_expr_dep']
        qzm_expr_indep=inference_outputs['qzm_expr_indep']
        pred_expr_dep_m =  inference_outputs['pred_expr_dep_m']
        pred_expr_indep_m =  inference_outputs['pred_expr_indep_m']
        mse_loss = nn.MSELoss()
        aligned_loss_dep = mse_loss(pred_expr_dep_m, qzm_expr_dep)
        aligned_loss_indep = mse_loss(pred_expr_indep_m, qzm_expr_indep)



        if self.expr_train and not self.acc_train:
        ### RNA only
            reconst_loss = -generative_outputs["px"].log_prob(x).sum(-1)
            # print("RNA reconst_losss shape {}".format(reconst_loss.shape))
            kl_local_for_warmup = kl_divergence_z
            kl_local_no_warmup = kl_divergence_l
            weighted_kl_local = kl_weight * kl_local_for_warmup + kl_local_no_warmup

        elif  not self.expr_train and  self.acc_train:

            reconst_loss = rl_accessibility
            # print("ATAC reconst_losss shape {}".format(reconst_loss.shape))
            kl_local_for_warmup = kl_divergence_acc
            weighted_kl_local = kl_weight * kl_local_for_warmup
            # print("the output layer grad without sparsity is {}".format(self.z_decoder_accessibility.output[0].weights.grad))

            if self.gates_finetune == True:
                sparsity_regu = 1e-4 * self.z_decoder_accessibility.get_gate_regu(z_acc)
                # print("original reconst_loss {}, sparsity loss: {}".format(reconst_loss, sparsity_regu))
                reconst_loss += sparsity_regu
            # weighted_kl_local += sparsity_regu
            # print("the output layer grad is {}".format(self.z_decoder_accessibility.output[0].weights.grad))
            # print("the gates layer grad is {}".format(self.z_decoder_accessibility.gate_layer.FC.fc_layers[0].weights.grad))


        elif  self.expr_train and  self.acc_train:
            reconst_loss = -(self.n_input_regions +0.0)/(self.n_input_genes)*generative_outputs["px"].log_prob(x).sum(-1) + rl_accessibility
            kl_local_for_warmup = kl_divergence_z + kl_divergence_acc
            kl_local_no_warmup = kl_divergence_l
            weighted_kl_local = kl_weight * kl_local_for_warmup + kl_local_no_warmup
            # if self.gates_finetune == True:
                # sparsity_regu = 1e-4 * self.z_decoder_accessibility.get_gate_regu(z_acc)
                # # print("original reconst_loss {}, sparsity loss: {}".format(reconst_loss, sparsity_regu))
                # reconst_loss += sparsity_regu
            # sparsity_regu = 0.1 * self.z_decoder_accessibility.get_gate_regu()
            # reconst_loss += sparsity_regu
            # weighted_kl_local += sparsity_regu
        
        if self.finetune == 1:
            z_expr_dep=inference_outputs['z_expr_dep']       
            z_acc_dep=inference_outputs['z_acc_dep']

            qzm_expr_dep=inference_outputs['qzm_expr_dep']
            qzv_expr_dep=inference_outputs['qzv_expr_dep']
            qzm_acc_dep=inference_outputs['qzm_acc_dep']
            qzv_acc_dep=inference_outputs['qzv_acc_dep']
            
            
            qzm_expr_indep=inference_outputs['qzm_expr_indep']
            qzv_expr_indep=inference_outputs['qzv_expr_indep']
            qzm_acc_indep=inference_outputs['qzm_acc_indep']
            qzv_acc_indep=inference_outputs['qzv_acc_indep']

            # print("loss qzv_acc_dep type: {}".format(type(qzv_acc_dep)) )       
            ## lagging part 
            z_expr_indep=inference_outputs['z_expr_indep']
            z_acc_indep= inference_outputs['z_acc_indep']

            time = inference_outputs['time_key']

            kld_paired = kld(
            Normal(qzm_expr_dep, qzv_expr_dep.sqrt()), Normal(qzm_acc_dep, qzv_acc_dep.sqrt())) + kld(
            Normal(qzm_acc_dep, qzv_acc_dep.sqrt()), Normal(qzm_expr_dep, qzv_expr_dep.sqrt()))
            kld_paired = kld_paired.sum(dim=1)
            
            ## use sampled
            # a2rscore_coupled, _, _ = torch_infer_nonsta_dir(z_acc_dep, z_expr_dep, time)
            # r2ascore_coupled, _, _ = torch_infer_nonsta_dir(z_expr_dep, z_acc_dep, time)
            
            ## use mean
            a2rscore_coupled, _, _ = torch_infer_nonsta_dir(qzm_acc_dep, qzm_expr_dep, time)
            r2ascore_coupled, _, _ = torch_infer_nonsta_dir(qzm_expr_dep, qzm_acc_dep, time)
            
            # self.alpha=0.002
            a2rscore_coupled_loss = torch.maximum(self.alpha - a2rscore_coupled + 1e-3, torch.tensor(0))
            r2ascore_coupled_loss = torch.maximum(self.alpha - r2ascore_coupled + 1e-3, torch.tensor(0))

            a2rscore_coupled_loss = torch.maximum(self.alpha - a2rscore_coupled , torch.tensor(0))
            r2ascore_coupled_loss = torch.maximum(self.alpha - r2ascore_coupled, torch.tensor(0))
            
            ## we could use the mean to estimate the score
            ## used sampled
#             a2rscore_lagging, _, _ = torch_infer_nonsta_dir(z_acc_indep, z_expr_indep, time)
#             r2ascore_lagging, _, _ = torch_infer_nonsta_dir(z_expr_indep, z_acc_indep, time)
            
            ## use mean
            a2rscore_lagging, _, _ = torch_infer_nonsta_dir(qzm_acc_indep, qzm_expr_indep, time)
            r2ascore_lagging, _, _ = torch_infer_nonsta_dir(qzm_expr_indep, qzm_acc_indep, time)
            

            # a2r_r2a_score_loss =  torch.maximum(a2rscore_lagging-r2ascore_lagging+1e-4, torch.tensor(0))
            a2r_r2a_score_loss =  torch.maximum(a2rscore_lagging-r2ascore_lagging, torch.tensor(0))
            a2rscore_lagging = torch.maximum(-self.alpha + a2rscore_lagging, torch.tensor(0))
            r2ascore_decoupled_loss = torch.maximum(self.alpha - r2ascore_lagging, torch.tensor(0))


            # print(a2rscore_coupled, r2ascore_coupled, a2rscore_lagging, r2ascore_lagging)

            # self.beta_2 = 5e5
            # self.beta_3 = 1e8
            # self.beta_1 = 1e6

            nod_loss =   self.beta_1 *  a2rscore_lagging.to(torch.float64)  + self.beta_3 * a2r_r2a_score_loss + \
            self.beta_2 * a2rscore_coupled_loss + self.beta_2 * a2rscore_coupled_loss + self.beta_2*r2ascore_coupled_loss \
                + self.beta1 * r2ascore_decoupled_loss
            reconst_loss = nod_loss * torch.ones_like(reconst_loss)
            # print("kld_paird loss {}, kld_divergence_acc {}, kld_paired {}"\
            #     .format(kl_divergence_z.shape, kl_divergence_acc.shape, kld_paired.shape))
            kl_local_for_warmup = kl_divergence_z + kl_divergence_acc + kld_paired
            weighted_kl_local = kl_weight * kl_local_for_warmup + kl_local_no_warmup
            # print("nod_loss {}".format(nod_loss))

        
        elif   self.finetune == 2:  

            z_expr_dep=inference_outputs['z_expr_dep']       
            z_acc_dep=inference_outputs['z_acc_dep']

            qzm_expr_dep=inference_outputs['qzm_expr_dep']
            qzv_expr_dep=inference_outputs['qzv_expr_dep']
            qzm_acc_dep=inference_outputs['qzm_acc_dep']
            qzv_acc_dep=inference_outputs['qzv_acc_dep']

            # print("loss qzv_acc_dep type: {}".format(type(qzv_acc_dep)) )       
            ## lagging part 
            z_expr_indep=inference_outputs['z_expr_indep']
            z_acc_indep= inference_outputs['z_acc_indep']

            time = inference_outputs['time_key']
            # print("time : {}".format(time))

            # print("qzm_expr_dep {}, qzv_expr_dep {}, qzm_acc_dep {}, qzv_acc_dep {}".format(qzm_expr_dep, qzv_expr_dep, qzm_acc_dep, qzv_acc_dep))

            kld_paired = kld(
            Normal(qzm_expr_dep, qzv_expr_dep.sqrt()), Normal(qzm_acc_dep, qzv_acc_dep.sqrt())) + kld(
            Normal(qzm_acc_dep, qzv_acc_dep.sqrt()), Normal(qzm_expr_dep, qzv_expr_dep.sqrt()))
            kld_paired = kld_paired.sum(dim=1)


            a2rscore_coupled, _, _ = torch_infer_nonsta_dir(z_acc_dep, z_expr_dep, time)
            r2ascore_coupled, _, _ = torch_infer_nonsta_dir(z_expr_dep, z_acc_dep, time)
            # self.alpha=0.02

            a2rscore_coupled_loss = torch.maximum(self.alpha - a2rscore_coupled, torch.tensor(0))
            r2ascore_coupled_loss = torch.maximum(self.alpha - r2ascore_coupled, torch.tensor(0))
            a2rscore_lagging, _, _ = torch_infer_nonsta_dir(z_acc_indep, z_expr_indep, time)
            r2ascore_lagging, _, _ = torch_infer_nonsta_dir(z_expr_indep, z_acc_indep, time)
            a2r_r2a_score_loss =  torch.maximum(a2rscore_lagging-r2ascore_lagging+1e-4, torch.tensor(0)) 
            a2rscore_lagging = torch.maximum(-self.alpha + a2rscore_lagging, torch.tensor(0))
            r2ascore_lagging = torch.maximum(-self.alpha + r2ascore_lagging, torch.tensor(0))



            # a2rscore_coupled_loss = torch.exp(a2rscore_coupled_loss)
            # r2ascore_coupled_loss = torch.exp(r2ascore_coupled_loss)
            # a2rscore_lagging =  torch.exp(a2rscore_lagging)
            # r2ascore_lagging =  torch.exp(r2ascore_lagging)
            # a2r_r2a_score_loss =  torch.exp(r2ascore_lagging) 
            # a2rscore_lagging = torch.exp( a2rscore_lagging)
            # r2ascore_lagging = torch.exp(r2ascore_lagging)


            # self.beta_2 = 5e5
            # self.beta_3 = 1e8
            # self.beta_1 = 1e6


            nod_loss =   self.beta_1 *  a2rscore_lagging.to(torch.float64) + self.beta_1 * r2ascore_lagging.to(torch.float64)\
                  + self.beta_3 * a2r_r2a_score_loss + self.beta_2 * a2rscore_coupled_loss \
                     + self.beta_2 * a2rscore_coupled_loss + self.beta_2*r2ascore_coupled_loss
            # nod_loss_copy = nod_loss.copy()
            # nod_loss_copy  = nod_loss_copy.cpu()
            # reconst_loss_copy = reconst_loss.copy()

            # print("reconst_loss {}, nod_loss {}".format(reconst_loss, nod_loss))
            reconst_loss = reconst_loss + nod_loss * torch.ones_like(reconst_loss)
            if self.gates_finetune == True:
                sparsity_regu = 0.01 * self.z_decoder_accessibility.get_gate_regu(z_acc)
                # print("original reconst_loss {}, sparsity loss: {}".format(reconst_loss, sparsity_regu))
                reconst_loss += sparsity_regu
            kl_local_for_warmup = kl_divergence_z + kl_divergence_acc + kld_paired
            kl_local_no_warmup = kl_divergence_l
            weighted_kl_local = kl_weight * kl_local_for_warmup + kl_local_no_warmup
        
        loss = torch.mean(reconst_loss + weighted_kl_local + aligned_loss_dep + aligned_loss_indep)

        kl_local = dict(
            kl_divergence_l=kl_divergence_l, kl_divergence_z=kl_divergence_z
        )
        kl_global = torch.tensor(0.0)
        return LossRecorder(loss, reconst_loss, kl_local, kl_global)

    def get_reconstruction_loss_accessibility(self, x, p, d):
        f = torch.sigmoid(self.region_factors) if self.region_factors is not None else 1
        return torch.nn.BCELoss(reduction="none")(p * d * f, (x > 0).float()).sum(
            dim=-1
        )
    
    @torch.no_grad()
    def sample(
        self,
        tensors,
        n_samples=1,
        use_z_mean = False,
        library_size=1,
    ) -> np.ndarray:
        """
        Generate observation samples from the posterior predictive distribution.

        The posterior predictive distribution is written as :math:`p(\hat{x} \mid x)`.

        Parameters
        ----------
        tensors
            Tensors dict
        n_samples
            Number of required samples for each cell
        library_size
            Library size to scale scamples to

        Returns
        -------
        x_new : :py:class:`torch.Tensor`
            tensor with shape (n_cells, n_genes, n_samples)
        """
        inference_kwargs = dict(n_samples=n_samples)
        generative_kwargs=dict(use_z_mean=use_z_mean),
        _, generative_outputs, = self.forward(
            tensors,
            inference_kwargs=inference_kwargs,
            compute_loss=False,
            generative_kwargs=generative_kwargs

        )

        dist = generative_outputs["px"]
        if self.gene_likelihood == "poisson":
            l_train = generative_outputs["px"].mu
            l_train = torch.clamp(l_train, max=1e8)
            dist = torch.distributions.Poisson(
                l_train
            )  # Shape : (n_samples, n_cells_batch, n_genes)
        if n_samples > 1:
            exprs = dist.sample().permute(
                [1, 2, 0]
            )  # Shape : (n_cells_batch, n_genes, n_samples)
        else:
            exprs = dist.sample()

        return exprs.cpu()

    @torch.no_grad()
    @auto_move_data
    def marginal_ll(self, tensors, n_mc_samples):
        sample_batch = tensors[REGISTRY_KEYS.X_KEY]
        batch_index = tensors[REGISTRY_KEYS.BATCH_KEY]

        to_sum = torch.zeros(sample_batch.size()[0], n_mc_samples)

        for i in range(n_mc_samples):
            # Distribution parameters and sampled variables
            inference_outputs, _, losses = self.forward(tensors)
            qz = inference_outputs["qz"]
            ql = inference_outputs["ql"]
            z = inference_outputs["z"]
            library = inference_outputs["library"]

            # Reconstruction Loss
            reconst_loss = losses.reconstruction_loss

            # Log-probabilities
            p_z = (
                Normal(torch.zeros_like(qz.loc), torch.ones_like(qz.scale))
                .log_prob(z)
                .sum(dim=-1)
            )
            p_x_zl = -reconst_loss
            q_z_x = qz.log_prob(z).sum(dim=-1)
            log_prob_sum = p_z + p_x_zl - q_z_x

            if not self.use_observed_lib_size:
                (
                    local_library_log_means,
                    local_library_log_vars,
                ) = self._compute_local_library_params(batch_index)

                p_l = (
                    Normal(local_library_log_means, local_library_log_vars.sqrt())
                    .log_prob(library)
                    .sum(dim=-1)
                )
                q_l_x = ql.log_prob(library).sum(dim=-1)

                log_prob_sum += p_l - q_l_x

            to_sum[:, i] = log_prob_sum

        batch_log_lkl = logsumexp(to_sum, dim=-1) - np.log(n_mc_samples)
        log_lkl = torch.sum(batch_log_lkl).item()
        return log_lkl


class MASKVAE(VAE):
    """
    Linear-decoded Variational auto-encoder model.

    Implementation of [Svensson20]_.

    This model uses a linear decoder, directly mapping the latent representation
    to gene expression levels. It still uses a deep neural network to encode
    the latent representation.

    Compared to standard VAE, this model is less powerful, but can be used to
    inspect which genes contribute to variation in the dataset. It may also be used
    for all scVI tasks, like differential expression, batch correction, imputation, etc.
    However, batch correction may be less powerful as it assumes a linear model.

    Parameters
    ----------
    n_input
        Number of input genes
    n_batch
        Number of batches
    n_labels
        Number of labels
    n_hidden
        Number of nodes per hidden layer (for encoder)
    n_latent
        Dimensionality of the latent space
    n_layers_encoder
        Number of hidden layers used for encoder NNs
    dropout_rate
        Dropout rate for neural networks
    dispersion
        One of the following

        * ``'gene'`` - dispersion parameter of NB is constant per gene across cells
        * ``'gene-batch'`` - dispersion can differ between different batches
        * ``'gene-label'`` - dispersion can differ between different labels
        * ``'gene-cell'`` - dispersion can differ for every gene in every cell
    log_variational
        Log(data+1) prior to encoding for numerical stability. Not normalization.
    gene_likelihood
        One of

        * ``'nb'`` - Negative binomial distribution
        * ``'zinb'`` - Zero-inflated negative binomial distribution
        * ``'poisson'`` - Poisson distribution
    use_batch_norm
        Bool whether to use batch norm in decoder
    bias
        Bool whether to have bias term in linear decoder
    """

    def __init__(
        self,
        n_input: int,
        n_batch: int = 0,
        n_labels: int = 0,
        n_hidden: int = 128,
        n_latent: int = 10,
        n_layers_encoder: int = 1,
        dropout_rate: float = 0.1,
        dispersion: str = "gene",
        log_variational: bool = True,
        gene_likelihood: str = "nb",
        use_batch_norm: bool = True,
        bias: bool = False,
        latent_distribution: str = "normal",
        **vae_kwargs,
    ):
        super().__init__(
            n_input=n_input,
            n_batch=n_batch,
            n_labels=n_labels,
            n_hidden=n_hidden,
            n_latent=n_latent,
            n_layers=n_layers_encoder,
            dropout_rate=dropout_rate,
            dispersion=dispersion,
            log_variational=log_variational,
            gene_likelihood=gene_likelihood,
            latent_distribution=latent_distribution,
            use_observed_lib_size=False,
            **vae_kwargs,
        )
        self.use_batch_norm = use_batch_norm
        self.z_encoder = Encoder(
            n_input,
            n_latent,
            n_layers=n_layers_encoder,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            distribution=latent_distribution,
            use_batch_norm=True,
            use_layer_norm=False,
            return_dist=True,
        )
        self.l_encoder = Encoder(
            n_input,
            1,
            n_layers=1,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            use_batch_norm=True,
            use_layer_norm=False,
            return_dist=True,
        )
        self.decoder = LinearDecoderSCVI(
            n_latent,
            n_input,
            n_cat_list=[n_batch],
            use_batch_norm=use_batch_norm,
            use_layer_norm=False,
            bias=bias,
        )

    @torch.no_grad()
    def get_loadings(self) -> np.ndarray:
        """Extract per-gene weights (for each Z, shape is genes by dim(Z)) in the linear decoder."""
        # This is BW, where B is diag(b) batch norm, W is weight matrix
        if self.use_batch_norm is True:
            w = self.decoder.factor_regressor.fc_layers[0][0].weight
            bn = self.decoder.factor_regressor.fc_layers[0][1]
            sigma = torch.sqrt(bn.running_var + bn.eps)
            gamma = bn.weight
            b = gamma / sigma
            b_identity = torch.diag(b)
            loadings = torch.matmul(b_identity, w)
        else:
            loadings = self.decoder.factor_regressor.fc_layers[0][0].weight
        loadings = loadings.detach().cpu().numpy()
        if self.n_batch > 1:
            loadings = loadings[:, : -self.n_batch]

        return loadings
