from typing import Dict, Iterable, Optional

import numpy as np
import torch
from torch.distributions import Normal, Poisson
from torch.distributions import kl_divergence as kld

from scvi import REGISTRY_KEYS
from scvi._compat import Literal
from scvi.distributions import NegativeBinomial, ZeroInflatedNegativeBinomial
from scvi.module._peakvae import Decoder as DecoderPeakVI
from scvi.module.base import BaseModuleClass, LossRecorder, auto_move_data
from scvi.nn import DecoderSCVI, Encoder, FCLayers
from scvi.module import MULTIVAE

from anndata import AnnData
from typing import Dict, Iterable, List, Optional, Sequence, Union
from scvi.model import MULTIVI 


class MULTIVAE_Parallel(MULTIVAE):
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

        mask_expr = x_rna.sum(dim=1) > 0
        mask_acc = x_chr.sum(dim=1) > 0

        if cont_covs is not None and self.encode_covariates:
            encoder_input_expression = torch.cat((x_rna, cont_covs), dim=-1)
            encoder_input_accessibility = torch.cat((x_chr, cont_covs), dim=-1)
        else:
            encoder_input_expression = x_rna
            encoder_input_accessibility = x_chr

        if cat_covs is not None and self.encode_covariates:
            categorical_input = torch.split(cat_covs, 1, dim=1)
        else:
            categorical_input = tuple()

        # Z Encoders
        qz_acc, z_acc = self.z_encoder_accessibility(
            encoder_input_accessibility, batch_index, *categorical_input
        )
        qz_expr, z_expr = self.z_encoder_expression(
            encoder_input_expression, batch_index, *categorical_input
        )
        qzm_acc = qz_acc.loc
        qzm_expr = qz_expr.loc
        qzv_acc = qz_acc.scale**2
        qzv_expr = qz_expr.scale**2
        # L encoders
        libsize_expr = self.l_encoder_expression(
            encoder_input_expression, batch_index, *categorical_input
        )
        libsize_acc = self.l_encoder_accessibility(
            encoder_input_accessibility, batch_index, *categorical_input
        )

        # ReFormat Outputs
        if n_samples > 1:
            untran_za = qz_acc.sample((n_samples,))
            z_acc = self.z_encoder_accessibility.z_transformation(untran_za)
            untran_zr = qz_expr.sample((n_samples,))
            z_expr = self.z_encoder_expression.z_transformation(untran_zr)

            libsize_expr = libsize_expr.unsqueeze(0).expand(
                (n_samples, libsize_expr.size(0), libsize_expr.size(1))
            )
            libsize_acc = libsize_acc.unsqueeze(0).expand(
                (n_samples, libsize_acc.size(0), libsize_acc.size(1))
            )

        ## Sample from the average distribution
        # qzp_m = (qzm_acc + qzm_expr) / 2
        # qzp_v = (qzv_acc + qzv_expr) / (2**0.5)
        # zp = Normal(qzp_m, qzp_v.sqrt()).rsample()

        ## choose the correct latent representation based on the modality
        # qz_m = self._mix_modalities(qzp_m, qzm_expr, qzm_acc, mask_expr, mask_acc)
        # qz_v = self._mix_modalities(qzp_v, qzv_expr, qzv_acc, mask_expr, mask_acc)
        # z = self._mix_modalities(zp, z_expr, z_acc, mask_expr, mask_acc)

        outputs = dict(
            # z=z,
            # qz_m=qz_m,
            # qz_v=qz_v,
            z_expr=z_expr,
            qzm_expr=qzm_expr,
            qzv_expr=qzv_expr,
            z_acc=z_acc,
            qzm_acc=qzm_acc,
            qzv_acc=qzv_acc,
            libsize_expr=libsize_expr,
            libsize_acc=libsize_acc,
        )
        return outputs   


    ### change this function 
    def _get_generative_input(self, tensors, inference_outputs, transform_batch=None):
  
        z_expr = inference_outputs["z_expr"]
        qzm_expr = inference_outputs["qzm_expr"]
        libsize_expr = inference_outputs["libsize_expr"]


        z_acc = inference_outputs["z_acc"]
        qzm_acc = inference_outputs["qzm_acc"]
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
            z_expr=z_expr,
            qzm_expr=qzm_expr,
            
            z_acc=z_acc,
            qzm_acc=qzm_acc,

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

    def loss(
        self, tensors, inference_outputs, generative_outputs, kl_weight: float = 1.0
    ):
        # Get the data
        x = tensors[REGISTRY_KEYS.X_KEY]

        x_rna = x[:, : self.n_input_genes]
        x_chr = x[:, self.n_input_genes :]

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

        # Compute KLD between Z and N(0,I)
        # qz_m = inference_outputs["qz_m"]
        # qz_v = inference_outputs["qz_v"]
        # kl_div_z = kld(
        #     Normal(qz_m, torch.sqrt(qz_v)),
        #     Normal(0, 1),
        # ).sum(dim=1)

        # Compute KLD between distributions for paired data
        qzm_expr = inference_outputs["qzm_expr"]
        qzv_expr = inference_outputs["qzv_expr"]
        qzm_acc = inference_outputs["qzm_acc"]
        qzv_acc = inference_outputs["qzv_acc"]

        kl_div_z = kld(
            Normal(qzm_expr, torch.sqrt(qzv_expr)), Normal(0, 1)
        ) + kld(
            Normal(qzm_acc, torch.sqrt(qzv_acc)), Normal(0, 1)
        )
        # kld_paired = torch.where(
        #     torch.logical_and(mask_acc, mask_expr),
        #     kld_paired.T,
        #     torch.zeros_like(kld_paired).T,
        # ).sum(dim=0)

        # KL WARMUP
        kl_local_for_warmup = kl_div_z
        weighted_kl_local = kl_weight * kl_local_for_warmup

        # PENALTY
        # distance_penalty = kl_weight * torch.pow(z_acc - z_expr, 2).sum(dim=1)

        # TOTAL LOSS
        loss = torch.mean(recon_loss + weighted_kl_local)

        kl_local = dict(kl_divergence_z=kl_div_z)
        kl_global = torch.tensor(0.0)
        return LossRecorder(loss, recon_loss, kl_local, kl_global)

class MultiVI_Parallel(MULTIVI):

    def __init__(
        self,
        adata: AnnData,
        n_genes: int,
        n_regions: int,
        n_hidden: Optional[int] = None,
        n_latent: Optional[int] = None,
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
        **model_kwargs,
    ):
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

        keys = {"z": "z", "qz_m": "qz_m", "qz_v": "qz_v", "z_expr": "z_expr", 
        "qzm_expr": "qzm_expr", "qzv_expr": "qzv_expr", "z_acc": "z_acc", "qzm_acc": "qzm_acc", "qzv_acc": "qzv_acc"}
        

        adata = self._validate_anndata(adata)
        scdl = self._make_data_loader(
            adata=adata, indices=indices, batch_size=batch_size
        )
        latent = []
        latent_expr = []
        latent_atac = []
        for tensors in scdl:
            inference_inputs = self.module._get_inference_input(tensors)
            outputs = self.module.inference(**inference_inputs)
            qz_m = outputs[keys["qz_m"]]
            qz_v = outputs[keys["qz_v"]]
            z = outputs[keys["z"]]
            
            qzm_expr = outputs[keys["qzm_expr"]]
            qzv_expr = outputs[keys["qzv_expr"]]
            z_expr = outputs[keys["z_expr"]]

            qzm_acc = outputs[keys["qzm_acc"]]
            qzm_acc = outputs[keys["qzv_acc"]]
            z_acc = outputs[keys["z_acc"]]


            if give_mean:
                # does each model need to have this latent distribution param?
                if self.module.latent_distribution == "ln":
                    samples = Normal(qz_m, qz_v.sqrt()).sample([1])
                    z = torch.nn.functional.softmax(samples, dim=-1)
                    z = z.mean(dim=0)

                    samples_expr = Normal(qzm_expr, qzv_expr.sqrt()).sample([1])
                    z_expr = torch.nn.functional.softmax(samples_expr, dim=-1)
                    z_expr = z_expr.mean(dim=0)

                    samples_atac = Normal(qzm_acc, qzm_acc.sqrt()).sample([1])
                    z_acc = torch.nn.functional.softmax(samples_atac, dim=-1)
                    z_acc = z_acc.mean(dim=0)

                else:
                    z = qz_m
                    z_acc = qzm_acc
                    z_expr = qzm_expr

            
            latent += [z.cpu()]
            latent_atac+= [z_acc.cpu()]
            latent_expr += [z_expr.cpu()]

        return torch.cat(latent).numpy() , torch.cat(latent_atac).numpy(), torch.cat(latent_expr).numpy()