from typing import Dict, Iterable, Optional

import numpy as np
import torch
from torch.distributions import Normal, kl_divergence

from scvi import REGISTRY_KEYS
from scvi._compat import Literal
from scvi.module.base import BaseModuleClass, LossRecorder, auto_move_data
from scvi.nn import Encoder, FCLayers
from torch.nn.functional import gumbel_softmax, sigmoid, softmax
import torch.nn as nn 
from typing import Sequence
from torch.special import erf
import numpy as np

# class Local_Sparse_layer(nn.Module):
#     def __init__(self, size_in, size_out):
#         super().__init__()



class Decoder(torch.nn.Module):
    """
    Decodes data from latent space of ``n_input`` dimensions ``n_output``dimensions.

    Uses a fully-connected neural network of ``n_hidden`` layers.

    Parameters
    ----------
    n_input
        The dimensionality of the input (latent space)
    n_output
        The dimensionality of the output (data space)
    n_cat_list
        A list containing the number of categories
        for each category of interest. Each category will be
        included using a one-hot encoding
    n_layers
        The number of fully-connected hidden layers
    n_hidden
        The number of nodes per hidden layer
    inject_covariates
        Whether to inject covariates in each layer, or just the first (default).
    use_batch_norm
        Whether to use batch norm in layers
    use_layer_norm
        Whether to use layer norm in layers
    deeply_inject_covariates
        Whether to deeply inject covariates into all layers. If False (default),
        covairates will only be included in the input layer.
    """

    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_cat_list: Iterable[int] = None,
        n_layers: int = 2,
        n_hidden: int = 128,
        use_batch_norm: bool = False,
        use_layer_norm: bool = True,
        deep_inject_covariates: bool = False,
    ):
        super().__init__()
        self.px_decoder = FCLayers(
            n_in=n_input,
            n_out=n_hidden,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=0,
            activation_fn=torch.nn.LeakyReLU,
            use_batch_norm=use_batch_norm,
            use_layer_norm=use_layer_norm,
            inject_covariates=deep_inject_covariates,
        )
        self.output = torch.nn.Sequential(
            torch.nn.Linear(n_hidden, n_output), torch.nn.Sigmoid()
        )

    def forward(self, z: torch.Tensor, *cat_list: int):
        x = self.output(self.px_decoder(z, *cat_list))
        return x


class parallel_linear_layer(torch.nn.Module):
  
  def __init__(self, h, w):
    super(parallel_linear_layer, self).__init__()
    self.weights = nn.Parameter(torch.Tensor(h, w))  # define the trainable parameter
    self.bias = nn.Parameter(torch.Tensor(w))
    # print("linear weight shape {}".format(self.weights.shape))
    # print("bias shape is {}".format(self.bias.shape))


  def forward(self, x):
    # assuming x is of size b-n-h-w
    # print("input shape {}, weight shape {}".format(x.shape, self.weights.shape))
    x = torch.mul(x, self.weights.unsqueeze(0))  # element-wise multiplication
    x = torch.sum(x, dim=1)
    x = torch.add(x, self.bias)
    # print("parallel output shape is {}".format(x.shape))
    return x


class GateLayer(torch.nn.Module):
    def __init__(self, n_input, n_output, n_hidden, n_layers, sigma=1, finetune=False) :
        super().__init__()
        self.n_input = n_input
        self.n_output = n_output
        self.n_layer = n_layers
        self.FC = FCLayers(
            n_in=n_input,
            n_out=n_output,
            n_layers=n_layers,
            n_hidden=n_hidden
        )
        self.sigma = sigma
        self.finetune = finetune
    
    def set_finetune(self, finetune):
        self.finetune = finetune    

    def forward(self, x):
        
        # if not self.finetune:
        #     # print("Forward Gates finetune {}".format(self.finetune))

        #     u = torch.sigmoid(self.FC(x))
        #     z = u

        # ## finetune the sparsity parameters
        # else:
        u = torch.sigmoid(self.FC(x))
        # print("finished sigmoid u")
        e = torch.normal(torch.zeros_like(u), self.sigma*torch.ones_like(u))
        # print("get the e tensor")
        z = u + e + 0.5
        z = torch.minimum(z, torch.ones_like(z))
        # print("get min")
        z = torch.maximum(torch.zeros_like(z), z)

        ## calculate the loss
        # u = -(u + 0.5) / (np.sqrt(2) * self.sigma)
        # u = 0.5 - 0.5 * erf(u)
        # bs, _ = u.shape
        # sparse_loss = torch.sum(u) / bs 

        return z


    @torch.no_grad()
    def inference(self, x):
        u = torch.sigmoid(self.FC(x))
        z = u + 0.5
        z = torch.min(z, torch.ones_like(z))
        z = torch.max(0, z)
        return z
    
    # @torch.no_grad()
    def sparsity_loss(self, z):
        u = torch.sigmoid(self.FC(z))
        u = -(u + 0.5) / (np.sqrt(2) * self.sigma)
        u = 0.5 - 0.5 * erf(u)
        bs, _ = u.shape

        return torch.sum(u) / bs 

        

class GateDecoder(torch.nn.Module):
    """
    Decodes data from latent space of ``n_input`` dimensions ``n_output``dimensions.

    Uses a fully-connected neural network of ``n_hidden`` layers.

    Parameters
    ----------
    n_input
        The dimensionality of the input (latent space)
    n_output
        The dimensionality of the output (data space)
    n_cat_list
        A list containing the number of categories
        for each category of interest. Each category will be
        included using a one-hot encoding
    n_layers
        The number of fully-connected hidden layers
    n_hidden
        The number of nodes per hidden layer
    inject_covariates
        Whether to inject covariates in each layer, or just the first (default).
    use_batch_norm
        Whether to use batch norm in layers
    use_layer_norm
        Whether to use layer norm in layers
    deeply_inject_covariates
        Whether to deeply inject covariates into all layers. If False (default),
        covairates will only be included in the input layer.
    """

    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_cat_list: Iterable[int] = None,
        n_layers: int = 2,
        n_hidden_local: int = 8,
        n_hidden_global: int=128,
        use_batch_norm: bool = False,
        use_layer_norm: bool = True,
        deep_inject_covariates: bool = False,
        fine_tune  = False
    ):
        super().__init__()
        self.n_input = n_input
        self.n_output = n_output
        self.n_hidden_local = n_hidden_local
        self.n_hidden_global = n_hidden_global
        self.n_cat_list = n_cat_list
        self.fine_tune = fine_tune
        # self.gate_layer = FCLayers(
        #     n_in=n_input,
        #     n_out=n_input,
        #     # n_out = n_input*n_hidden_local,
        #     n_cat_list=n_cat_list,
        #     n_layers=n_layers,
        #     n_hidden=n_hidden_global,
        #     dropout_rate=0,
        #     activation_fn=torch.nn.LeakyReLU,
        #     use_batch_norm=use_batch_norm,
        #     use_layer_norm=use_layer_norm,
        # )

        # self.gate_layer = FCLayers(
        #     n_in=n_input,
        #     n_out=n_input*n_output,
        #     # n_out = n_input*n_hidden_local,
        #     n_cat_list=n_cat_list,
        #     n_layers=1,
        #     n_hidden=n_hidden_global,
        #     dropout_rate=0,
        #     activation_fn=torch.nn.LeakyReLU,
        #     use_batch_norm=use_batch_norm,
        #     use_layer_norm=use_layer_norm,
        # )

        self.gate_layer = GateLayer(
            n_input=n_input,
            n_output=n_input*n_output,
            n_layers=1,
            n_hidden=n_hidden_global,
            sigma= 0.1,
            finetune=False
        )


        feature_list = []
        for i in range(self.n_input):

            feature_list.append(FCLayers(
            n_in=1,
            n_out=n_hidden_local,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden_local,
            dropout_rate=0,
            activation_fn=torch.nn.LeakyReLU,
            use_batch_norm=use_batch_norm,
            use_layer_norm=use_layer_norm,
            inject_covariates=deep_inject_covariates))

        self.feature_nns = nn.ModuleList(feature_list)



        # output_list = []
        # for i in range(self.n_output):

        #     output_list.append( 
        #     torch.nn.Sequential(
        #     torch.nn.Linear(n_hidden_local*n_input, 1),
        #     torch.nn.Sigmoid()))

        # self.output_networks = nn.ModuleList(output_list)

        # self.feature_nns = nn.ModuleList([FCLayers(
        #     n_in=1,
        #     # name= f"FeatureNN_{i}",
        #     n_out=n_hidden_local,
        #     n_cat_list=n_cat_list,
        #     n_layers=n_layers,
        #     n_hidden=n_hidden_local,
        #     dropout_rate=0,
        #     activation_fn=torch.nn.LeakyReLU,
        #     use_batch_norm=use_batch_norm,
        #     use_layer_norm=use_layer_norm,
        #     inject_covariates=deep_inject_covariates,
        # ) for i in range(self.n_input)])
        # # print("feature_nns length is {}".format(len(self.feature_nns)))
        # print("feature_nns[1] is {}".format(self.feature_nns[0]))


        # self.feature_nns = FCLayers(
        #     n_in=n_input,
        #     # name= f"FeatureNN_{i}",
        #     n_out=n_input * n_hidden_local,
        #     n_cat_list=n_cat_list,
        #     n_layers=n_layers,
        #     n_hidden=n_hidden_local,
        #     dropout_rate=0,
        #     activation_fn=torch.nn.LeakyReLU,
        #     use_batch_norm=use_batch_norm,
        #     use_layer_norm=use_layer_norm,
        #     inject_covariates=deep_inject_covariates,
        # ) 
        # print("feature_nns length is {}".format(len(self.feature_nns)))
        # print("feature_nns[1] is {}".format(self.feature_nns[0]))



        # self.output = FCLayers(
        #     n_in=n_input * n_hidden_local,
        #     n_out=n_output,
        #     n_cat_list=n_cat_list,
        #     n_layers=1,
        #     n_hidden=n_hidden_local,
        #     dropout_rate=0,
        #     activation_fn=torch.nn.LeakyReLU,
        #     use_batch_norm=use_batch_norm,
        #     use_layer_norm=use_layer_norm,
        #     inject_covariates=deep_inject_covariates,
        # )

        # self.output = torch.nn.Sequential(
        #     torch.nn.Linear(n_hidden_local*n_input, n_output),
        #     torch.nn.Sigmoid()
        # )

        self.output = torch.nn.Sequential(
            parallel_linear_layer(n_hidden_local*n_input, n_output),
            torch.nn.Sigmoid()
        )

        # self.output = torch.nn.Sequential(
        #     torch.nn.Linear(, n_output),
        #     torch.nn.Sigmoid()
        # )

        # self.gumble = gumbel_softmax(logits=self.n_input)
        print("gate decoder initialization n_input {}, n_output {}, \
        n_hidden_local {}, n_hidden_global {}, n_cat_list {}, *cat_list {}".format(self.n_input, self.n_output, \
            self.n_hidden_local, self.n_hidden_global, n_cat_list, *n_cat_list))

        # self.px_decoder = FCLayers(
        #     n_in=1,
        #     n_out=n_hidden,
        #     n_cat_list=n_cat_list,
        #     n_layers=n_layers,
        #     n_hidden=n_hidden,
        #     dropout_rate=0,
        #     activation_fn=torch.nn.LeakyReLU,
        #     use_batch_norm=use_batch_norm,
        #     use_layer_norm=use_layer_norm,
        #     inject_covariates=deep_inject_covariates,
        # )
        # self.output = torch.nn.Sequential(
        #     torch.nn.Linear(n_hidden, n_output), torch.nn.Sigmoid()
        # )
    def calc_feature_outputs(self, inputs, *cat_list: int):
        """Returns the output computed by each feature net."""
        # print("inputs shape of feature_nn is {}".format(inputs.shape))
        results = []
        _, col = inputs.shape
        for i in range(col):
            # print("{} single input shape {}".format(i, inputs[:,i].unsqueeze(1).shape))
            result = self.feature_nns[i](inputs[:,i].unsqueeze(1), *cat_list)
            # print("single result shape {}".format(result.shape))
            results.append(result)

        # results = self.feature_nns(inputs, *cat_list)    


        # return [self.feature_nns[i](inputs[:, i]) for i in range(self.n_input)]
        return results

    def calc_gates(self, inputs, *cat_list: int):
        # print("inputs shape of calc_gates: {}".format(inputs.shape))

        # return gumbel_softmax(self.gate_layer(inputs, *cat_list))
        return self.gate_layer(inputs, *cat_list)


    # def calc_gates_feature(self, inputs:torch.Tensor,  *cat_list: int):
    #     features = self.calc_feature_outputs(inputs, *cat_list)
    #     gates = self.calc_gates(inputs, *cat_list)
    #     # return torch.mul(gates, features)
    #     g_rows = gates.shape[0] 
    #     # print("g_rows {}".format(g_rows))
    #     # print("gates shape: {}".format(gates.shape))
    #     # print("gates output {}".format(gates))

    #     for i in range(self.n_input):
    #         temp = gates[:, i].unsqueeze(1)
    #         # print("unsqueezed shape {}".format(temp.shape))
    #         temp = temp.repeat(1, self.n_hidden_local)
    #         # print("temp shape {}, hidden_local {}".format(temp.shape, self.n_hidden_local))
    #         # print("feature[i] shape is {}".format(features[i].shape))
    #         features[i] = torch.mul(temp, features[i])
    #     return features

    # def calc_gates_feature(self, inputs:torch.Tensor,  *cat_list: int):
    #     features = self.calc_feature_outputs(inputs, *cat_list)
    #     gates = self.calc_gates(inputs, *cat_list)
    #     # print("linear layer shape {}".format(self.output[0].weight.shape))

    #     # print("gates.shape {}".format(gates.shape))
    #     gates = torch.mean(gates, 0)
    #     gates = torch.reshape(gates, (self.n_input, self.n_output))
    #     # gates_param = torch.autograd.Variable(gates.clone(), requires_grad=True)
    #     # print("gates_param is leaf: {}".format(gates_param.is_leaf))
    #     # with torch.no_grad():
    #     Y = torch.empty([self.n_input*self.n_hidden_local, self.n_output], device='cuda')
    #     for i in range(self.n_input):
    #         single_gate = gates[i, :]
    #         single_gate = single_gate.repeat(self.n_hidden_local, 1)
    #         # single_gate = torch.autograd.Variable(single_gate.clone(), requires_grad=True)
    #         # print("single_gate is leaf: {}".format(single_gate.is_leaf))
    #         # print("Y is leaf: {}".format(Y.is_leaf))
    #         Y[i*self.n_hidden_local : (i+1)*self.n_hidden_local, :] = single_gate
        
    #     with torch.no_grad():
    
    #         self.output[0].weight = torch.nn.parameter.Parameter(torch.mul(self.output[0].weight, Y.t()))
        
    #     return features, gates

    def calc_gates_repeat(self, inputs:torch.Tensor,  *cat_list: int):
        gates = self.calc_gates(inputs, *cat_list)
        gates = torch.mean(gates, 0)
        gates = torch.reshape(gates, (self.n_input, self.n_output))

        Y = torch.empty([self.n_input*self.n_hidden_local, self.n_output], device='cuda')
        for i in range(self.n_input):
            single_gate = gates[i, :]
            single_gate = single_gate.repeat(self.n_hidden_local, 1)
            # single_gate = torch.autograd.Variable(single_gate.clone(), requires_grad=True)
            # print("single_gate is leaf: {}".format(single_gate.is_leaf))
            # print("Y is leaf: {}".format(Y.is_leaf))
            Y[i*self.n_hidden_local : (i+1)*self.n_hidden_local, :] = single_gate
        
        return Y


    # def calc_gated_feature(self, gates, contact_output):
    #     feature_matrix = []
    #     feature_list = []
    #     batch, _ = contact_output.shape
    #     print("gates shape {}, concat shape {}".format(gates.shape, contact_output.shape))
    #     for b in range(batch):
    #         for i in range(gates.shape[1]):
    #             gated_feature = torch.mul(gates[:, i], contact_output[b,:])
    #             feature_list.append(gated_feature)
    #         feature_matrix.append(feature_list)    
    #     return feature_matrix

    # def reshape_output(self, feature_matrix):
    #     batch = len(feature_matrix)
    #     reshap_matrix = []

    #     for i in range(self.n_output):
    #         peak_matrix = []
    #         for b in range(batch):
    #             peak_matrix.append(feature_matrix[b][i])
    #         peak_matrix = torch.tensor(peak_matrix)
    #         reshap_matrix.append(peak_matrix)
    #     return reshap_matrix

    # def peak_forward(self, reshap_matrix):
    #     output_list = []
    #     for i in range(self.n_output):
            
    #         output_list.append(self.output_networks[i](reshap_matrix[i]))

    #     output_list = torch.cat(output_list)
    #     return output_list    
    # def calc_gates_feature(self, inputs:torch.Tensor,  *cat_list: int):
    #     features = self.calc_feature_outputs(inputs, *cat_list)
    #     gates = self.calc_gates(inputs, *cat_list)
    #     # print("linear layer shape {}".format(self.output[0].weight.shape))

    #     # print("gates.shape {}".format(gates.shape))
    #     gates = torch.mean(gates, 0)
    #     gates = torch.reshape(gates, (self.n_input, self.n_output))
    #     # gates_param = torch.autograd.Variable(gates.clone(), requires_grad=True)
    #     # print("gates_param is leaf: {}".format(gates_param.is_leaf))

    #     Y = torch.empty([self.n_input*self.n_hidden_local, self.n_output], device='cuda')
    #     for i in range(self.n_input):
    #         single_gate = gates[i, :]
    #         single_gate = single_gate.repeat(self.n_hidden_local, 1)
    #         # single_gate = torch.autograd.Variable(single_gate.clone(), requires_grad=True)
    #         # print("single_gate is leaf: {}".format(single_gate.is_leaf))
    #         # print("Y is leaf: {}".format(Y.is_leaf))
    #         Y[i*self.n_hidden_local : (i+1)*self.n_hidden_local, :] = single_gate
    #     # print("Y shape {}".format(Y.shape))

    #     # features = torch.mm(features, Y)
    #     # print("after product feature shape {}".format(features.shape))
    #     # print("linear layer shape {}".format(self.output[0].weight.shape))
    #     # self.output[0].weight = torch.mul(self.output[0].weight, Y.t())
    #     return features, Y
    # def forward(self, z: torch.Tensor, *cat_list: int):
    #     # print("z shape is {}".format(z.shape))
    #     individual_outputs = self.calc_gates_feature(z, self.n_cat_list)
    #     conc_out = torch.cat(individual_outputs, dim=-1)
    #     # print("conc_out shape {}".format(conc_out.shape))
    #     x = self.output(conc_out)
    #     return x

    # def local_repeat(self, gates):
    #     Y =  torch.FloatTensor([])
    #     for i in range(self.n_input):
    #         single_gate = gates[i, :]
    #         single_gate = single_gate.repeat(self.n_hidden_local, 1)
    #         Y = torch.cat([Y, single_gate])
    #     return Y    



    # def local_forward(self, conc_out, input):
    #     gates = self.calc_gates(input)
    #     batch = input.shape[0]
    #     results =  torch.FloatTensor([])
    #     for i in range(batch):
    #         local_gate = gates[i, :]
    #         local_gate = torch.reshape(local_gate, ( self.n_input, self.n_output))
    #         local_gate = self.local_repeat(local_gate)
    #         self.output[0].weight = torch.mul(self.output[0].weight, local_gate.t())
    #         result =  torch.add(torch.mm(x, self.output[0].weights.t()),  self.output[0].bias)
    #         result = result.


    # def forward(self, z: torch.Tensor, *cat_list: int):
    #     # print("z shape is {}".format(z.shape))
    #     # individual_outputs, gates = self.calc_gates_feature(z, self.n_cat_list)
    #     individual_outputs = self.calc_feature_outputs(z, self.n_cat_list)
    #     conc_out = torch.cat(individual_outputs, dim=-1)
    #     # print("conc_out shape {}".format(conc_out.shape))
    #     # print("gates shape {}".format(gates.shape))
    #     # features = torch.matmul(conc_out, gates)
    #     x = self.output(conc_out)
    #     return x  
    def set_finetune(self, finetune):
        self.fine_tune = finetune
        self.gate_layer.set_finetune(finetune)


    def forward(self, z: torch.Tensor, *cat_list: int):
        # print("z shape is {}".format(z.shape))
        # gates = self.calc_gates_repeat(z)
        individual_outputs = self.calc_feature_outputs(z, self.n_cat_list)
        conc_out = torch.cat(individual_outputs, dim=-1)
        # print("before gates shape is {}".format(gates.shape))
        conc_out = conc_out.unsqueeze(-1)
        # print("after contacnated feature shape is {}".format(conc_out.shape))
        # print("gate layer fine tune parameters are {}".format(self.fine_tune))
        if self.fine_tune:
            gates = self.calc_gates_repeat(z)
            x = torch.mul(conc_out, gates)

        else:
            x = conc_out    

        x = self.output(x)
        # feature_list = self.calc_gated_feature(gates, conc_out)
        # reshape_matrix = self.reshape_output(feature_list)
        # print("")
        # output = self.peak_forward(reshape_matrix)

        
        # print("conc_out shape {}".format(conc_out.shape))
        # print("gates shape {}".format(gates.shape))
        # features = torch.matmul(conc_out, gates)
        # x = self.output(conc_out)
        return x
    @torch.no_grad()
    def get_gate_regu(self, z):
        return self.gate_layer.sparsity_loss(z) 

    @torch.no_grad()
    def get_loading_global(self, z:torch.Tensor):
        _, gates = self.calc_gates_feature(z, self.n_cat_list)
        return gates.detach().cpu().numpy()

    @torch.no_grad()
    def get_loading_local(self, inputs:torch.Tensor):
        gates = self.calc_gates(inputs, self.n_cat_list)
        gates = torch.reshape(gates, (gates.shape[0], self.n_input, self.n_output))

        return gates.detach().cpu().numpy()


    @torch.no_grad()
    def get_loading_global_weights(self, inputs:torch.Tensor):

        loadings = self.calc_gates_repeat(inputs)
        # print(loadings.device, self.output[0].weights.unsqueeze(0).device)
        loadings = torch.mul(self.output[0].weights.unsqueeze(0), loadings)
        # loadings =np.abs(loadings.detach().cpu().numpy())
        # loadings = self.output[0].weights.unsqueeze(0)
        loadings =np.abs(loadings.detach().cpu().numpy())
        # print("loadings shape is {}".format(loadings.shape))
        aggregate_loadings = []
        for i in range(self.n_input):
            aggregate_loadings.append(np.mean(loadings[:, i*self.n_hidden_local:(i+1)*self.n_hidden_local, :], axis=1))
        aggregate_loadings = np.concatenate(aggregate_loadings, axis=0)

        return aggregate_loadings


    @torch.no_grad()
    def get_loading_feature(self):

        # loadings = self.calc_gates_repeat(inputs)
        # print(loadings.device, self.output[0].weights.unsqueeze(0).device)
        # loadings = torch.mul(self.output[0].weights.unsqueeze(0), loadings)
        # loadings =np.abs(loadings.detach().cpu().numpy())
        loadings = self.output[0].weights
        loadings =np.abs(loadings.detach().cpu().numpy())
        # print("loadings shape is {}".format(loadings.shape))
        # aggregate_loadings = []
        # for i in range(self.n_input):
        #     aggregate_loadings.append(np.mean(loadings[:, i*self.n_hidden_local:(i+1)*self.n_hidden_local, :], axis=1))
        # aggregate_loadings = np.concatenate(aggregate_loadings, axis=0)

        return loadings


    @torch.no_grad()
    def get_loading_merged_feature(self):

        loadings = self.output[0].weights
        loadings =np.abs(loadings.detach().cpu().numpy())
        # print("loading shape is {}".format(loadings.shape))
        aggregate_loadings = []
        for i in range(self.n_input):
            aggregate_loadings.append(np.expand_dims(np.mean(loadings[i*8:(i+1)*8, :], axis=0),axis=0))
        aggregate_loadings = np.concatenate(aggregate_loadings, axis=0)

        return aggregate_loadings    

         
    

class BinaryGateDecoder(torch.nn.Module):
    """
    Decodes data from latent space of ``n_input`` dimensions ``n_output``dimensions.

    Uses a fully-connected neural network of ``n_hidden`` layers.

    Parameters
    ----------
    n_input
        The dimensionality of the input (latent space)
    n_output
        The dimensionality of the output (data space)
    n_cat_list
        A list containing the number of categories
        for each category of interest. Each category will be
        included using a one-hot encoding
    n_layers
        The number of fully-connected hidden layers
    n_hidden
        The number of nodes per hidden layer
    inject_covariates
        Whether to inject covariates in each layer, or just the first (default).
    use_batch_norm
        Whether to use batch norm in layers
    use_layer_norm
        Whether to use layer norm in layers
    deeply_inject_covariates
        Whether to deeply inject covariates into all layers. If False (default),
        covairates will only be included in the input layer.
    """

    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_cat_list: Iterable[int] = None,
        n_layers: int = 2,
        n_hidden_local: int = 8,
        n_hidden_global: int=128,
        use_batch_norm: bool = False,
        use_layer_norm: bool = True,
        deep_inject_covariates: bool = False,
        fine_tune  = False
    ):
        super().__init__()
        self.n_input = n_input
        self.n_output = n_output
        self.n_hidden_local = n_hidden_local
        self.n_hidden_global = n_hidden_global
        self.n_cat_list = n_cat_list
        self.fine_tune = fine_tune
        
        ## Binary Masks has two decouple and couple senarios
        self.gate_layer = GateLayer(
            n_input=n_input,
            n_output=2*n_output,
            n_layers=1,
            n_hidden=n_hidden_global,
            sigma= 0.1,
            finetune=False
        )


        feature_list = []
        for i in range(self.n_input):

            feature_list.append(FCLayers(
            n_in=1,
            n_out=n_hidden_local,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden_local,
            dropout_rate=0,
            activation_fn=torch.nn.LeakyReLU,
            use_batch_norm=use_batch_norm,
            use_layer_norm=use_layer_norm,
            inject_covariates=deep_inject_covariates))

        self.feature_nns = nn.ModuleList(feature_list)



        self.output = torch.nn.Sequential(
            parallel_linear_layer(n_hidden_local*n_input, n_output),
            torch.nn.Sigmoid()
        )

 
        print("gate decoder initialization n_input {}, n_output {}, \
        n_hidden_local {}, n_hidden_global {}, n_cat_list {}, *cat_list {}".format(self.n_input, self.n_output, \
            self.n_hidden_local, self.n_hidden_global, n_cat_list, *n_cat_list))

       
    def calc_feature_outputs(self, inputs, *cat_list: int):
        """Returns the output computed by each feature net."""
        # print("inputs shape of feature_nn is {}".format(inputs.shape))
        results = []
        _, col = inputs.shape
        for i in range(col):
            # print("{} single input shape {}".format(i, inputs[:,i].unsqueeze(1).shape))
            result = self.feature_nns[i](inputs[:,i].unsqueeze(1), *cat_list)
            # print("single result shape {}".format(result.shape))
            results.append(result)

        return results

    def calc_gates(self, inputs, *cat_list: int):

        return self.gate_layer(inputs, *cat_list)

    def calc_gates_repeat(self, inputs:torch.Tensor,  *cat_list: int):
        gates = self.calc_gates(inputs, *cat_list)
        gates = torch.mean(gates, 0)
        gates = torch.reshape(gates, (self.n_input, self.n_output))

        Y = torch.empty([self.n_input*self.n_hidden_local, self.n_output], device='cuda')
        for i in range(self.n_input):
            single_gate = gates[i, :]
            single_gate = single_gate.repeat(self.n_hidden_local, 1)
            # single_gate = torch.autograd.Variable(single_gate.clone(), requires_grad=True)
            # print("single_gate is leaf: {}".format(single_gate.is_leaf))
            # print("Y is leaf: {}".format(Y.is_leaf))
            Y[i*self.n_hidden_local : (i+1)*self.n_hidden_local, :] = single_gate
        
        return Y


    def set_finetune(self, finetune):
        self.fine_tune = finetune
        self.gate_layer.set_finetune(finetune)


    def forward(self, z: torch.Tensor, *cat_list: int):
        # print("z shape is {}".format(z.shape))
        # gates = self.calc_gates_repeat(z)
        individual_outputs = self.calc_feature_outputs(z, self.n_cat_list)
        conc_out = torch.cat(individual_outputs, dim=-1)
        # print("before gates shape is {}".format(gates.shape))
        conc_out = conc_out.unsqueeze(-1)
        # print("after contacnated feature shape is {}".format(conc_out.shape))
        if self.fine_tune:
            gates = self.calc_gates_repeat(z)
            x = torch.mul(conc_out, gates)

        else:
            x = conc_out    

        x = self.output(x)
       
        return x
    @torch.no_grad()
    def get_gate_regu(self, z):
        return self.gate_layer.sparsity_loss(z) 

    @torch.no_grad()
    def get_loading_global(self, z:torch.Tensor):
        _, gates = self.calc_gates_feature(z, self.n_cat_list)
        return gates.detach().cpu().numpy()

    @torch.no_grad()
    def get_loading_local(self, inputs:torch.Tensor):
        gates = self.calc_gates(inputs, self.n_cat_list)
        gates = torch.reshape(gates, (gates.shape[0], self.n_input, self.n_output))

        return gates.detach().cpu().numpy()


    @torch.no_grad()
    def get_loading_global_weights(self, inputs:torch.Tensor):

        # loadings = self.calc_gates_repeat(inputs)
        # loadings = torch.mul(self.output[0].weights.unsqueeze(0), loadings)
        loadings = self.output[0].weights.unsqueeze(0)
        loadings = loadings.detach().cpu().numpy()
        return loadings





class Decoder(torch.nn.Module):
    """
    Decodes data from latent space of ``n_input`` dimensions ``n_output``dimensions.

    Uses a fully-connected neural network of ``n_hidden`` layers.

    Parameters
    ----------
    n_input
        The dimensionality of the input (latent space)
    n_output
        The dimensionality of the output (data space)
    n_cat_list
        A list containing the number of categories
        for each category of interest. Each category will be
        included using a one-hot encoding
    n_layers
        The number of fully-connected hidden layers
    n_hidden
        The number of nodes per hidden layer
    inject_covariates
        Whether to inject covariates in each layer, or just the first (default).
    use_batch_norm
        Whether to use batch norm in layers
    use_layer_norm
        Whether to use layer norm in layers
    deeply_inject_covariates
        Whether to deeply inject covariates into all layers. If False (default),
        covairates will only be included in the input layer.
    """

    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_cat_list: Iterable[int] = None,
        n_layers: int = 2,
        n_hidden: int = 128,
        use_batch_norm: bool = False,
        use_layer_norm: bool = True,
        deep_inject_covariates: bool = False,
    ):
        super().__init__()
        self.px_decoder = FCLayers(
            n_in=n_input,
            n_out=n_hidden,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=0,
            activation_fn=torch.nn.LeakyReLU,
            use_batch_norm=use_batch_norm,
            use_layer_norm=use_layer_norm,
            inject_covariates=deep_inject_covariates,
        )
        self.output = torch.nn.Sequential(
            torch.nn.Linear(n_hidden, n_output), torch.nn.Sigmoid()
        )

    def forward(self, z: torch.Tensor, *cat_list: int):
        x = self.output(self.px_decoder(z, *cat_list))
        return x


class PEAKVAE(BaseModuleClass):
    """
    Variational auto-encoder model for ATAC-seq data.

    This is an implementation of the peakVI model descibed in.

    Parameters
    ----------
    n_input_regions
        Number of input regions.
    n_batch
        Number of batches, if 0, no batch correction is performed.
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
    model_depth
        Model library size factors or not.
    region_factors
        Include region-specific factors in the model
    use_batch_norm
        One of the following

        * ``'encoder'`` - use batch normalization in the encoder only
        * ``'decoder'`` - use batch normalization in the decoder only
        * ``'none'`` - do not use batch normalization (default)
        * ``'both'`` - use batch normalization in both the encoder and decoder
    use_layer_norm
        One of the following

        * ``'encoder'`` - use layer normalization in the encoder only
        * ``'decoder'`` - use layer normalization in the decoder only
        * ``'none'`` - do not use layer normalization
        * ``'both'`` - use layer normalization in both the encoder and decoder (default)
    latent_distribution
        which latent distribution to use, options are

        * ``'normal'`` - Normal distribution (default)
        * ``'ln'`` - Logistic normal distribution (Normal(0, I) transformed by softmax)
    deeply_inject_covariates
        Whether to deeply inject covariates into all layers of the decoder. If False (default),
        covairates will only be included in the input layer.

    """

    def __init__(
        self,
        n_input_regions: int,
        n_batch: int = 0,
        n_hidden: Optional[int] = None,
        n_latent: Optional[int] = None,
        n_layers_encoder: int = 2,
        n_layers_decoder: int = 2,
        n_continuous_cov: int = 0,
        n_cats_per_cov: Optional[Iterable[int]] = None,
        dropout_rate: float = 0.1,
        model_depth: bool = True,
        region_factors: bool = True,
        use_batch_norm: Literal["encoder", "decoder", "none", "both"] = "none",
        use_layer_norm: Literal["encoder", "decoder", "none", "both"] = "both",
        latent_distribution: str = "normal",
        deeply_inject_covariates: bool = False,
        encode_covariates: bool = False,
    ):
        super().__init__()

        self.n_input_regions = n_input_regions
        self.n_hidden = (
            int(np.sqrt(self.n_input_regions)) if n_hidden is None else n_hidden
        )
        self.n_latent = int(np.sqrt(self.n_hidden)) if n_latent is None else n_latent
        self.n_layers_encoder = n_layers_encoder
        self.n_layers_decoder = n_layers_decoder
        self.n_cats_per_cov = n_cats_per_cov
        self.n_continuous_cov = n_continuous_cov
        self.model_depth = model_depth
        self.dropout_rate = dropout_rate
        self.latent_distribution = latent_distribution
        self.use_batch_norm_encoder = use_batch_norm in ("encoder", "both")
        self.use_batch_norm_decoder = use_batch_norm in ("decoder", "both")
        self.use_layer_norm_encoder = use_layer_norm in ("encoder", "both")
        self.use_layer_norm_decoder = use_layer_norm in ("decoder", "both")
        self.deeply_inject_covariates = deeply_inject_covariates
        self.encode_covariates = encode_covariates

        cat_list = (
            [n_batch] + list(n_cats_per_cov) if n_cats_per_cov is not None else []
        )

        n_input_encoder = self.n_input_regions + n_continuous_cov * encode_covariates
        encoder_cat_list = cat_list if encode_covariates else None
        self.z_encoder = Encoder(
            n_input=n_input_encoder,
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

        self.z_decoder = Decoder(
            n_input=self.n_latent + self.n_continuous_cov,
            n_output=n_input_regions,
            n_hidden=self.n_hidden,
            n_cat_list=cat_list,
            n_layers=self.n_layers_decoder,
            use_batch_norm=self.use_batch_norm_decoder,
            use_layer_norm=self.use_layer_norm_decoder,
            deep_inject_covariates=self.deeply_inject_covariates,
        )

        self.d_encoder = None
        if self.model_depth:
            # Decoder class to avoid variational split
            self.d_encoder = Decoder(
                n_input=n_input_encoder,
                n_output=1,
                n_hidden=self.n_hidden,
                n_cat_list=encoder_cat_list,
                n_layers=self.n_layers_encoder,
            )
        self.region_factors = None
        if region_factors:
            self.region_factors = torch.nn.Parameter(torch.zeros(self.n_input_regions))

    def _get_inference_input(self, tensors):
        x = tensors[REGISTRY_KEYS.X_KEY]
        batch_index = tensors[REGISTRY_KEYS.BATCH_KEY]
        cont_covs = tensors.get(REGISTRY_KEYS.CONT_COVS_KEY)
        cat_covs = tensors.get(REGISTRY_KEYS.CAT_COVS_KEY)
        input_dict = dict(
            x=x,
            batch_index=batch_index,
            cont_covs=cont_covs,
            cat_covs=cat_covs,
        )
        return input_dict

    def _get_generative_input(self, tensors, inference_outputs, transform_batch=None):
        z = inference_outputs["z"]
        qz_m = inference_outputs["qz"].loc
        batch_index = tensors[REGISTRY_KEYS.BATCH_KEY]
        cont_covs = tensors.get(REGISTRY_KEYS.CONT_COVS_KEY)

        cat_covs = tensors.get(REGISTRY_KEYS.CAT_COVS_KEY)

        if transform_batch is not None:
            batch_index = torch.ones_like(batch_index) * transform_batch
        input_dict = {
            "z": z,
            "qz_m": qz_m,
            "batch_index": batch_index,
            "cont_covs": cont_covs,
            "cat_covs": cat_covs,
        }
        return input_dict

    def get_reconstruction_loss(self, p, d, f, x):
        rl = torch.nn.BCELoss(reduction="none")(p * d * f, (x > 0).float()).sum(dim=-1)
        return rl

    @auto_move_data
    def inference(
        self,
        x,
        batch_index,
        cont_covs,
        cat_covs,
        n_samples=1,
    ) -> Dict[str, torch.Tensor]:
        """Helper function used in forward pass."""
        if cat_covs is not None and self.encode_covariates:
            categorical_input = torch.split(cat_covs, 1, dim=1)
        else:
            categorical_input = tuple()
        if cont_covs is not None and self.encode_covariates:
            encoder_input = torch.cat([x, cont_covs], dim=-1)
        else:
            encoder_input = x
        # if encode_covariates is False, cat_list to init encoder is None, so
        # batch_index is not used (or categorical_input, but it's empty)
        qz, z = self.z_encoder(encoder_input, batch_index, *categorical_input)
        d = (
            self.d_encoder(encoder_input, batch_index, *categorical_input)
            if self.model_depth
            else 1
        )

        if n_samples > 1:
            # when z is normal, untran_z == z
            untran_z = qz.sample((n_samples,))
            z = self.z_encoder.z_transformation(untran_z)

        return dict(d=d, qz=qz, z=z)

    @auto_move_data
    def generative(
        self,
        z,
        qz_m,
        batch_index,
        cont_covs=None,
        cat_covs=None,
        use_z_mean=False,
    ):
        """Runs the generative model."""

        if cat_covs is not None:
            categorical_input = torch.split(cat_covs, 1, dim=1)
        else:
            categorical_input = tuple()

        latent = z if not use_z_mean else qz_m
        if cont_covs is None:
            decoder_input = latent
        elif latent.dim() != cont_covs.dim():
            decoder_input = torch.cat(
                [latent, cont_covs.unsqueeze(0).expand(latent.size(0), -1, -1)], dim=-1
            )
        else:
            decoder_input = torch.cat([latent, cont_covs], dim=-1)

        p = self.z_decoder(decoder_input, batch_index, *categorical_input)

        return dict(p=p)

    def loss(
        self, tensors, inference_outputs, generative_outputs, kl_weight: float = 1.0
    ):
        x = tensors[REGISTRY_KEYS.X_KEY]
        qz = inference_outputs["qz"]
        d = inference_outputs["d"]
        p = generative_outputs["p"]

        kld = kl_divergence(
            qz,
            Normal(0, 1),
        ).sum(dim=1)

        f = torch.sigmoid(self.region_factors) if self.region_factors is not None else 1
        rl = self.get_reconstruction_loss(p, d, f, x)

        loss = (rl.sum() + kld * kl_weight).sum()

        return LossRecorder(loss, rl, kld, kl_global=torch.tensor(0.0))





class MASKPEAKVAE(BaseModuleClass):
    """
    Variational auto-encoder model for ATAC-seq data.

    This is an implementation of the peakVI model descibed in.

    Parameters
    ----------
    n_input_regions
        Number of input regions.
    n_batch
        Number of batches, if 0, no batch correction is performed.
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
    model_depth
        Model library size factors or not.
    region_factors
        Include region-specific factors in the model
    use_batch_norm
        One of the following

        * ``'encoder'`` - use batch normalization in the encoder only
        * ``'decoder'`` - use batch normalization in the decoder only
        * ``'none'`` - do not use batch normalization (default)
        * ``'both'`` - use batch normalization in both the encoder and decoder
    use_layer_norm
        One of the following

        * ``'encoder'`` - use layer normalization in the encoder only
        * ``'decoder'`` - use layer normalization in the decoder only
        * ``'none'`` - do not use layer normalization
        * ``'both'`` - use layer normalization in both the encoder and decoder (default)
    latent_distribution
        which latent distribution to use, options are

        * ``'normal'`` - Normal distribution (default)
        * ``'ln'`` - Logistic normal distribution (Normal(0, I) transformed by softmax)
    deeply_inject_covariates
        Whether to deeply inject covariates into all layers of the decoder. If False (default),
        covairates will only be included in the input layer.

    """

    def __init__(
        self,
        n_input_regions: int,
        n_batch: int = 0,
        n_hidden: Optional[int] = None,
        n_latent: Optional[int] = None,
        n_layers_encoder: int = 2,
        n_layers_decoder: int = 2,
        n_continuous_cov: int = 0,
        n_cats_per_cov: Optional[Iterable[int]] = None,
        dropout_rate: float = 0.1,
        model_depth: bool = True,
        region_factors: bool = True,
        use_batch_norm: Literal["encoder", "decoder", "none", "both"] = "none",
        use_layer_norm: Literal["encoder", "decoder", "none", "both"] = "both",
        latent_distribution: str = "normal",
        deeply_inject_covariates: bool = False,
        encode_covariates: bool = False,
    ):
        super().__init__()

        self.n_input_regions = n_input_regions
        self.n_hidden = (
            int(np.sqrt(self.n_input_regions)) if n_hidden is None else n_hidden
        )
        self.n_latent = int(np.sqrt(self.n_hidden)) if n_latent is None else n_latent
        self.n_layers_encoder = n_layers_encoder
        self.n_layers_decoder = n_layers_decoder
        self.n_cats_per_cov = n_cats_per_cov
        self.n_continuous_cov = n_continuous_cov
        self.model_depth = model_depth
        self.dropout_rate = dropout_rate
        self.latent_distribution = latent_distribution
        self.use_batch_norm_encoder = use_batch_norm in ("encoder", "both")
        self.use_batch_norm_decoder = use_batch_norm in ("decoder", "both")
        self.use_layer_norm_encoder = use_layer_norm in ("encoder", "both")
        self.use_layer_norm_decoder = use_layer_norm in ("decoder", "both")
        self.deeply_inject_covariates = deeply_inject_covariates
        self.encode_covariates = encode_covariates

        cat_list = (
            [n_batch] + list(n_cats_per_cov) if n_cats_per_cov is not None else []
        )

        n_input_encoder = self.n_input_regions + n_continuous_cov * encode_covariates
        encoder_cat_list = cat_list if encode_covariates else None
        self.z_encoder = Encoder(
            n_input=n_input_encoder,
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

        self.z_decoder = GateDecoder(
            n_input=self.n_latent + self.n_continuous_cov,
            n_output=n_input_regions,
            n_hidden_global=self.n_hidden,
            n_cat_list=cat_list,
            n_layers=self.n_layers_decoder,
            use_batch_norm=self.use_batch_norm_decoder,
            use_layer_norm=self.use_layer_norm_decoder,
            deep_inject_covariates=self.deeply_inject_covariates,
        )

        self.d_encoder = None
        if self.model_depth:
            # Decoder class to avoid variational split
            self.d_encoder = Decoder(
                n_input=n_input_encoder,
                n_output=1,
                n_hidden=self.n_hidden,
                n_cat_list=encoder_cat_list,
                n_layers=self.n_layers_encoder,
            )
        self.region_factors = None
        if region_factors:
            self.region_factors = torch.nn.Parameter(torch.zeros(self.n_input_regions))

    def _get_inference_input(self, tensors):
        x = tensors[REGISTRY_KEYS.X_KEY]
        batch_index = tensors[REGISTRY_KEYS.BATCH_KEY]
        cont_covs = tensors.get(REGISTRY_KEYS.CONT_COVS_KEY)
        cat_covs = tensors.get(REGISTRY_KEYS.CAT_COVS_KEY)
        input_dict = dict(
            x=x,
            batch_index=batch_index,
            cont_covs=cont_covs,
            cat_covs=cat_covs,
        )
        return input_dict

    def _get_generative_input(self, tensors, inference_outputs, transform_batch=None):
        z = inference_outputs["z"]
        qz_m = inference_outputs["qz"].loc
        batch_index = tensors[REGISTRY_KEYS.BATCH_KEY]
        cont_covs = tensors.get(REGISTRY_KEYS.CONT_COVS_KEY)

        cat_covs = tensors.get(REGISTRY_KEYS.CAT_COVS_KEY)

        if transform_batch is not None:
            batch_index = torch.ones_like(batch_index) * transform_batch
        input_dict = {
            "z": z,
            "qz_m": qz_m,
            "batch_index": batch_index,
            "cont_covs": cont_covs,
            "cat_covs": cat_covs,
        }
        return input_dict

    def get_reconstruction_loss(self, p, d, f, x):
        rl = torch.nn.BCELoss(reduction="none")(p * d * f, (x > 0).float()).sum(dim=-1)
        return rl

    @auto_move_data
    def inference(
        self,
        x,
        batch_index,
        cont_covs,
        cat_covs,
        n_samples=1,
    ) -> Dict[str, torch.Tensor]:
        """Helper function used in forward pass."""
        if cat_covs is not None and self.encode_covariates:
            categorical_input = torch.split(cat_covs, 1, dim=1)
        else:
            categorical_input = tuple()
        if cont_covs is not None and self.encode_covariates:
            encoder_input = torch.cat([x, cont_covs], dim=-1)
        else:
            encoder_input = x
        # if encode_covariates is False, cat_list to init encoder is None, so
        # batch_index is not used (or categorical_input, but it's empty)
        qz, z = self.z_encoder(encoder_input, batch_index, *categorical_input)
        d = (
            self.d_encoder(encoder_input, batch_index, *categorical_input)
            if self.model_depth
            else 1
        )

        if n_samples > 1:
            # when z is normal, untran_z == z
            untran_z = qz.sample((n_samples,))
            z = self.z_encoder.z_transformation(untran_z)

        return dict(d=d, qz=qz, z=z)

    @auto_move_data
    def generative(
        self,
        z,
        qz_m,
        batch_index,
        cont_covs=None,
        cat_covs=None,
        use_z_mean=False,
    ):
        """Runs the generative model."""

        if cat_covs is not None:
            categorical_input = torch.split(cat_covs, 1, dim=1)
        else:
            categorical_input = tuple()

        latent = z if not use_z_mean else qz_m
        if cont_covs is None:
            decoder_input = latent
        elif latent.dim() != cont_covs.dim():
            decoder_input = torch.cat(
                [latent, cont_covs.unsqueeze(0).expand(latent.size(0), -1, -1)], dim=-1
            )
        else:
            decoder_input = torch.cat([latent, cont_covs], dim=-1)

        p = self.z_decoder(decoder_input, batch_index, *categorical_input)

        return dict(p=p)

    def loss(
        self, tensors, inference_outputs, generative_outputs, kl_weight: float = 1.0
    ):
        x = tensors[REGISTRY_KEYS.X_KEY]
        qz = inference_outputs["qz"]
        d = inference_outputs["d"]
        p = generative_outputs["p"]

        kld = kl_divergence(
            qz,
            Normal(0, 1),
        ).sum(dim=1)

        f = torch.sigmoid(self.region_factors) if self.region_factors is not None else 1
        rl = self.get_reconstruction_loss(p, d, f, x)

        loss = (rl.sum() + kld * kl_weight).sum()

        return LossRecorder(loss, rl, kld, kl_global=torch.tensor(0.0))


    def get_loadings(self) -> np.ndarray:
        """Extract per-gene weights (for each Z, shape is genes by dim(Z)) in the GUMBEL decoder."""
        # This is BW, where B is diag(b) batch norm, W is weight matrix
        if self.use_batch_norm is True:
            w = self.decoder.z_decoder.px_decoder[0][0].weight
            bn = self.decoder.factor_regressor.px_decoder[0][1]
            sigma = torch.sqrt(bn.running_var + bn.eps)
            gamma = bn.weight
            b = gamma / sigma
            b_identity = torch.diag(b)
            loadings = torch.matmul(b_identity, w)
        else:
            loadings = self.decoder.factor_regressor.px_decoder[0][0].weight
        loadings = loadings.detach().cpu().numpy()
        if self.n_batch > 1:
            loadings = loadings[:, : -self.n_batch]

        return loadings
    

