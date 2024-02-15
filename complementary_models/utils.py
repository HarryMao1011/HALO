from torch import Tensor
import numpy as np
import torch
import anndata
import pandas as pd

def torch_kernel(x, xKern, theta):
    # KERNEL Compute the rbf kernel
    ## theta
    n2 = torch_dist2(x, xKern)
    
    wi2 = theta[0] / 2
    kx = theta[1] * torch.exp(-n2 * wi2)
    bw_new = 1 / theta[0]
    return kx, bw_new


def torch_dist2(x, c, device='cuda'):
    # DIST2	Calculates squared distance between two sets of points.
    #
    # Description
    # D = DIST2(X, C) takes two matrices of vectors and calculates the
    # squared Euclidean distance between them.  Both matrices must be of
    # the same column dimension.  If X has M rows and N columns, and C has
    # L rows and N columns, then the result has M rows and L columns.  The
    # I, Jth entry is the  squared distance from the Ith row of X to the
    # Jth row of C.
    #
    # See also
    # GMMACTIV, KMEANS, RBFFWD
    #

    ndata, dimx = x.shape
    ncentres, dimc = c.shape
    if (dimx != dimc):
        raise Exception('Data dimension does not match dimension of centres')

   
    # ncen_ones = torch.ones(ncentres, 1).to(device=device)


    n2 =  torch.transpose((torch.ones(ncentres, 1).to(device=device) * torch.sum(torch.transpose(torch.mul(x, x), 0 ,1), axis=0)) ,0,1)+ \
         torch.ones((ndata, 1)).to(device=device) * torch.sum(torch.transpose(torch.mul(c, c), 0 ,1), axis=0) - \
         2 * torch.matmul(x, torch.transpose(c, 0, 1)) 
    # print("n2 device {}".format(n2.get_device()))     



    n2 = torch.nn.functional.relu(n2, inplace=True)


    return n2  

def torch_infer_nonsta_dir(X, Y, c_indx, width=0.1, IF_GP=False, device='cuda'):
    """learn the nonstationary driving force of the causal mechanism
    X: parents; Y; effect
    width: the kernel width for X and Y
    c_indx: surrogate variable to capture the distribution shift; 
    If If_GP = True (TODO), learning the kernel width for P(Y|X). Set it to False can speed up the process!!!
    """
    width = 0.1
    T,d = X.shape
    ## normalization
    X = X - torch.mean(X, 0)
    X = X / torch.std(X, axis=0, unbiased=True)
    
    # X = torch.matmul(X, np.diag(1./np.std(X, axis=0, ddof=1)))
    ## normalization
    Y = Y - torch.mean(Y, 0)
    Y = Y / torch.std(Y, axis=0, unbiased=True)

    theta = 1/width**2
    Lambda = 1
    Wt = 1
    theta = 1/width**2
    Kxx, _ = torch_kernel(X,X, [theta,1])
    Kxx , _ = torch_kernel(X, X, [theta,1])
    Kyy, _ = torch_kernel(Y, Y, [theta,1])
    Ktt, _ = torch_kernel(c_indx, c_indx, [1/Wt**2,1])


    invK = torch.inverse(torch.mul(Kxx,Ktt) +  Lambda * torch.eye(T).to(device=device))

    Kxx3 = torch.matmul(torch.matmul(Kxx, Kxx),Kxx)
    prod_invK =  torch.matmul(torch.matmul(invK, Kyy), invK)
    ###now finding Ml
    Ml = torch.matmul(torch.matmul(1/T**2*Ktt, torch.mul(Kxx3, prod_invK)), Ktt).to(dtype=torch.float64)
    # print("Ml dtype is {}".format(Ml.dtype))

    ###the square distance
    D = torch.matmul(torch.diag(torch.diag(Ml)),torch.ones(Ml.shape, dtype=torch.float64).to(device=device)) + torch.matmul(torch.ones(Ml.shape, dtype=torch.float64).to(device=device), torch.diag(torch.diag(Ml))) - 2*Ml
    # print("D dtype is {}".format(D.dtype))

    DD = D[torch.where(torch.tril(torch.ones(D.shape, dtype=float).to(device=device),-1) != 0)]
    sigma2_square = torch.quantile(DD, 0.5)
    Mg = torch.exp(-D/sigma2_square/2)


    invK2 = torch.inverse(Ktt +  Lambda * torch.eye(T).to(device=device))
    Ml2 = torch.matmul(Ktt, invK2)
    Ml2 = torch.matmul(Ml2, Kxx)
    Ml2 = torch.matmul(Ml2, invK2)
    Ml2 = torch.matmul(Ml2, Ktt).to(dtype=torch.float64)
    # print("Ml2 dtype is {}".format(Ml2.dtype))

    D2 =torch.matmul(torch.diag(torch.diag(Ml2).to(device=device)),torch.ones(Ml2.shape, dtype=torch.float64).to(device=device))\
         + torch.matmul(torch.ones(Ml2.shape, dtype=torch.float64).to(device=device), torch.diag(torch.diag(Ml2))) - 2*Ml2
    # % Gaussian kernel
    DD2 = D2[torch.where(torch.tril(torch.ones(D2.shape).to(device=device),-1) != 0)]
    sigma2_square2 = torch.quantile(DD2, 0.5)

    # sigma2_square2 = median( D2(find(tril(ones(size(D2)),-1))) );
    Mg2 = torch.exp(-D2/sigma2_square2/2)

    # %% 
    H = torch.eye(T,dtype=torch.float64).to(device=device)-1/T*torch.ones([T,T], dtype=torch.float64).to(device=device)
    Mg = torch.matmul(torch.matmul(H, Mg), H)
    Mg2 = torch.matmul(torch.matmul(H, Mg2), H)
    testStat = 1/T**2*torch.sum(torch.sum(torch.mul(Mg, Mg2)))


    return testStat, Mg, Mg2



def split_rna(adata_mvi, modality="modality"):
    rna_index = adata_mvi.var[modality] =="Gene Expression"
    rna_data = anndata.AnnData(X=adata_mvi.X[:, rna_index],
    obs=adata_mvi.obs, 
    var=adata_mvi.var[rna_index])
    return rna_data

def split_atac(adata_mvi,  modality="modality"):
    atac_index = adata_mvi.var[modality] =="Peaks"
    atac_data = anndata.AnnData(X=adata_mvi.X[:, atac_index],
                            obs=adata_mvi.obs, 
                            var=adata_mvi.var[atac_index])
    return atac_data

def split_atac_rna(adata_mvi):
    return split_atac(adata_mvi),split_rna(adata_mvi)                    


def parsing_chr_se(row):
    row = row['gene_ids']
    row = row.split("\t")[0]
    chrome = row[: row.find(":")]
    start = row[row.find(":")+1: row.find("-")]
    end = row[row.find("-")+1 :]
    newcol = pd.Series([chrome, start, end]) 
    return newcol

def reindex_atac(atac_data):

    atac_data.var["peak_id"] = np.arange(1, atac_data.var.shape[0]+1)
    atac_data.var[['chr', 'start', 'end']] =  atac_data.var.apply(parsing_chr_se, axis=1) 
    atac_data.var = atac_data.var[['peak_id', 'chr', 'start', 'end', "n_cells"]]
    atac_data.var.reset_index(drop=True)
    atac_data.var.set_index("peak_id",inplace=True)
    return atac_data   

def generate_peakid(row):
    chr = row['chr']
    start = row["start"]
    end = row["end"]
    peak = chr+'_' + start+ '_' + end
    return peak

def merge_atac_annotation(annotation, atac_data):
    atac_annotation = pd.read_csv(annotation, sep='\t')
    atac_data.var['peak'] =  atac_data.var.apply(generate_peakid, axis=1)
    new_var = pd.merge(atac_data.var, atac_annotation, left_on='peak', right_on='peak')
    atac_data.var = new_var

    return atac_data


    