import numpy as np
import scanpy as sc
import anndata
import pandas as pd
import logging
import mira
import warnings
from scipy.sparse import csr_matrix
from scipy.sparse import diags
from scipy.interpolate import make_interp_spline
import matplotlib.pyplot as plt
from causal_utils import infer_nonsta_dir



### The function of getting the normalized atac seq data



def tfidf_norm(adata_atac, scale_factor=1e4):
    """TF-IDF normalization.
    This function normalizes counts in an AnnData object with TF-IDF.
    Parameters
    ----------
    adata_atac: :class:`~anndata.AnnData`
        ATAC anndata object.
    scale_factor: `float` (default: 1e4)
        Value to be multiplied after normalization.
    copy: `bool` (default: `False`)
        Whether to return a copy or modify `.X` directly.
    Returns
    -------
    If `copy==True`, a new ATAC anndata object which stores normalized counts in `.X`.
    """
    npeaks = adata_atac.X.sum(1)
    npeaks_inv = csr_matrix(1.0/npeaks)
    tf = adata_atac.X.multiply(npeaks_inv)
    idf = diags(np.ravel(adata_atac.X.shape[0] / adata_atac.X.sum(0))).log1p()
    new_X = tf.dot(idf) * scale_factor
    return new_X


def aggregate_local_atac(table, norm_X, atac_data):
    indices = table.index.to_numpy()
    col_index = [atac_data.var.index.get_loc(e) for e in indices]
    gene_peaks = norm_X[:, col_index]
    gene_peaks = np.sum(gene_peaks, axis=1)
    # print(gene_peaks)
    
    return gene_peaks

def get_gene_local_atac(genename, litemodel, rna_data, atac_data, norm_X):
    table = litemodel[genename].get_influential_local_peaks(atac_data, decay_periods = .8)
    peaks = aggregate_local_atac(table, norm_X,atac_data)
    rna_data.obs["local_peaks"] = peaks

def get_gene_local_atac_mtx(genenames, litemodel, rna_data, atac_data, norm_X):
    for genename in genenames:
        table = litemodel[genename].get_influential_local_peaks(atac_data, decay_periods = .8)
        peaks = aggregate_local_atac(table, norm_X)
        name = genename+"local_peaks"
        rna_data.obs[name] = peaks


def get_peaks(rna_data, atac_data, litemodel, decay_periods=.8):
    norm_X = atac_data.layers["norm"]
    cellnum =rna_data.shape[0]
    genenames = rna_data.var.index.tolist()
    peaks = []
    for gene in genenames:
        try:
            table = litemodel[gene].get_influential_local_peaks(atac_data, decay_periods = decay_periods)
            peak = aggregate_local_atac(table, norm_X, atac_data)
        except:
            peak = [0]*cellnum
        peaks.append(peak)
        # print(len(peaks))
    peaks  = np.array(peaks).transpose()
    peaks = csr_matrix(peaks)
    rna_data.layers["peaks"] = peaks    




def get_norm_gene_local(genename,time, rna_data):
    peak_mean = rna_data.obs.groupby(time)["local_peaks"].mean()
    rna_data.obs["temp_gene"] =  rna_data[:,[genename.capitalize()]].X.toarray()
    gene_mean = rna_data.obs.groupby(time)["temp_gene"].mean()
    timelist = rna_data.obs.time.unique().tolist()
    genesplie = make_interp_spline(np.array(timelist), gene_mean.value)

    df = pd.DataFrame([peak_mean, gene_mean, genesplie]).T
    df = df.reset_index()
    fig, ax = plt.subplots()
    df.plot(x=time, y = "local_peaks", ax=ax)
    df.plot(x=time, y = "temp_gene", ax=ax,secondary_y=True, color="r")



def smooth_gene_peaks_plot(genename,time, linewidth, rna_data, points=1000, save=False, legend=True):
    peak_mean = rna_data.obs.groupby(time)["local_peaks"].mean()
    rna_data.obs["temp_gene"] =   rna_data[:,[genename.upper()]].X.toarray()
    gene_mean = rna_data.obs.groupby(time)["temp_gene"].mean()
    timelist = rna_data.obs[time].unique()
    timelist = timelist.astype(np.float64)
    timelist = sorted(timelist)
    genesplie = make_interp_spline(np.array(timelist), gene_mean.values)
    peaksplie = make_interp_spline(np.array(timelist), peak_mean.values)
    X_ = np.linspace(timelist[0], timelist[-1], points)
    smooth_gene = genesplie(X_)
    smooth_peaks = peaksplie(X_)
    df = pd.DataFrame(index=X_)
    df["smooth_gene"] = smooth_gene
    df["smooth_peaks"] = smooth_peaks
    df = df.reset_index() 
    fig, ax = plt.subplots()
    # ax.grid(True)
        
    df.plot(x="index", y = "smooth_peaks", ax=ax, linewidth=linewidth, legend=legend)
    df.plot(x="index", y = "smooth_gene", ax=ax,secondary_y=True, color="r", linewidth=linewidth, legend=legend)
    if save:
        fig.savefig(save)


def smooth_gene_peaks_plot(genename,time, linewidth, rna_data, points=1000, save=False, legend=True):
    peak_mean = rna_data.obs.groupby(time)["local_peaks"].mean()
    rna_data.obs["temp_gene"] =   rna_data[:,[genename.upper()]].X.toarray()
    gene_mean = rna_data.obs.groupby(time)["temp_gene"].mean()
    timelist = rna_data.obs[time].unique()
    timelist = timelist.astype(np.float64)
    timelist = sorted(timelist)
    genesplie = make_interp_spline(np.array(timelist), gene_mean.values)
    peaksplie = make_interp_spline(np.array(timelist), peak_mean.values)
    X_ = np.linspace(timelist[0], timelist[-1], points)
    smooth_gene = genesplie(X_)
    smooth_peaks = peaksplie(X_)
    df = pd.DataFrame(index=X_)
    df["smooth_gene"] = smooth_gene
    df["smooth_peaks"] = smooth_peaks
    df = df.reset_index() 
    fig, ax = plt.subplots()
    # ax.grid(True)
        
    df.plot(x="index", y = "smooth_peaks", ax=ax, linewidth=linewidth, legend=legend)
    df.plot(x="index", y = "smooth_gene", ax=ax,secondary_y=True, color="r", linewidth=linewidth, legend=legend)
    if save:
        fig.savefig(save)        



def gene_level_casual(genename,time, litemodel,rna_data,atac_data, norm_X, width=0.1):
    get_gene_local_atac(genename, litemodel, rna_data, atac_data, norm_X)
    # peak_mean = rna_data.obs.groupby(time)["local_peaks"].mean().values
    rna_data.obs["temp_gene"] =   rna_data[:,[genename.upper()]].X.toarray()
    peaks = rna_data.obs["local_peaks"]
    genes = rna_data.obs["temp_gene"]
    times = rna_data.obs[time]
    # print("peaks shap {}, genes {}, time {}".format(peaks.shape, genes.shape, times.shape))
    gene_mean = genes
    peak_mean = peaks
    timelist = times
    # gene_mean = rna_data.obs.groupby(time)["temp_gene"].mean().values
    gene_mean = np.expand_dims(gene_mean, axis=1)
    peak_mean = np.expand_dims(peak_mean, axis=1)
    # timelist = rna_data.obs[time].unique()
    timelist = timelist.astype(np.float64)
    timelist = sorted(timelist)
    timelist = np.expand_dims(timelist, axis=1)

    score_ar, _, _ = infer_nonsta_dir(peak_mean, gene_mean, timelist, width=0.1)
    score_ra, _, _ =infer_nonsta_dir(gene_mean, peak_mean, timelist, width=0.1)
    # score_ar = np.log(score_ar)
    # score_ra = np.log(score_ra)
    score = score_ar - score_ra

    return score, score_ar, score_ra



def gene_level_casual_nonzero(genename,time, litemodel,rna_data,atac_data, norm_X, width=0.1, thresh=0.001):
    get_gene_local_atac(genename, litemodel, rna_data, atac_data, norm_X)
    # peak_mean = rna_data.obs.groupby(time)["local_peaks"].mean().values
    rna_data.obs["temp_gene"] =   rna_data[:,[genename.upper()]].X.toarray()
    peaks = rna_data.obs[rna_data.obs.local_peaks>0]['local_peaks']
    genes = rna_data.obs[rna_data.obs.local_peaks>0]["temp_gene"]
    times = rna_data.obs[rna_data.obs.local_peaks>0][time]
    gene_mean = genes
    peak_mean = peaks
    timelist = times
    # gene_mean = rna_data.obs.groupby(time)["temp_gene"].mean().values
    gene_mean = np.expand_dims(gene_mean, axis=1)
    peak_mean = np.expand_dims(peak_mean, axis=1)
    # timelist = rna_data.obs[time].unique()
    timelist = timelist.astype(np.float64)
    timelist = sorted(timelist)
    timelist = np.expand_dims(timelist, axis=1)

    score_ar, _, _ = infer_nonsta_dir(peak_mean, gene_mean, timelist, width=width)
    score_ra, _, _ =infer_nonsta_dir(gene_mean, peak_mean, timelist, width=width)
    # score_ar = np.log(score_ar)
    # score_ra = np.log(score_ra)
    # score = -(score_ar - score_ra)
    score = -(score_ar - score_ra)
    couple_score =  score_ar - score_ra - thresh

    return score, score_ar, score_ra, couple_score


def smooth_gene_peaks_plot(genename,time, rna_data, linewidth, points=1000, save=False, legend=True):
    peak_mean = rna_data.obs.groupby(time)["local_peaks"].mean()
    rna_data.obs["temp_gene"] =   rna_data[:,[genename.upper()]].X.toarray()
    gene_mean = rna_data.obs.groupby(time)["temp_gene"].mean()
    timelist = rna_data.obs[time].unique()
    timelist = timelist.astype(np.float64)
    timelist = sorted(timelist)
    genesplie = make_interp_spline(np.array(timelist), gene_mean.values)
    peaksplie = make_interp_spline(np.array(timelist), peak_mean.values)
    X_ = np.linspace(timelist[0], timelist[-1], points)
    smooth_gene = genesplie(X_)
    smooth_peaks = peaksplie(X_)
    df = pd.DataFrame(index=X_)
    df["smooth_gene"] = smooth_gene
    df["smooth_peaks"] = smooth_peaks
    df = df.reset_index() 
    fig, ax = plt.subplots()
    # ax.grid(True)
        
    df.plot(x="index", y = "smooth_peaks", ax=ax, linewidth=linewidth, legend=legend)
    df.plot(x="index", y = "smooth_gene", ax=ax,secondary_y=True, color="r", linewidth=linewidth, legend=legend)
    if save:
        fig.savefig(save)



def smooth_gene_peaks_plot_interv(genename,time, rna_data, linewidth, points=1000, save=False, legend=True, ticks=True, gcolor="b", pcolor="r", shade=0.3, title=False, gquant=0.6, pquant=0.6):
    localpeaks ="local_peaks"
    peak_mean = rna_data.obs[rna_data.obs.local_peaks>0].groupby(time)["local_peaks"].median().fillna(0)
    peak_std =  rna_data.obs[rna_data.obs.local_peaks>0].groupby(time)["local_peaks"].std()
    peak_max = rna_data.obs[rna_data.obs.local_peaks>0].groupby(time)[localpeaks].quantile(pquant).fillna(0)
    peak_min = rna_data.obs[rna_data.obs.local_peaks>0].groupby(time)[localpeaks].quantile(1- pquant).fillna(0)
    localpeaks = "local_peaks"
    rna_data.obs["temp_gene"] =   rna_data[:,[genename]].X.toarray()
    gene_mean = rna_data.obs[rna_data.obs.temp_gene>0].groupby(time)["temp_gene"].median().fillna(0)
    gene_std = rna_data.obs.groupby(time)["temp_gene"].std()

    # gene_max = rna_data.obs.groupby(time).apply(lambda g: g[g.temp_gene > g.temp_gene.quantile(.2)]).temp_gene.to_frame().groupby(time).mean().temp_gene.to_numpy() 
    # gene_min = rna_data.obs.groupby(time).apply(lambda g: g[g.temp_gene <= g.temp_gene.quantile(.9)]).temp_gene.to_frame().groupby(time).mean().temp_gene.to_numpy()

    gene_max = rna_data.obs[rna_data.obs.temp_gene>0].groupby(time)['temp_gene'].quantile(gquant).fillna(0)
    gene_min = rna_data.obs[rna_data.obs.temp_gene>0].groupby(time)['temp_gene'].quantile(1 - gquant).fillna(0)


    timelist = rna_data.obs[time].unique()
    timelist = timelist.astype(np.float64)
    timelist = sorted(timelist)

    genesplie = make_interp_spline(np.array(timelist), gene_mean.values)
    peaksplie = make_interp_spline(np.array(timelist), peak_mean.values)
    X_ = np.linspace(timelist[0], timelist[-1], points)
    smooth_gene = genesplie(X_)
    smooth_peaks = peaksplie(X_)



    genesplie_max = make_interp_spline(np.array(timelist), gene_max)
    peaksplie_max = make_interp_spline(np.array(timelist), peak_max)
    smooth_gene_max = genesplie_max(X_)
    smooth_peaks_max = peaksplie_max(X_)
    

    # genesplie_min = make_interp_spline(np.array(timelist), gene_min.values)
    # peaksplie_min = make_interp_spline(np.array(timelist), peak_min.values)
    # smooth_gene_min = genesplie_min(X_)
    # smooth_peaks_min = peaksplie_min(X_)


    # genesplie_std = make_interp_spline(np.array(timelist), gene_std.values)
    # peaksplie_std = make_interp_spline(np.array(timelist), peak_std.values)
    # smooth_gene_std = genesplie_std(X_)
    # smooth_peaks_std = peaksplie_std(X_)
   

    genesplie_min = make_interp_spline(np.array(timelist), gene_min)
    peaksplie_min = make_interp_spline(np.array(timelist), peak_min)
    smooth_gene_min = genesplie_min(X_)
    smooth_peaks_min = peaksplie_min(X_)


    # print(smooth_gene_max)
    # print("+++++++++++")
    # print(smooth_gene_min)
    # print("++++++++++")
    # print(smooth_gene)

    fig, ax = plt.subplots()
    # ax.plot(X_,smooth_gene)
    # ax.fill_between(X_, smooth_gene_min, smooth_gene_max, color='b', alpha=.5)
    ax.axis('off')

    if title == True:
        ax.set_xlabel('latent time')
        ax.set_ylabel('gene expression', color=gcolor)
    ax.plot(X_,smooth_gene, color=gcolor)
    ax.fill_between(X_, smooth_gene_min, smooth_gene_max, color=gcolor, alpha=shade)
    if ticks == False:
        ax.set_yticks([])
        ax.set_xticks([])
    if  title==False:
        ax.set_xticklabels([])
        ax.set_yticklabels([])     
     


    # ax.fill_between(X_, smooth_gene-smooth_gene_std, smooth_gene+smooth_gene_std, color='b', alpha=.1)


    ax2 = ax.twinx()
    ax2.axis('off')

    if title ==True:
        ax2.set_ylabel('normalized peaks', color=pcolor)
    ax2.plot(X_,smooth_peaks, color=pcolor)
    # ax2.fill_between(X_, smooth_peaks-smooth_peaks_std, smooth_peaks + smooth_peaks_std, color='r', alpha=.1)
    ax2.fill_between(X_, smooth_peaks_min, smooth_peaks_max, color=pcolor, alpha=shade)
    if ticks==False:
        ax2.set_yticks([])
        ax2.set_xticks([])
    if  title==False:
        ax2.set_xticklabels([]) 
        ax2.set_yticklabels([])     


    # df = pd.DataFrame(index=X_)
    # df["smooth_gene"] = smooth_gene
    # df["smooth_peaks"] = smooth_peaks
    # df = df.reset_index() 
    # fig, ax = plt.subplots()
    # # ax.grid(True)
        
    # df.plot(x="index", y = "smooth_peaks", ax=ax, linewidth=linewidth, legend=legend)
    # df.plot(x="index", y = "smooth_gene", ax=ax,secondary_y=True, color="r", linewidth=linewidth, legend=legend)
    if save:
        fig.savefig(save, dpi=300)
        plt.close(fig)    # close the figure window



def smooth_gene_peaks_plot_interv_2(genename,time, litemodel, rna_data, atac_data, linewidth, norm_X,
points=1000, save=False, legend=True, ticks=True, gcolor="b", pcolor="r", shade=0.3, title=False):
    
    get_gene_local_atac(genename, litemodel, rna_data, atac_data, norm_X)
    localpeaks ="local_peaks"
    peak_mean = rna_data.obs.groupby(time)["local_peaks"].median()
    peak_max = rna_data.obs.groupby(time)[localpeaks].quantile(0.6)
    peak_min = rna_data.obs.groupby(time)[localpeaks].quantile(0.4)
    


    rna_data.obs["temp_gene"] =   rna_data[:,[genename]].X.toarray()
    gene_mean = rna_data.obs[rna_data.obs.temp_gene>=0].groupby(time)["temp_gene"].median()
    # gene_std = rna_data.obs.groupby(time)["temp_gene"].std()

    # gene_max = gene_mean + 0.5*gene_std
    # gene_min = gene_mean - 0.5*gene_std


    # gene_mean = rna_data.obs.groupby(time)["temp_gene"].median()
    gene_max = rna_data.obs.groupby(time)["temp_gene"].quantile(0.6)
    gene_min = rna_data.obs.groupby(time)["temp_gene"].quantile(0.4)
    # print(gene_mean.max())
    # print(gene_max)
    # print(gene_min)


    timelist = rna_data.obs[time].unique()
    timelist = timelist.astype(np.float64)
    timelist = sorted(timelist)
    genesplie = make_interp_spline(np.array(timelist), gene_mean.values)
    peaksplie = make_interp_spline(np.array(timelist), peak_mean.values)
    T = np.linspace(timelist[0], timelist[-1], points)
    smooth_gene = genesplie(T)
    smooth_peaks = peaksplie(T)



    genesplie_max = make_interp_spline(np.array(timelist), gene_max)
    peaksplie_max = make_interp_spline(np.array(timelist), peak_max)
    smooth_gene_max = genesplie_max(T)
    smooth_peaks_max = peaksplie_max(T)


    genesplie_min = make_interp_spline(np.array(timelist), gene_min)
    peaksplie_min = make_interp_spline(np.array(timelist), peak_min)
    smooth_gene_min = genesplie_min(T)
    smooth_peaks_min = peaksplie_min(T)


    smooth_gene[smooth_gene<0] = 0
    smooth_gene_min[smooth_gene_min<0] = 0
    smooth_gene_max[smooth_gene_max<0] = 0

    fig, ax = plt.subplots()

    if title == True:
        ax.set_xlabel('latent time')
        ax.set_ylabel('gene expression', color=gcolor)
    ax.plot(T,smooth_gene, color=gcolor)
    ax.fill_between(T, smooth_gene_min, smooth_gene_max, color=gcolor, alpha=shade)
    if ticks == False:
        ax.axis('off')

        ax.set_yticks([])
        ax.set_xticks([])
    if  title==False:
        ax.set_xticklabels([])
        ax.set_yticklabels([])     
    

    ax2 = ax.twinx()

    if title ==True:
        ax2.set_ylabel('normalized peaks', color=pcolor)
    ax2.plot(T,smooth_peaks, color=pcolor)
    # ax2.fill_between(T, smooth_peaks-smooth_peaks_std, smooth_peaks + smooth_peaks_std, color='r', alpha=.1)
    ax2.fill_between(T, smooth_peaks_min, smooth_peaks_max, color=pcolor, alpha=shade)
    if ticks==False:
        ax2.axis('off')

        ax2.set_yticks([])
        ax2.set_xticks([])
    if  title==False:
        ax2.set_xticklabels([]) 
        ax2.set_yticklabels([])     

    if save:
        fig.savefig(save, dpi=300)
        plt.close(fig)    # close the figure window



def plot_ratio(df, cols=["decouple_score"], name="value_ratios"):
    
    ratios = pd.DataFrame(columns=['Decoupled', 'Coupled'])

    for col in cols:
        gt0 = (df[col] > 0).sum()
        lt0 = (df[col] < 0).sum()
        total = gt0 + lt0
        if total > 0:
            ratios.loc[col] = [gt0 / total, lt0 / total]
        else:
            ratios.loc[col] = [0, 0]  # avoid division by zero

    # Plotting
    ratios.plot(kind='bar', figsize=(8, 6))
    plt.ylabel('Ratio')
    plt.title('Ratio of Decoupled and Coupled Gene-Peak Pairs')
    plt.ylim(0, 1)
    plt.tight_layout()
    # plt.show()
    plt.savefig("{}.png".format(name), dpi=300)
