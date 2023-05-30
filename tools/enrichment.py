import tools.enrichr_enrichments as enrichr
from tools.plots.enrichment_plot import plot_enrichments as mira_plot_enrichments
import numpy as np



def _argsort_genes(latent_num, loadings):
        
        return np.argsort(loadings[latent_num, :])

def get_top_genes(top_num, loadingmatrix, latent_index, rnadata, colname = "gene_short_name"):
    gene_index = _argsort_genes(latent_index,  loadings=loadingmatrix)[-top_num : ]
    gene_name = rnadata.var[colname][gene_index]
    return gene_name.tolist()


def post_topic(latent_index, loadings, rnadata, colname, top_n = 500, min_genes = 200, max_genes = 600, enrichments={}):
    '''
    Post the top genes from topic to Enrichr for geneset enrichment analysis.
    Parameters
    ----------
    topic_num : int
        Topic number to post geneset
    top_n : int, default = 500
        Number of genes to post
    min_genes : int > 0
        If top_n is None, all activations (distributed standard normal) 
        greater than 3 will be posted. If this is less than **min_genes**,
        then **min_genes** will be posted.
    max_genes : int > 0
        If top_n is None, a maximum of **max_genes** will be posted.
    Examples
    --------
    .. code-block:: python
        >>> rna_model.post_topic(10, top_n = 500)
        >>> rna_model.post_topic(10)
    '''

    list_id = enrichr.post_genelist(
        get_top_genes(top_num = top_n, loadingmatrix=loadings, latent_index=latent_index, rnadata=rnadata, colname=colname)
    )

    return dict(
        list_id = list_id,
        results = {}
    )

def get_post_genelist(genelist):
    list_id = enrichr.post_genelist(genelist)

    return dict(
        list_id = list_id,
        results = {}
    )

def post_topics(num_latent, latent_index, loadings,  rnadata, colname,  top_n = 500, min_genes = 200, max_genes = 600):
    '''
    Iterate through all topics and post top genes to Enrichr.
    Parameters
    ----------
    top_n : int, default = 500
        Number of genes to post
    min_genes : int > 0
        If top_n is None, all activations (distributed standard normal) 
        greater than 3 will be posted. If this is less than **min_genes**,
        then **min_genes** will be posted.
    max_genes : int > 0
        If top_n is None, a maximum of **max_genes** will be posted.
    Examples
    --------
    .. code-block:: python
        >>> rna_model.post_topics()
    '''
    enrichments = {}
    for i in range(num_latent):
        enrichments[str(i)] = post_topic(i,latent_index, loadings,  rnadata, colname, top_n = top_n, min_genes = min_genes, max_genes = max_genes)

    return enrichments    


def fetch_topic_enrichments(enrichments, latent_index, ontologies = enrichr.LEGACY_ONTOLOGIES):
    '''
    Fetch Enrichr enrichments for a topic. Will return results for the ontologies listed.
    Parameters
    ----------
    topic_num : int
        Topic number to fetch enrichments
    ontologies : list[str], default = mira.tools.enrichr_enrichments.LEGACY_ONTOLOGIES
        List of ontology names from which to retrieve results. May provide
        a list of any onlogies hosted on Enrichr.
    Examples
    --------
    .. code-block:: python
        >>> rna_model.fetch_topic_enrichments(10, ontologies = ['WikiPathways_2019_Mouse'])        
    '''

    try:
        enrichments
    except AttributeError:
        raise AttributeError('User must run "post_topic" or "post_topics" before getting enrichments')

    try:
        list_id = enrichments[latent_index]['list_id']
    except (KeyError, IndentationError):
        raise KeyError('User has not posted topic yet, run "post_topic" first.')

    enrichments[latent_index]['results'].update(
        enrichr.fetch_ontologies(list_id, ontologies = ontologies)
    )
    #for ontology in ontologies:
        #self.enrichments[topic_num]['results'].update(enrichr.get_ontology(list_id, ontology=ontology))

def fetch_genelist_enrichments(enrichment, ontologies = enrichr.LEGACY_ONTOLOGIES):
    '''
    Fetch Enrichr enrichments for a topic. Will return results for the ontologies listed.
    Parameters
    ----------
    topic_num : int
        Topic number to fetch enrichments
    ontologies : list[str], default = mira.tools.enrichr_enrichments.LEGACY_ONTOLOGIES
        List of ontology names from which to retrieve results. May provide
        a list of any onlogies hosted on Enrichr.
    Examples
    --------
    .. code-block:: python
        >>> rna_model.fetch_topic_enrichments(10, ontologies = ['WikiPathways_2019_Mouse'])        
    '''

    try:
        enrichment
    except AttributeError:
        raise AttributeError('User must run "post_topic" or "post_topics" before getting enrichments')

    try:
        list_id = enrichment['list_id']
    except (KeyError, IndentationError):
        raise KeyError('User has not posted topic yet, run "post_topic" first.')

    enrichment['results'].update(
        enrichr.fetch_ontologies(list_id, ontologies = ontologies)
    )
    return enrichment





def get_enrichments(enrichments, latent_index):
    '''
    Return the enrichment results for a  given topic.
    Paramters
    ---------
    topic_num : int
        Topic for which to return enrichment results
    
    Returns
    -------
    enrichments : dict
        Dictionary with schema:
        .. code-block::
            
            {
                <ontology> : {
                    [
                        {'rank' : <rank>,
                        'term' : <term>,
                        'pvalue' : <pval>,
                        'zscore': <zscore>,
                        'combined_score': <combined_score>,
                        'genes': [<gene1>, ..., <geneN>],
                        'adj_pvalue': <adj_pval>},
                        ...,
                    ]
                }
            }   
            
    '''

    try:
        return enrichments[latent_index]['results']
    except (KeyError, IndentationError):
        raise KeyError('User has not posted topic yet, run "post_topic" first.')     


def plot_enrichments(enrichments, topic_num, show_genes = True, show_top = 10, barcolor = 'lightgrey', label_genes = [],
        text_color = 'black', plots_per_row = 2, height = 4, aspect = 2.5, max_genes = 15, pval_threshold = 1e-5,
        color_by_adj = True, palette = 'Reds', gene_fontsize=10):

    '''
    Make plot of geneset enrichments results.
    Parameters
    ----------
    topic_num : int
        Topic for which to plot results
    show_genes : boolean, default = True
        Whether to show gene names on enrichment barplot bars
    show_top : int > 0, default = 10
        Plot this many top terms for each ontology
    barcolor : str or tuple[int] (r,g,b,a) or tuple[int] (r,g,b)
        Color of barplot bars
    label_genes : list[str] or np.ndarray[str]
        Add an asterisc by the gene name of genes in this list. Useful for
        finding transcription factors or signaling factors of interest in
        enrichment results.
    text_color : str or tuple[int] (r,g,b,a) or tuple[int] (r,g,b)
        Color of text on plot
    plots_per_row : int > 0, default = 2
        Number of onotology plots per row in figure
    height : float > 0, default = 4
        Height of each ontology plot
    aspect : float > 0, default = 2.5
        Aspect ratio of ontology plot
    max_genes : int > 0, default = 15
        Maximum number of genes to plot on each term bar
    pval_threshold : float (0, 1), default = 1e-5
        Upper bound on color map for adjusted p-value coloring of bar
        outlines.
    color_by_adj : boolean, default = True
        Whether to outline term bars with adjusted p-value
    palette : str
        Color palette for adjusted p-value
    gene_fontsize : float > 0, default = 10
        Fontsize of gene names on term bars
    Returns
    -------
    ax : matplotlib.pyplot.axes
    Examples
    --------
    .. code-block:: python
        >>> rna_model.post_topic(13, 500)
        >>> rna_model.fetch_topic_enrichments(13, 
        ... ontologies=['WikiPathways_2019_Mouse','BioPlanet_2019'])
        >>> rna_model.plot_enrichments(13, height=4, show_top=6, max_genes=10, 
        ... aspect=2.5, plots_per_row=1)
    .. image:: /_static/mira.topics.ExpressionTopicModel.plot_enrichments.svg
        :width: 1200
    '''

    try:
        enrichments
    except AttributeError:
        raise AttributeError('User must run "post_topic" or "post_topics" before getting enrichments')

    try:
        results = enrichments[topic_num]['results']
    except (KeyError, IndentationError):
        raise KeyError('User has not posted topic yet, run "post_topic" first.')

    if len(results) == 0:
        raise Exception('No results for this topic, user must run "get_topic_enrichments" or "get_enrichments" before plotting.')

    return mira_plot_enrichments(results, text_color = text_color, label_genes = label_genes,
        show_top = show_top, barcolor = barcolor, show_genes = show_genes, max_genes = max_genes,
        plots_per_row = plots_per_row, height = height, aspect = aspect, pval_threshold = pval_threshold,
        palette = palette, color_by_adj = color_by_adj, gene_fontsize = gene_fontsize)


def plot_enrichments_genesets(results, show_genes = True, show_top = 10, barcolor = 'lightgrey', label_genes = [],
        text_color = 'black', plots_per_row = 2, height = 4, aspect = 2.5, max_genes = 15, pval_threshold = 1e-5,
        color_by_adj = True, palette = 'Reds', gene_fontsize=10):

    '''
    Make plot of geneset enrichments results.
    Parameters
    ----------
    topic_num : int
        Topic for which to plot results
    show_genes : boolean, default = True
        Whether to show gene names on enrichment barplot bars
    show_top : int > 0, default = 10
        Plot this many top terms for each ontology
    barcolor : str or tuple[int] (r,g,b,a) or tuple[int] (r,g,b)
        Color of barplot bars
    label_genes : list[str] or np.ndarray[str]
        Add an asterisc by the gene name of genes in this list. Useful for
        finding transcription factors or signaling factors of interest in
        enrichment results.
    text_color : str or tuple[int] (r,g,b,a) or tuple[int] (r,g,b)
        Color of text on plot
    plots_per_row : int > 0, default = 2
        Number of onotology plots per row in figure
    height : float > 0, default = 4
        Height of each ontology plot
    aspect : float > 0, default = 2.5
        Aspect ratio of ontology plot
    max_genes : int > 0, default = 15
        Maximum number of genes to plot on each term bar
    pval_threshold : float (0, 1), default = 1e-5
        Upper bound on color map for adjusted p-value coloring of bar
        outlines.
    color_by_adj : boolean, default = True
        Whether to outline term bars with adjusted p-value
    palette : str
        Color palette for adjusted p-value
    gene_fontsize : float > 0, default = 10
        Fontsize of gene names on term bars
    Returns
    -------
    ax : matplotlib.pyplot.axes
    Examples
    --------
    .. code-block:: python
        >>> rna_model.post_topic(13, 500)
        >>> rna_model.fetch_topic_enrichments(13, 
        ... ontologies=['WikiPathways_2019_Mouse','BioPlanet_2019'])
        >>> rna_model.plot_enrichments(13, height=4, show_top=6, max_genes=10, 
        ... aspect=2.5, plots_per_row=1)
    .. image:: /_static/mira.topics.ExpressionTopicModel.plot_enrichments.svg
        :width: 1200
    '''


    results = results['results']

    return mira_plot_enrichments(results, text_color = text_color, label_genes = label_genes,
        show_top = show_top, barcolor = barcolor, show_genes = show_genes, max_genes = max_genes,
        plots_per_row = plots_per_row, height = height, aspect = aspect, pval_threshold = pval_threshold,
        palette = palette, color_by_adj = color_by_adj, gene_fontsize = gene_fontsize)                       