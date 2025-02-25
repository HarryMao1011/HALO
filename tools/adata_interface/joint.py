from .topic_model import fetch_topic_comps
import pandas as pd


def fetch_obsms(self, adata1, adata2,*,key):

    return {
        'x' : fetch_topic_comps(self, adata1, key = key)['topic_compositions'],
        'y' : fetch_topic_comps(self, adata2, key = key)['topic_compositions'],
    }


def format_corr_dataframe(adata, output,*,
    adata1_name, adata2_name):

    corrmatrix = output

    df = pd.DataFrame(
        corrmatrix,
        index = ['Topic ' + str(i) for i in range(corrmatrix.shape[0])],
        columns = ['Topic ' + str(i) for i in range(corrmatrix.shape[1])]
    )

    df.index.name = adata1_name
    df.columns.name = adata2_name

    return df