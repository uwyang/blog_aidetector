import seaborn as sns
import pandas as pd

import matplotlib.pyplot as plt

def plot_tsne(df, hue='prompt_name', x_col='tsne_human_p30_1', y_col='tsne_human_p30_2', \
              title='t-SNE Embedding of Combined DataFrame', legend_pos='best', \
                hue_order=None, palette=None, hue_min=None, hue_max=None, s=3, markerscale = 6, alpha=0.7):
    fig, ax = plt.subplots(figsize=(12, 8))
    if palette is None: 
        palette = sns.color_palette("Set1") + sns.color_palette("Set2")
    
    if pd.api.types.is_numeric_dtype(df[hue]):
        norm = plt.Normalize(vmin=hue_min if hue_min is not None else df[hue].min(), vmax=hue_max if hue_max is not None else df[hue].max())
        sm = plt.cm.ScalarMappable(cmap=palette, norm=norm)
        sm.set_array([])
        sns.scatterplot(data=df, x=x_col, y=y_col, hue=hue, palette=palette, hue_norm=norm, alpha=alpha, s=s, ax=ax)
        cbar = fig.colorbar(sm, ax=ax)
    else:
        sns.scatterplot(data=df, x=x_col, y=y_col, hue=hue, alpha=alpha, s=s, palette=palette, hue_order=hue_order, ax=ax)
    
    ax.set_title(title, fontsize=20)
    ax.set_xlabel('t-SNE Dimension 1', fontsize=15)
    ax.set_ylabel('t-SNE Dimension 2', fontsize=15)
    ax.legend(title=hue, markerscale=markerscale, loc=legend_pos)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.show()
