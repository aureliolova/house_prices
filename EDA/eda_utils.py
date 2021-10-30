import os
import sys

import datetime as dt
import re
import math
import copy
import typing as ty
import functools as ftools
import uuid

import pandas as pd
import numpy as np

from matplotlib import pyplot as plt
plt.style.use('seaborn')
import seaborn as sns

def make_barplot(data:pd.DataFrame, bar_col:str, pct:bool=False, return_data:bool=False) -> ty.Tuple[plt.Figure, plt.Axes]:
    '''
    Produce a count barplot of bar_col in data (excluding nulls)
    '''
    plot_data = data.copy()
    plot_data.dropna(subset=[bar_col], inplace=True)
    plot_data.loc[:, 'count'] = 1
    plot_data = plot_data.loc[:, [bar_col, 'count']]
    count_data = (
        plot_data
        .groupby(bar_col, as_index=False)
        .count()
        .sort_values(by=['count'], ascending=False)
        .reset_index(drop=True)
    )
    
    if pct:
        count_data.loc[:, 'count'] /= len(plot_data)
    
    if return_data:
        return count_data

    fig, ax = plt.subplots(figsize=(20, 10))
    ax.bar(x=count_data.index, height=count_data.loc[:, 'count'])
    ax.set_xticks(count_data.index)
    ax.set_xticklabels(count_data.loc[:, bar_col])
    
    total_rows = len(data)
    non_null_rows = len(plot_data)
    
    ax.set_title(f'{bar_col}: {non_null_rows=} out of {total_rows=}')
    
    
    return fig, ax

def make_cumprop_plot(data:pd.DataFrame, bar_col:str, pct:bool=False, return_data:bool=False) -> ty.Tuple[plt.Figure, plt.Axes]:
    '''
    Produce a cumulative distribution plot
    '''
    plot_data = data.copy()
    plot_data.dropna(subset=[bar_col], inplace=True)
    plot_data.loc[:, 'count'] = 1
    plot_data = plot_data.loc[:, [bar_col, 'count']]
    count_data = (
        plot_data
        .groupby(bar_col, as_index=False)
        .count()
        .sort_values(by=['count'], ascending=False)
        .reset_index(drop=True)
    )
    
    if pct:
        count_data.loc[:, 'count'] /= len(plot_data)
    
    count_data.loc[:, 'accum'] = count_data.loc[:, 'count'].cumsum()
    
    if return_data:
        return count_data
    
    fig, ax = plt.subplots(figsize=(20, 10))
    ax.bar(x=count_data.index, height=count_data.loc[:, 'accum'])
    ax.set_xticks(count_data.index)
    ax.set_xticklabels(count_data.loc[:, bar_col])
    
    total_rows = len(data)
    non_null_rows = len(plot_data)
    
    ax.set_title(f'{bar_col}: {non_null_rows=} out of {total_rows=}')
    
    return fig, ax

def make_classed_boxplot(data:pd.DataFrame, key_col:str, val_col:str, log:bool=False, return_data:bool=False) -> ty.Tuple[plt.Figure, plt.Axes]:
    '''
    Produce multiple boxplot using val_col for each element in key_col.
    '''
    plot_data = (
        data
        .copy()
        .dropna(subset=[key_col, val_col], how='any')
        .loc[:, [key_col, val_col]]
        .sort_values(by=key_col)
    )
    plot_data.loc[:, 'plot_id'] = [uuid.uuid1().hex for x in range(len(plot_data))]
    piv_plot_data = (
        plot_data
        .pivot_table(index=['plot_id'], columns=[key_col], values=val_col, aggfunc='sum')
    )
    if log:
        piv_plot_data = np.log10(piv_plot_data)
        ylab = f'log10 {val_col}'
    else:
        ylab = val_col
    mask = ~np.isnan(piv_plot_data.to_numpy())
    
    boxplot_data = np.array([d[m] for d, m in zip(piv_plot_data.to_numpy().T, mask.T)], dtype=object)
    names = piv_plot_data.columns.to_numpy() 
    if return_data:
        return boxplot_data, names
    
    fig, ax = plt.subplots(figsize=(20, 20))
    ax.boxplot(x=boxplot_data)
    ax.set_xlabel(key_col)
    ax.set_xticks(np.arange(1, piv_plot_data.shape[1]+1))
    ax.set_xticklabels(names)
    
    ax.set_ylabel(ylab)
    
    total_rows = len(data)
    non_null_rows = len(plot_data)
    
    ax.set_title(f'{key_col}: {non_null_rows=} out of {total_rows=}')
    
    return fig, ax

def main_class_var_figure(data:pd.DataFrame, class_col:str, val_col:str, log_bp:bool=False) -> ty.Tuple[plt.Figure, plt.Axes]:
    '''
    Produce a figure containing a count bar plot with cumulative percent line by class indicated in class_col; second figure contains separate boxplots of val_col by class_col.
    '''
    barplot_data = make_barplot(data=data, bar_col=class_col, pct=False, return_data=True)
    
    cumplot_data = make_cumprop_plot(data=data, bar_col=class_col, pct=True, return_data=True)
    cumplot_data.loc[:, 'scaled'] = cumplot_data.loc[:, 'accum'] * barplot_data.loc[:, 'count'].max()
    
    boxplot_data, boxplot_names = make_classed_boxplot(data=data, key_col=class_col, val_col=val_col, log=log_bp, return_data=True)
    bp_sorting_index = [enum[0] for enum in sorted(enumerate(boxplot_names), key=lambda e_tup: barplot_data.loc[:, class_col].to_list().index(e_tup[1]))]
    boxplot_data = boxplot_data[bp_sorting_index]
    boxplot_names = boxplot_names[bp_sorting_index]

    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(20, 10))

    total_rows = len(data)
    non_null_rows = barplot_data.loc[:, 'count'].sum()

    ax[0].set_title(f'{class_col}: {non_null_rows=} out of {total_rows=}')
    ax[0].bar(x=barplot_data.index, height=barplot_data.loc[:, 'count'])
    ax[0].set_xticks(barplot_data.index)
    ax[0].set_xticklabels(barplot_data.loc[:, class_col])

    secax_func = lambda x: x/max(x)
    _inv_secax_func = lambda pct, mx: pct*mx
    inv_secax_func = ftools.partial(_inv_secax_func, **{'mx':barplot_data.loc[:, 'count'].max()})

    ax[0].plot(cumplot_data.index, cumplot_data.loc[:, 'scaled'], 'y-o')
    ax[0].set_xlabel(class_col)
    ax[0].set_ylabel('count')
    secax = ax[0].secondary_yaxis('right', functions=(secax_func, inv_secax_func))
    secax.set_label('% of total')

    ax[1].boxplot(boxplot_data)
    ax[1].set_xticks(barplot_data.index+1)
    ax[1].set_xticklabels(boxplot_names)

    ax[1].set_xlabel(class_col)
    if log_bp:
        ylab = f'log10 {val_col}'
    else:
        ylab = val_col
    
    ax[1].set_ylabel(ylab)

    return fig, ax

def make_hist(data:pd.DataFrame, num_col:str, ax:plt.Axes=None, bin_mult:float=1, log:bool=False, return_data:bool=False) -> ty.Tuple[plt.Figure, plt.Axes]:
    hist_data = (
        data
        .copy()
        .dropna(subset=[num_col])
        .loc[:, num_col]
    )

    bins = math.ceil(math.log2(hist_data.max())*bin_mult)
    if log:
        hist_data = np.log10(hist_data)
        xlab = f'log10 {num_col}'
    else:
        xlab = num_col

    if return_data:
        return hist_data
    if not ax:
        fig, ax = plt.subplots(figsize=(20, 10))
    
    ax.hist(hist_data, density=True, bins=bins)

    ax.set_xlabel(xlab)
    ax.set_ylabel('density')
    total_rows = len(data)
    non_null_rows = len(hist_data)
    pct_null = non_null_rows / total_rows * 100

    ax.set_title(f'{num_col}: {non_null_rows=} out of {total_rows=} ({pct_null:1,.2f}%)')
    if not ax:
        return fig, ax
    else:
        return ax

def make_boxplot(data:pd.DataFrame, num_col:str, log:bool=False, ax:plt.Axes=None) -> ty.Tuple[plt.Figure, plt.Axes]:
    hist_data = make_hist(data=data, num_col=num_col, return_data=True, log=log)

    if not ax:
        fig, ax = plt.subplots(figsize=(20, 10))
    
    ax.boxplot(hist_data, vert=False)
    
    if not ax:
        return fig, ax
    else:
        return ax
        
def make_scatterplot(data:pd.DataFrame, pred_col:str, resp_col:str, ax:plt.Axes=None, log_pred:bool=False, log_resp:bool=False, return_data:bool=False) -> ty.Tuple[plt.Figure, plt.Axes]:
    scatter_data = (
        data
        .copy()
        .dropna(subset=[pred_col, resp_col], how='any')
    )
    
    if log_pred:
        scatter_data.loc[:, pred_col] = np.log10(scatter_data.loc[:, pred_col])
        xlab = f'log10 {pred_col}'
    else:
        xlab = pred_col

    if log_resp:
        scatter_data.loc[:, resp_col] = np.log10(scatter_data.loc[:, resp_col])
        ylab = f'log10 {resp_col}'
    else:
        ylab = resp_col

    if return_data:
        return scatter_data

    if not ax:
        fig, ax = plt.subplots(figsize=(20, 10))
    ax.scatter(scatter_data.loc[:, pred_col], scatter_data.loc[:, resp_col])
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    if not ax: 
        return fig, ax
    else:
        return ax

def main_cont_var_figure(data:pd.DataFrame, cont_col:str, tgt_col:str, hist_bin_mult:float=1, log_cont:bool=False, log_tgt:bool=False) -> plt.Figure:
    fig = plt.figure(figsize=(20, 20))

    gridspec = fig.add_gridspec(9, 1)
    
    hist_ax = fig.add_subplot(gridspec[0:4])
    make_hist(data=data, num_col=cont_col, ax=hist_ax, bin_mult=hist_bin_mult, log=log_cont)

    bp_ax = fig.add_subplot(gridspec[4])
    make_boxplot(data=data, num_col=cont_col, log=log_cont, ax=bp_ax)

    scatter_ax = fig.add_subplot(gridspec[5:9])
    make_scatterplot(data=data, pred_col=cont_col, resp_col=tgt_col, ax=scatter_ax, log_pred=log_cont, log_resp=log_tgt)

    return fig
