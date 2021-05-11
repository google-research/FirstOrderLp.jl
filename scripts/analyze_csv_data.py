# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# TODO


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

OPT = 'TERMINATION_REASON_OPTIMAL'

def solved_problems_vs_xaxis_figs(dfs, xaxis, xlabel, prefix):
    plt.figure()
    for k, df_k in dfs.items():
        stats_df = df_k.groupby(xaxis) \
                   [xaxis] \
        .agg('count') \
        .pipe(pd.DataFrame) \
        .rename(columns = {xaxis: 'frequency'})

        stats_df['cum_solved_count'] = stats_df['frequency'].cumsum()
        stats_df = stats_df.drop(columns = 'frequency').reset_index()
        plt.plot(stats_df[xaxis],
                stats_df['cum_solved_count'],
                label = k)

    plt.ylabel('Number of problems solved')
    plt.xlabel(xlabel)
    plt.legend()
    plt.savefig(f'{prefix}_{xaxis}_v_solved_probs.pdf')


def gen_solved_problems_plots(df, prefix):
    keys = df['experiment_label'].unique()
    dfs = {k: df[df['experiment_label'] == k] for k in keys}
    optimal_dfs = {k: v[v['termination_reason'] == OPT] for (k,v) in dfs.items()}

    solved_problems_vs_xaxis_figs(optimal_dfs, 'cumulative_kkt_matrix_passes',
                                  'Cumulative KKT matrix passes', prefix)
    solved_problems_vs_xaxis_figs(optimal_dfs, 'solve_time_sec',
                                  'Wall-clock time (secs)', prefix)


def gen_total_solved_problems_table(df, prefix):
    solved_probs = df[df['termination_reason'] == OPT] \
                    .groupby('experiment_label') \
                    ['experiment_label'] \
                    .agg('count') \
                    .pipe(pd.DataFrame) \
                    .rename(columns = {'experiment_label': 'Count'})
    solved_probs.index.name = 'Experiment'
    solved_probs = solved_probs.reset_index()

    table = solved_probs.to_latex(float_format="%.4f", longtable=False,
                        index=False,
                        caption=f'Number of problems solved.',
                        label=f't:solved-probs',
                        column_format='lc')
    with open(f'{prefix}_solved_probs_table.tex', "w") as f:
      f.write(table)


def plot_loghist(x, bins=25):
    x = x[~np.isnan(x)]
    hist, bins = np.histogram(x, bins=bins)
    logbins = np.logspace(np.log10(bins[0]),np.log10(bins[-1]),len(bins))
    plt.hist(x, bins=logbins)
    plt.xscale('log')

def gen_ratio_histograms(df, prefix):
    def performance_ratio_fn(df):
        df = df.reset_index()
        if (df['termination_reason'] != OPT).any():
            return np.nan
        return (df.iloc[0]['cumulative_kkt_matrix_passes'] / 
                df.iloc[1]['cumulative_kkt_matrix_passes'])

    ratios = df.groupby(['tolerance', 'instance_name']) \
        .apply(performance_ratio_fn) \
        .reset_index(name = 'default/vanilla')
    plt.figure()
    plot_loghist(ratios[ratios['tolerance'] == 1e-4]['default/vanilla'], bins=50)
    plt.savefig(f'{prefix}_performance_ratio_1e-4.pdf')
    plt.figure()
    plot_loghist(ratios[ratios['tolerance'] == 1e-8]['default/vanilla'], bins=50)
    plt.savefig(f'{prefix}_performance_ratio_1e-8.pdf')


df = pd.read_csv('pdhg_miplib_defaults_vs_vanilla_100k.csv')
gen_solved_problems_plots(df, 'miplib_defaults_v_vanilla')
gen_total_solved_problems_table(df, 'miplib_defaults_v_vanilla')
gen_ratio_histograms(df, 'miplib_defauls_v_vanilla')

df_pdhg_mp = pd.read_csv('miplib_pdhg_mp_1h.csv')
df_scs = pd.read_csv('miplib_scs_1h.csv')
df = pd.concat((df_pdhg_mp, df_scs))
gen_solved_problems_plots(df, 'miplib')
gen_total_solved_problems_table(df, 'miplib')

df_pdhg_mp = pd.read_csv('mittelmann_pdhg_mp_1h.csv')
df_scs = pd.read_csv('mittelmann_scs_1h.csv')
df = pd.concat((df_pdhg_mp, df_scs))
gen_solved_problems_plots(df, 'mittelmann')
gen_total_solved_problems_table(df, 'mittelmann')

