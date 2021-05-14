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
import scipy.stats
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
    plt.legend(bbox_to_anchor=(1.04, 0.5), loc='center left')
    plt.savefig(f'{prefix}_{xaxis}_v_solved_probs.pdf', bbox_inches = "tight")


def gen_solved_problems_plots(df, prefix):
    exps = df['experiment_label'].unique()
    dfs = {k: df[df['experiment_label'] == k] for k in exps}
    optimal_dfs = {k: v[v['termination_reason'] == OPT] for (k,v) in dfs.items()}

    solved_problems_vs_xaxis_figs(optimal_dfs, 'cumulative_kkt_matrix_passes',
                                  'Cumulative KKT matrix passes', prefix)
    solved_problems_vs_xaxis_figs(optimal_dfs, 'solve_time_sec',
                                  'Wall-clock time (secs)', prefix)



def gen_solved_problems_plots_split_tol(df, prefix):
    tols = df['tolerance'].unique()
    for t in tols:
        gen_solved_problems_plots(df[df['tolerance'] == t], prefix + f'_{t}')


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


def gen_total_solved_problems_table_split_tol(df, prefix):
    tols = df['tolerance'].unique()
    for t in tols:
        gen_total_solved_problems_table(df[df['tolerance'] == t], prefix + f'_{t}')


def plot_loghist(x, nbins):
    x = x[~np.isnan(x)]
    hist, bins = np.histogram(x, bins=nbins)
    logbins = np.logspace(np.log10(bins[0]+1e-6),np.log10(bins[-1]), nbins)
    plt.hist(x, bins=logbins)
    plt.xscale('log')


def gen_ratio_histograms_split_tol(df, prefix):
    tols = df['tolerance'].unique()
    for t in tols:
        gen_ratio_histograms(df[df['tolerance'] == t], prefix + f'_{t}')

def geomean(x):
    x = x[~np.isnan(x)]
    return scipy.stats.mstats.gmean(x)


def gen_ratio_histograms(df, prefix):

    (l1, l2) = df['experiment_label'].unique()

    def performance_ratio_fn(df):
        df = df.reset_index()
        if (df['termination_reason'] != OPT).any():
            return np.nan
        return (df.iloc[0]['cumulative_kkt_matrix_passes'] /
                df.iloc[1]['cumulative_kkt_matrix_passes'])

    ratios = df.groupby(['instance_name']) \
        .apply(performance_ratio_fn) \
        .reset_index(name = 'ratio')
    plt.figure()
    plt.title(f'({l1}):({l2})')
    plot_loghist(ratios['ratio'], 25)
    plt.savefig(f'{prefix}_performance_ratio.pdf')
    table = ratios.to_latex(float_format="%.2f",
                            longtable=False,
                            index=False,
                            caption=f'Performance_ratio.',
                            label=f't:solved-probs',
                            column_format='lc')
    with open(f'{prefix}_({l1}):({l2})_ratio_table.tex', "w") as f:
      f.write(table)
    gmean = geomean(ratios['ratio'])
    print(f'{prefix}: ratio ({l1}) / ({l2}) geomean: {gmean}')



# bisco pdhg vs vanilla pdhg
df = pd.read_csv('pdhg_miplib_defaults_vs_vanilla_100k.csv')
gen_solved_problems_plots_split_tol(df, 'miplib_defaults_v_vanilla')
gen_total_solved_problems_table_split_tol(df, 'miplib_defaults_v_vanilla')
gen_ratio_histograms_split_tol(df, 'miplib_defaults_v_vanilla')

# bisco vs mp vs scs on MIPLIB
df_pdhg_mp = pd.read_csv('miplib_pdhg_mp_1h.csv')
df_scs = pd.read_csv('miplib_scs_1h.csv')
df = pd.concat((df_pdhg_mp, df_scs))
gen_solved_problems_plots_split_tol(df, 'miplib')
gen_total_solved_problems_table_split_tol(df, 'miplib')

# bisco vs mp vs scs on MITTELMANN
df_pdhg_mp = pd.read_csv('mittelmann_pdhg_mp_1h.csv')
df_scs = pd.read_csv('mittelmann_scs_1h.csv')
df = pd.concat((df_pdhg_mp, df_scs))
gen_solved_problems_plots_split_tol(df, 'mittelmann')
gen_total_solved_problems_table_split_tol(df, 'mittelmann')

# bisco presolve vs no presolve
df = pd.read_csv('miplib_pdhg_nopresolve_100k.csv')
df_default = pd.read_csv('pdhg_miplib_defaults_vs_vanilla_100k.csv')
df_default = df_default[(df_default['experiment_label'] == 'pdhg defaults,1e-4') |
                        (df_default['experiment_label'] == 'pdhg defaults,1e-8')]

df = pd.concat((df, df_default))
gen_solved_problems_plots_split_tol(df, 'miplib_presolve')
gen_total_solved_problems_table_split_tol(df, 'miplib_presolve')

# bisco scaling vs no scaling
df = pd.read_csv('miplib_pdhg_scaling_100k.csv')
gen_solved_problems_plots_split_tol(df, 'miplib_scaling')
gen_total_solved_problems_table_split_tol(df, 'miplib_scaling')


# bisco scaling experiments
if True:
    df = pd.read_csv('miplib_pdhg_scaling_100k.csv')
    df = df[(df['experiment_label'] == 'tol 1e-8,5 rounds,pock_chambolle alpha=1,miplib_scaling_100k') | (df['experiment_label'] == 'tol 1e-8,10 rounds,l2,miplib_scaling_100k')]
    gen_solved_problems_plots_split_tol(df, 'miplib_new_default')
    gen_total_solved_problems_table_split_tol(df, 'miplib_new_default')
    gen_ratio_histograms_split_tol(df, 'miplib_new_default')

# bisco restart vs no restart
df = pd.read_csv('miplib_pdhg_restarts_100k.csv')
gen_solved_problems_plots_split_tol(df, 'miplib_restarts')
gen_total_solved_problems_table_split_tol(df, 'miplib_restarts')

