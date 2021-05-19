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
plt.rcParams.update({'figure.max_open_warning': 0})

OPT = 'TERMINATION_REASON_OPTIMAL'
KKT_PASSES_LIMIT = 1e5
TIME_LIMIT_SECS = 60 * 60 # 1hr
# shift to use for shifted geomentric mean
SGM_SHIFT = int(10)
# penalised average runtime:
PAR = 2. # can be None, which removes unsolved instead of penalizing

SCALING_EXPS_TO_USE = [
    'off,off',
    'off,pock_chambolle alpha=1',
    '10 rounds,off',
    '10 rounds,pock_chambolle alpha=1',
]

PRIMALWEIGHT_EXPS_TO_USE = [
    'adaptive',
    'Fixed 1e-0',
]

# placeholder:
_BEST_STR = '_best_str_'


# Horrible HACK, but needs to be done
def label_lookup(label):
    if 'pdhg_enhanced' in label:
        return 'PLOP'
    if 'mirror-prox' in label:
        return 'Mirror Prox'
    if 'pdhg_vanilla' in label:
        return 'Vanilla PDHG'
    if 'scs-indirect' in label:
        return 'SCS Indirect'
    if 'scs-direct' in label:
        return 'SCS Direct'
    if 'nopresolve' in label:
        return 'No presolve'
    if 'no restarts' in label:
        return 'No restart'
    if 'adaptive theoretical' in label:
        return 'Adaptive restart (theory)'
    if 'adaptive enhanced' in label:
        return 'PLOP'
    if 'pdhg' in label and 'pdhg_mp_1h' in label:
        return 'PLOP'
    if 'off,off' in label:
        return 'No scaling'
    if r'off,pock_chambolle ($\alpha=1$)' in label:
        return 'Pock-Chambolle'
    if '10 rounds,off' in label:
        return 'Ruiz'
    if r'10 rounds,pock_chambolle ($\alpha=1$)' in label:
        return 'Ruiz + Pock-Chambolle'
    if 'stepsize' in label:
        if 'adaptive' in label:
            return 'PLOP'
        if 'fixed' in label:
            return 'Fixed step-size'
    if 'primalweight' in label:
        if 'adaptive' in label:
            return 'PLOP'
        if 'Fixed 1e-0' in label:
            return r'Fixed primal weight ($\theta=0$)'
        if _BEST_STR in label:
            return 'Best per-instance fixed primal weight'
    return label


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
                label=label_lookup(k))

    plt.ylabel('Number of problems solved')
    plt.xlabel(xlabel)
    plt.legend(loc='best')
    # plt.legend(bbox_to_anchor=(1.04, 0.5), loc='center left')
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

def shifted_geomean(x, shift):
    x = x[~np.isnan(x)]
    # return scipy.stats.mstats.gmean(x)
    sgm = np.exp(np.sum(np.log(x + shift)/len(x))) - shift
    return sgm if sgm > 0 else np.nan


def gen_total_solved_problems_table(df, prefix, par):
    solved_probs = df[df['termination_reason'] == OPT] \
                    .groupby('experiment_label') \
                    ['experiment_label'] \
                    .agg('count') \
                    .pipe(pd.DataFrame) \
                    .rename(columns = {'experiment_label': 'Solved count'})
    solved_probs.index.name = 'Experiment'
    solved_probs = solved_probs.reset_index()

    shift = SGM_SHIFT
    kkt_sgm = df.copy()
    if par is not None:
        kkt_sgm.loc[kkt_sgm['termination_reason'] != OPT, 'cumulative_kkt_matrix_passes'] = par * KKT_PASSES_LIMIT
    else:
        kkt_sgm.loc[kkt_sgm['termination_reason'] != OPT, 'cumulative_kkt_matrix_passes'] = np.nan

    # Hack for SCS direct
    kkt_sgm.loc[kkt_sgm['experiment_label'].str.contains('scs-direct'), 'cumulative_kkt_matrix_passes'] = np.nan

    kkt_sgm = kkt_sgm.groupby('experiment_label') \
                        ['cumulative_kkt_matrix_passes'] \
                        .agg(lambda _ : shifted_geomean(_, shift)) \
                        .pipe(pd.DataFrame) \
                        .rename(columns = {'cumulative_kkt_matrix_passes': f'KKT passes SGM{shift}'})
    kkt_sgm.index.name = 'Experiment'
    kkt_sgm = kkt_sgm.reset_index()

    wall_clock = df.copy()
    if par is not None:
        wall_clock.loc[wall_clock['termination_reason'] != OPT, 'solve_time_sec'] = par * TIME_LIMIT_SECS
    else:
        wall_clock.loc[wall_clock['termination_reason'] != OPT, 'solve_time_sec'] = np.nan

    wall_clock = wall_clock.groupby('experiment_label') \
                        ['solve_time_sec'] \
                        .agg(np.nanmean) \
                        .pipe(pd.DataFrame) \
                        .rename(columns = {'solve_time_sec': f'Mean solve time secs'})
    wall_clock.index.name = 'Experiment'
    wall_clock = wall_clock.reset_index()

    output = solved_probs.merge(kkt_sgm).merge(wall_clock)
    # rename the labels
    for e in output['Experiment']:
        output.loc[output['Experiment'] == e, 'Experiment'] = label_lookup(e)
    table = output.to_latex(float_format="%.1f", longtable=False,
                        index=False,
                        caption=f'Performance statistics.',
                        label=f't:solved-probs',
                        column_format='lc')
    with open(f'{prefix}_solved_probs_table.tex', "w") as f:
      f.write(table)


def gen_total_solved_problems_table_split_tol(df, prefix, par):
    tols = df['tolerance'].unique()
    for t in tols:
        gen_total_solved_problems_table(df[df['tolerance'] == t], prefix + f'_{t}', par)


def plot_loghist(x, nbins):
    x = x[~np.isnan(x)]
    hist, bins = np.histogram(x, bins=nbins)
    logbins = np.logspace(np.log10(bins[0]+1e-6),np.log10(bins[-1]), nbins)
    plt.hist(x, bins=logbins)
    plt.xscale('log')


def gen_ratio_histograms_split_tol(df, prefix, par):
    tols = df['tolerance'].unique()
    for t in tols:
        gen_ratio_histograms(df[df['tolerance'] == t], prefix + f'_{t}', par)

def gen_ratio_histograms(df, prefix, par):
    assert len(df['experiment_label'].unique()) == 2

    (l0, l1) = df['experiment_label'].unique()

    def performance_ratio_fn(df, par):
        df = df.reset_index()
        assert len(df) <= 2

        df0 = df[df['experiment_label'] == l0]
        df1 = df[df['experiment_label'] == l1]

        instance = df.instance_name.unique()

        if len(df0) == 1 and df0['termination_reason'].iloc[0] == OPT:
            kkt_passes_0 = df0['cumulative_kkt_matrix_passes'].iloc[0]
        else:
            kkt_passes_0 = par * KKT_PASSES_LIMIT
            if len(df0) == 0:
                print(f'{l0} missing {instance}')

        if len(df1) == 1 and df1['termination_reason'].iloc[0] == OPT:
            kkt_passes_1 = df1['cumulative_kkt_matrix_passes'].iloc[0]
        else:
            kkt_passes_1 = par * KKT_PASSES_LIMIT
            if len(df1) == 0:
                print(f'{l1} missing {instance}')

        # if (df['termination_reason'] != OPT).any():
        #    return np.nan
        return (kkt_passes_0 / kkt_passes_1)

    ratios = df.groupby(['instance_name']) \
        .apply(lambda _: performance_ratio_fn(_, par)) \
        .reset_index(name = 'ratio')
    plt.figure()
    plt.title(f'({label_lookup(l0)}):({label_lookup(l1)})')
    plot_loghist(ratios['ratio'], 25)
    plt.savefig(f'{prefix}_performance_ratio.pdf')
    table = ratios.to_latex(float_format="%.2f",
                            longtable=False,
                            index=False,
                            caption=f'Performance_ratio.',
                            label=f't:solved-probs',
                            column_format='lc')
    with open(f'{prefix}_({label_lookup(l0)}):({label_lookup(l1)})_ratio_table.tex', "w") as f:
      f.write(table)
    shift = 0.
    gmean = shifted_geomean(ratios['ratio'], shift)
    print(f'{prefix}: ratio ({label_lookup(l0)}) / ({label_lookup(l1)}) sgm{shift}: {gmean}')

# Unsolved problems might be missing from csv
def fill_in_missing_problems(df, instances_list):
    new_index = pd.Index(instances_list, name='instance_name')
    experiments = df['experiment_label'].unique()
    dfs = []
    for e in experiments:
        old_df = df[df['experiment_label'] == e]
        tol = old_df['tolerance'].unique()[0]
        new_df = old_df.set_index('instance_name').reindex(new_index).reset_index()
        # otherwise these would be nan
        new_df['tolerance'] = tol
        new_df['experiment_label'] = e
        dfs.append(new_df)
    return pd.concat(dfs)

with open('../benchmarking/miplib2017_instance_list') as f:
    miplib_instances = f.readlines()
miplib_instances = [p.strip() for p in miplib_instances if p[0] != '#']

with open('../benchmarking/mittelmann_instance_list') as f:
    mittelmann_instances = f.readlines()
mittelmann_instances = [p.strip() for p in mittelmann_instances if p[0] != '#']

# Pull out 'default' (ie best) pdhg implementation to compare against:
df_default = pd.read_csv('miplib_pdhg_enhanced_100k.csv')
df_default = fill_in_missing_problems(df_default, miplib_instances)

######################################################################

# bisco pdhg vs vanilla pdhg (JOIN DEFAULT)
df = pd.read_csv('miplib_pdhg_vanilla_100k.csv')
df = fill_in_missing_problems(df, miplib_instances)
df = pd.concat((df_default, df))
gen_solved_problems_plots_split_tol(df, 'miplib_defaults_v_vanilla')
gen_total_solved_problems_table_split_tol(df, 'miplib_defaults_v_vanilla', PAR)
gen_ratio_histograms_split_tol(df, 'miplib_defaults_v_vanilla', PAR)

######################################################################

# bisco vs mp vs scs on MIPLIB (JOIN MDHG/MP WITH SCS)
df_pdhg_mp = pd.read_csv('miplib_pdhg_mp_1h.csv')
df_pdhg_mp = fill_in_missing_problems(df_pdhg_mp, miplib_instances)
df_scs = pd.read_csv('miplib_scs_1h.csv')
df_scs = fill_in_missing_problems(df_scs, miplib_instances)
df = pd.concat((df_pdhg_mp, df_scs))
gen_solved_problems_plots_split_tol(df, 'miplib')
gen_total_solved_problems_table_split_tol(df, 'miplib', PAR)

######################################################################

# bisco vs mp vs scs on MITTELMANN (JOIN MDHG/MP WITH SCS)
df_pdhg_mp = pd.read_csv('mittelmann_pdhg_mp_1h.csv')
df_pdhg_mp = fill_in_missing_problems(df_pdhg_mp, mittelmann_instances)
df_scs = pd.read_csv('mittelmann_scs_1h.csv')
df_scs = fill_in_missing_problems(df_scs, mittelmann_instances)
df = pd.concat((df_pdhg_mp, df_scs))
gen_solved_problems_plots_split_tol(df, 'mittelmann')
gen_total_solved_problems_table_split_tol(df, 'mittelmann', PAR)

######################################################################

# bisco presolve vs no presolve (JOIN DEFAULT)
df = pd.read_csv('miplib_nopresolve_100k.csv')
df = pd.concat((df_default, df))
gen_solved_problems_plots_split_tol(df, 'miplib_presolve')
gen_total_solved_problems_table_split_tol(df, 'miplib_presolve', PAR)

######################################################################

# bisco scaling vs no scaling (NO JOIN DEFAULT)
df = pd.read_csv('miplib_scaling_100k.csv')
df = fill_in_missing_problems(df, miplib_instances)
# filter out un-needed scaling experiments:
df = pd.concat(df[df['experiment_label'].str.contains(e)] for e in SCALING_EXPS_TO_USE)
gen_solved_problems_plots_split_tol(df, 'miplib_scaling')
gen_total_solved_problems_table_split_tol(df, 'miplib_scaling', PAR)

######################################################################

# bisco restart vs no restart (NO JOIN DEFAULT)
df = pd.read_csv('miplib_restarts_100k.csv')
df = fill_in_missing_problems(df, miplib_instances)
gen_solved_problems_plots_split_tol(df, 'miplib_restarts')
gen_total_solved_problems_table_split_tol(df, 'miplib_restarts', PAR)

######################################################################

# bisco adaptive stepsize vs fixed stepsize (NO JOIN DEFAULT)
df = pd.read_csv('miplib_stepsize_100k.csv')
df = fill_in_missing_problems(df, miplib_instances)
gen_solved_problems_plots_split_tol(df, 'miplib_stepsize')
gen_total_solved_problems_table_split_tol(df, 'miplib_stepsize', PAR)

######################################################################

# bisco primalweight (NO JOIN DEFAULT)
df = pd.read_csv('miplib_primalweight_100k.csv')
df = fill_in_missing_problems(df, miplib_instances)

df_fixed = df[df['experiment_label'].str.contains('Fixed')]

# Pull out best performing fixed weight for each instance / tolerance:
df_best_fixed = df_fixed[df_fixed['termination_reason'] == OPT].reset_index()
best_idxs = df_best_fixed.groupby(['instance_name', 'tolerance'])['cumulative_kkt_matrix_passes'].idxmin()
df_best_fixed = df_best_fixed.loc[best_idxs]

for t in df_best_fixed['tolerance'].unique():
    # rename the experiment label
    df_best_fixed.loc[df_best_fixed['tolerance'] == t, 'experiment_label'] = \
        f'primalweight {_BEST_STR} {t}'

df_best_fixed = fill_in_missing_problems(df_best_fixed, miplib_instances)
df = pd.concat(df[df['experiment_label'].str.contains(e)] for e in PRIMALWEIGHT_EXPS_TO_USE)
df = pd.concat((df, df_best_fixed))
gen_solved_problems_plots_split_tol(df, 'miplib_primalweight')
gen_total_solved_problems_table_split_tol(df, 'miplib_primalweight', PAR)

