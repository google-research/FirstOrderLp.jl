# Copyright 2021 The FirstOrderLp Authors
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

#  This script generates all the experimental results used in the paper.
#  It requires python 3, numpy, pandas, and matplotlib installed to run.
#
#   `python analyze_csv_data.py`
#
#  It reads csv files containing experimental results from ./csv, and outputs
#  pdf figures to ./results/figs and latex tables to ./results/tex.


import itertools
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from cycler import cycler
plt.rcParams.update({'figure.max_open_warning': 0, 'font.size': 16})

# The 'TkAgg' matplotlib backend fails at the default recursion limit.
sys.setrecursionlimit(10000)

# This is required to generate plots that are easy to read when printed:
plt.rcParams['axes.prop_cycle'] = cycler(
    linestyle=[
        '-',
        '--',
        ':',
        '-.',
        '-',
        '--',
        ':',
        '-.',
        '-',
        '--'],
    color=[
        '#1f77b4',
        '#ff7f0e',
        '#2ca02c',
        '#d62728',
        '#9467bd',
        '#8c564b',
        '#e377c2',
        '#7f7f7f',
        '#bcbd22',
        '#17becf'])


# directory where the csv files are located
CSV_DIR = './csv'

# directory where all the figure pdf and table tex files are written to:
OUTPUT_DIR = './results'
FIGS_DIR = os.path.join(OUTPUT_DIR, 'figs')
TEX_DIR = os.path.join(OUTPUT_DIR, 'tex')

OPT = 'TERMINATION_REASON_OPTIMAL'
KKT_PASSES_LIMIT = 1e5
TIME_LIMIT_SECS = 60 * 60  # 1hr
# shift to use for shifted geometric mean
SGM_SHIFT = int(10)
# penalised average runtime:
PAR = 1.  # can be None, which removes unsolved instead of penalizing

# Which scaling experiments to present
SCALING_EXPS_TO_USE = [
    'off,off',
    'off,pock_chambolle alpha=1',
    '10 rounds,off',
    '10 rounds,pock_chambolle alpha=1',
]

# Which primal-weight experiments to present
PRIMALWEIGHT_EXPS_TO_USE = [
    'adaptive',
    #'Fixed 1e-0',
]

# placeholder:
_BEST_STR = '_best_str_'
_BEST_FIXED = '_best_fixed_'

# Dataset names:
MITTELMANN_STR = 'lp_benchmark'
MIPLIB_STR = 'mip_relaxations'
NETLIB_STR = 'netlib'

# Change table font size to fit paper:
LATEX_FONT_SIZE = '\\small'

# Naming for improvements plots:
_PDHG = 'PDHG'
_RESTARTS = '+restarts'
_SCALING = '+scaling'
_PRIMAL_WEIGHT = '+primal\nweight'
_STEPSIZE = '+step\nsize'
_PRESOLVE = '+presolve\n(= PDLP)'

# Order in which improvements should appear:
IMPROVEMENTS_ORDER = [
    _PDHG,
    _RESTARTS,
    _SCALING,
    _PRIMAL_WEIGHT,
    _STEPSIZE,
    _PRESOLVE]
IMPROVEMENTS_ORDER_IDX = dict(
    zip(IMPROVEMENTS_ORDER, range(len(IMPROVEMENTS_ORDER))))


# Horrible HACK, but needs to be done
def label_lookup(label):
    if 'pdhg_enhanced' in label:
        return 'PDLP'
    if 'mirror-prox' in label:
        return 'Enh. Extragradient'
    if 'pdhg_vanilla' in label:
        return 'PDHG'
    if 'scs-indirect' in label:
        return 'SCS (matrix-free)'
    if 'scs-direct' in label:
        return 'SCS'
    if 'nopresolve' in label:
        return 'No presolve'
    if 'no restarts' in label:
        return 'No restart'
    if 'adaptive theoretical' in label:
        return 'Adaptive restart (theory)'
    if 'adaptive enhanced' in label:
        return 'PDLP'
    if 'pdhg' in label and 'pdhg_mp_1h' in label:
        return 'PDLP'
    if 'off,off' in label:
        return 'No scaling'
    if 'off,pock_chambolle alpha=1' in label:
        return 'Pock-Chambolle'
    if '10 rounds,off' in label:
        return 'Ruiz'
    if '10 rounds,pock_chambolle alpha=1' in label:
        return 'Ruiz + Pock-Chambolle'
    if 'stepsize' in label:
        if 'adaptive' in label:
            return 'PDLP'
        if 'fixed' in label:
            return 'Fixed step-size'
    if 'scaling' in label:
        if _BEST_STR in label:
            return 'Best per-instance scaling'
    if 'primalweight' in label:
        if 'adaptive' in label:
            return 'PDLP'
        if 'Fixed 1e-0' in label:
            return r'Fixed PW ($\theta=0$)'
        if _BEST_STR in label:
            return 'Best per-instance PW'
        if _BEST_FIXED in label:
            return 'Best fixed PW'
    if 'improvements' in label:
        if 'vanilla' in label:
            return _PDHG
        st = ''
        if 'restarts' in label:
            st = _RESTARTS
        if 'scaling' in label:
            st = _SCALING
        if 'primal weight' in label:
            st = _PRIMAL_WEIGHT
        if 'step size' in label:
            st = _STEPSIZE
        if 'pdlp_final' in label:
            st = _PRESOLVE
        return st
    if 'malitskypock' in label:
        if _BEST_STR in label:
            return 'Best per-instance MP settings'
        return 'Best fixed MP setting'
    return label


def sanitize_title(title):
    title = title.replace('_', ' ').title()
    title = title.replace('Lp', 'LP')
    title = title.replace('Mip', 'MIP')
    title = title.replace('Pdlp', 'PDLP')
    title = title.replace('Pdhg', 'PDHG')
    title = title.replace('Scs', 'SCS')
    title = title.replace('Sgm', 'SGM')
    return title


# Generate plots of xaxis vs fraction of solved problems
def solved_problems_vs_xaxis_figs(
        dfs,
        xaxis,
        xlabel,
        prefix,
        num_instances,
        outer_legend=False):
    plt.figure()
    stats_dfs = {}
    for k, df_k in dfs.items():
        stats_df = df_k.groupby(xaxis)[xaxis] \
            .agg('count') \
            .pipe(pd.DataFrame) \
            .rename(columns={xaxis: 'frequency'})

        stats_df['cum_solved_count'] = stats_df['frequency'].cumsum() / \
            num_instances
        stats_df = stats_df.drop(columns='frequency').reset_index()
        stats_dfs[k] = stats_df

    max_xaxis = pd.concat(stats_dfs)[xaxis].max()

    for k, df_k in stats_dfs.items():
        if df_k.empty:
            continue
        df_k = df_k.append({xaxis: max_xaxis,
                            'cum_solved_count': df_k.iloc[-1]['cum_solved_count']},
                           ignore_index=True)
        df_k.reset_index()
        plt.plot(df_k[xaxis],
                 df_k['cum_solved_count'],
                 label=label_lookup(k))

    plt.ylabel('Fraction of problems solved')
    plt.xlabel(xlabel)
    plt.ylim((0, 1))
    plt.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    plt.title(sanitize_title(prefix))
    if outer_legend:
        plt.legend(bbox_to_anchor=(1.04, 0.5), loc='center left')
    else:
        plt.legend(loc='best')
    path = os.path.join(FIGS_DIR, f'{prefix}_{xaxis}_v_solved_probs.pdf')
    plt.savefig(
        path,
        bbox_inches="tight")


def gen_solved_problems_plots(df, prefix, num_instances, outer_legend=False):
    exps = df['experiment_label'].unique()
    dfs = {k: df[df['experiment_label'] == k] for k in exps}
    optimal_dfs = {k: v[v['termination_reason'] == OPT]
                   for (k, v) in dfs.items()}

    solved_problems_vs_xaxis_figs(
        optimal_dfs,
        'cumulative_kkt_matrix_passes',
        f'KKT matrix passes SGM{SGM_SHIFT}',
        prefix,
        num_instances,
        outer_legend)
    solved_problems_vs_xaxis_figs(
        optimal_dfs,
        'solve_time_sec',
        'Wall-clock time (secs)',
        prefix,
        num_instances,
        outer_legend)


def gen_solved_problems_plots_split_tol(
        df, prefix, num_instances, outer_legend=False):
    tols = df['tolerance'].unique()
    for t in tols:
        gen_solved_problems_plots(
            df[df['tolerance'] == t], prefix + f'_tol_{t:.0E}', num_instances, outer_legend)


def shifted_geomean(x, shift):
    x = x[~np.isnan(x)]
    sgm = np.exp(np.sum(np.log(x + shift) / len(x))) - shift
    return sgm if sgm > 0 else np.nan


def change_table_font_size(table):
    table = table.replace(
        '\\begin{table}\n',
        '\\begin{table}\n' +
        LATEX_FONT_SIZE +
        '\n')
    table = table.replace('\\caption{', '\\caption{' + LATEX_FONT_SIZE + ' ')
    return table


def gen_total_solved_problems_table(df, prefix, par):
    solved_probs = df[df['termination_reason'] == OPT] \
        .groupby('experiment_label')['experiment_label'] \
        .agg('count') \
        .pipe(pd.DataFrame) \
        .rename(columns={'experiment_label': 'Solved count'})
    solved_probs.index.name = 'Experiment'
    solved_probs = solved_probs.reset_index()

    shift = SGM_SHIFT
    kkt_sgm = df.copy()
    if par is not None:
        kkt_sgm.loc[kkt_sgm['termination_reason'] != OPT,
                    'cumulative_kkt_matrix_passes'] = par * KKT_PASSES_LIMIT
    else:
        kkt_sgm.loc[kkt_sgm['termination_reason'] !=
                    OPT, 'cumulative_kkt_matrix_passes'] = np.nan

    # Hack for SCS direct
    kkt_sgm.loc[kkt_sgm['experiment_label'].str.contains(
        'scs-direct'), 'cumulative_kkt_matrix_passes'] = np.nan

    kkt_sgm = kkt_sgm.groupby('experiment_label')['cumulative_kkt_matrix_passes'] \
        .agg(lambda _: shifted_geomean(_, shift)) \
        .pipe(pd.DataFrame) \
        .rename(columns={'cumulative_kkt_matrix_passes':
                         f'KKT passes SGM{shift}'})
    kkt_sgm.index.name = 'Experiment'
    kkt_sgm = kkt_sgm.reset_index()

    wall_clock = df.copy()
    if par is not None:
        wall_clock.loc[wall_clock['termination_reason'] !=
                       OPT, 'solve_time_sec'] = par * TIME_LIMIT_SECS
    else:
        wall_clock.loc[wall_clock['termination_reason']
                       != OPT, 'solve_time_sec'] = np.nan

    wall_clock = wall_clock.groupby('experiment_label')['solve_time_sec'] \
        .agg(lambda _: shifted_geomean(_, shift)) \
        .pipe(pd.DataFrame) \
        .rename(columns={'solve_time_sec': f'Solve time secs SGM10'})
    wall_clock.index.name = 'Experiment'
    wall_clock = wall_clock.reset_index()

    output = solved_probs.merge(kkt_sgm).merge(wall_clock)
    # rename the labels
    for e in output['Experiment']:
        output.loc[output['Experiment'] == e, 'Experiment'] = label_lookup(e)

    output = output.sort_values('Solved count', ascending=True)

    # HACK to fix improvements table ordering and line break
    if 'improvements' in prefix:
        output['rank'] = output['Experiment'].map(IMPROVEMENTS_ORDER_IDX)
        output.sort_values('rank', inplace=True)
        output.drop('rank', 1, inplace=True)
        to_write = output.copy()
        for e in to_write['Experiment']:
            to_write.loc[to_write['Experiment'] == e,
                         'Experiment'] = e.replace('\n', ' ')
    else:
        to_write = output

    table = to_write.to_latex(
        float_format="%.1f",
        longtable=False,
        index=False,
        caption=f'Performance statistics: {sanitize_title(prefix)}',
        label=f't:solved-probs-{prefix}',
        column_format='lccc',
        escape=False,
        na_rep='-')
    table = change_table_font_size(table)
    path = os.path.join(TEX_DIR, f'{prefix}_solved_probs_table.tex')
    with open(path, "w") as f:
        f.write(table)
    return output


def gen_total_solved_problems_table_split_tol(df, prefix, par):
    outputs = {}
    tols = df['tolerance'].unique()
    for t in tols:
        outputs[t] = gen_total_solved_problems_table(
            df[df['tolerance'] == t], prefix + f'_tol_{t:.0E}', par)
    return outputs


def plot_loghist(x, nbins):
    x = x[~np.isnan(x)]
    hist, bins = np.histogram(x, bins=nbins)
    logbins = np.logspace(np.log10(bins[0] + 1e-6), np.log10(bins[-1]), nbins)
    plt.hist(x, bins=logbins)
    plt.xscale('log')


def gen_ratio_histograms_split_tol(df, prefix, par):
    tols = df['tolerance'].unique()
    for t in tols:
        gen_ratio_histograms(df[df['tolerance'] == t],
                             prefix + f'_tol_{t:.0E}',
                             'cumulative_kkt_matrix_passes',
                             f'KKT matrix passes SGM{SGM_SHIFT}',
                             KKT_PASSES_LIMIT,
                             par)
        gen_ratio_histograms(df[df['tolerance'] == t],
                             prefix + f'_tol_{t:.0E}',
                             'solve_time_sec',
                             'Wall-clock time (secs)',
                             TIME_LIMIT_SECS,
                             par)


def gen_ratio_histograms(df, prefix, xaxis, xlabel, limit, par):
    assert len(df['experiment_label'].unique()) == 2

    (l0, l1) = df['experiment_label'].unique()

    def performance_ratio_fn(df, par):
        df = df.reset_index()
        assert len(df) <= 2

        df0 = df[df['experiment_label'] == l0]
        df1 = df[df['experiment_label'] == l1]

        instance = df.instance_name.unique()

        if len(df0) == 1 and df0['termination_reason'].iloc[0] == OPT:
            kkt_passes_0 = df0[xaxis].iloc[0]
        else:
            kkt_passes_0 = par * limit

        if len(df1) == 1 and df1['termination_reason'].iloc[0] == OPT:
            kkt_passes_1 = df1[xaxis].iloc[0]
        else:
            kkt_passes_1 = par * limit

        # if (df['termination_reason'] != OPT).any():
        #    return np.nan
        return (kkt_passes_0 / kkt_passes_1)

    ratios = df.groupby(['instance_name']) \
        .apply(lambda _: performance_ratio_fn(_, par)) \
        .reset_index(name='ratio')
    plt.figure(figsize=(10, 6))
    plt.title(sanitize_title(
        f'{prefix} {xlabel} {label_lookup(l0)}:{label_lookup(l1)}'))
    plot_loghist(ratios['ratio'], min(len(ratios) // 3, 25))
    path = os.path.join(
        FIGS_DIR,
        f'{prefix}_{label_lookup(l0)}_{label_lookup(l1)}_{xaxis}_performance_ratio.pdf')
    plt.savefig(path)
    table = ratios.to_latex(float_format="%.2f",
                            longtable=False,
                            index=False,
                            caption=f'Performance ratio.',
                            label=f't:ratio-{prefix}',
                            column_format='lc',
                            na_rep='-')
    table = change_table_font_size(table)
    path = os.path.join(TEX_DIR, f'{prefix}_{label_lookup(l0)}:'
                                 f'{label_lookup(l1)}_{xaxis}_ratio_table.tex')
    with open(path, "w") as f:
        f.write(table)

# Unsolved problems might be missing from csv, make sure all are accounted for.


def fill_in_missing_problems(df, instances_list):
    new_index = pd.Index(instances_list, name='instance_name')
    experiments = df['experiment_label'].unique()
    dfs = []
    for e in experiments:
        old_df = df[df['experiment_label'] == e]
        tol = old_df['tolerance'].unique()[0]
        new_df = old_df.set_index('instance_name').reindex(
            new_index).reset_index()
        # otherwise these would be nan
        new_df['tolerance'] = tol
        new_df['experiment_label'] = e
        dfs.append(new_df)
    return pd.concat(dfs)


def improvements_plot(dfs, prefix, key, ascending):
    normalized_dfs = []
    for df in dfs:
        df[key] /= df[df['Experiment'] == 'PDHG'][key].to_numpy()[0]
        normalized_dfs.append(df)

    df = pd.concat(normalized_dfs)
    fig = plt.figure(figsize=(10, 6))

    markers = itertools.cycle(['o', 'v', '^', '<', '>', 's'])
    for tol in df['tolerance'].unique():
        _df = df[df['tolerance'] == tol].reset_index(drop=True)
        plt.plot(
            _df[key].to_numpy(),
            linestyle='--',
            marker=next(markers),
            markersize=12,
            label=f'tolerance {tol:.0E}')
        plt.yscale('log')
        plt.ylabel('Normalized ' + key, fontsize=20)
        plt.title(sanitize_title(prefix), fontsize=20)
        plt.yticks(fontsize=20)
        plt.xticks(range(len(_df['Experiment'])),
                   _df['Experiment'].to_list(),
                   fontsize=20)
        ax = plt.gca()
        ax.yaxis.set_major_locator(ticker.LogLocator(subs=[1, 2, 3, 5, 7]))
        ax.yaxis.set_major_formatter(
            ticker.LogFormatterSciNotation(
                labelOnlyBase=False,
                minor_thresholds=(4, 2)))
        # ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f")

    if len(dfs) > 1:
        plt.legend(loc='best', prop={'size': 20})
    name = key.replace(' ', '_')
    path = os.path.join(FIGS_DIR, f'{prefix}_{name}.pdf')
    plt.savefig(
        path,
        bbox_inches="tight")


def gen_all_improvement_plots(outputs, prefix):
    dfs = []
    for tol, df in outputs.items():
        df = df.copy()
        df['tolerance'] = tol
        dfs.append(df)
    improvements_plot(
        dfs,
        prefix,
        'KKT passes SGM10',
        ascending=False)
    improvements_plot(
        dfs,
        prefix,
        'Solve time secs SGM10',
        ascending=False)
    improvements_plot(
        dfs,
        prefix,
        'Solved count',
        ascending=True)


# First, make output directories
if not os.path.exists(FIGS_DIR):
    os.makedirs(FIGS_DIR)
if not os.path.exists(TEX_DIR):
    os.makedirs(TEX_DIR)

# Get clean list of all problems we tested on:
with open('../benchmarking/mip_relaxations_instance_list') as f:
    miplib_instances = f.readlines()
miplib_instances = [p.strip() for p in miplib_instances if p[0] != '#']

with open('../benchmarking/lp_benchmark_instance_list') as f:
    mittelmann_instances = f.readlines()
mittelmann_instances = [p.strip() for p in mittelmann_instances if p[0] != '#']

with open('../benchmarking/netlib_benchmark_instance_list') as f:
    netlib_instances = f.readlines()
netlib_instances = [p.strip() for p in netlib_instances if p[0] != '#']

# Pull out 'default' (ie best) pdhg implementation to compare against:
df_default = pd.read_csv(
    os.path.join(
        CSV_DIR,
        'miplib_pdhg_enhanced_100k.csv'))
df_default = fill_in_missing_problems(df_default, miplib_instances)

######################################################################

# PDLP pdhg vs vanilla pdhg (JOIN DEFAULT)
df = pd.read_csv(os.path.join(CSV_DIR, 'miplib_pdhg_vanilla_100k.csv'))
df = fill_in_missing_problems(df, miplib_instances)
df = pd.concat((df_default, df))
gen_solved_problems_plots_split_tol(df, f'{MIPLIB_STR}', len(miplib_instances))
gen_total_solved_problems_table_split_tol(df, f'{MIPLIB_STR}', PAR)
gen_ratio_histograms_split_tol(df, f'{MIPLIB_STR}', PAR)

######################################################################

df = pd.read_csv(os.path.join(CSV_DIR, 'mittelmann_pdhg_enhanced_100k.csv'))
df = fill_in_missing_problems(df, mittelmann_instances)
df_vanilla = pd.read_csv(
    os.path.join(
        CSV_DIR,
        'mittelmann_improvements_100k.csv'))
df_vanilla = df_vanilla[df_vanilla['enhancements'] == 'vanilla']
df_vanilla = fill_in_missing_problems(df_vanilla, mittelmann_instances)
df = pd.concat((df, df_vanilla))
gen_solved_problems_plots_split_tol(
    df, f'{MITTELMANN_STR}', len(mittelmann_instances))
gen_total_solved_problems_table_split_tol(df, f'{MITTELMANN_STR}', PAR)
gen_ratio_histograms_split_tol(df, f'{MITTELMANN_STR}', PAR)

######################################################################

df = pd.read_csv(os.path.join(CSV_DIR, 'netlib_pdhg_enhanced_100k.csv'))
df = fill_in_missing_problems(df, netlib_instances)
df_vanilla = pd.read_csv(
    os.path.join(
        CSV_DIR,
        'netlib_improvements_100k.csv'))
df_vanilla = df_vanilla[df_vanilla['enhancements'] == 'vanilla']
df_vanilla = fill_in_missing_problems(df_vanilla, netlib_instances)
df = pd.concat((df, df_vanilla))
gen_solved_problems_plots_split_tol(
    df, f'{NETLIB_STR}', len(netlib_instances))
gen_total_solved_problems_table_split_tol(df, f'{NETLIB_STR}', PAR)
gen_ratio_histograms_split_tol(df, f'{NETLIB_STR}', PAR)


######################################################################

# Scaling results (JOIN DEFAULT)
df = pd.read_csv(os.path.join(CSV_DIR, 'miplib_malitskypock_100k.csv'))
mp_solved = df[df['termination_reason'] == OPT] \
    .groupby(['experiment_label', 'tolerance'])['experiment_label'] \
    .agg('count') \
    .pipe(pd.DataFrame) \
    .rename(columns={'experiment_label': 'solved'}) \
    .reset_index()
dfs = []
for t in df['tolerance'].unique():
    _df = mp_solved[mp_solved['tolerance'] == t]
    best_mp_run = _df.loc[_df['solved'].idxmax()]['experiment_label']
    dfs.append(df[df['experiment_label'] == best_mp_run])
df_best_ind = fill_in_missing_problems(pd.concat(dfs), miplib_instances)

# Pull out best performing scaling for each instance / tolerance:
df_best_fixed = df[df['termination_reason'] == OPT].reset_index()
best_idxs = df_best_fixed.groupby(['instance_name', 'tolerance'])[
    'cumulative_kkt_matrix_passes'].idxmin()
df_best_fixed = df_best_fixed.loc[best_idxs]

for t in df_best_fixed['tolerance'].unique():
    # rename the experiment label
    df_best_fixed.loc[df_best_fixed['tolerance'] == t, 'experiment_label'] = \
        f'malitskypock {_BEST_STR} {t}'

df_best_fixed = fill_in_missing_problems(df_best_fixed, miplib_instances)
df_stepsize = pd.read_csv(os.path.join(CSV_DIR, 'miplib_stepsize_100k.csv'))
df_stepsize = fill_in_missing_problems(df_stepsize, miplib_instances)

df = pd.concat((df_stepsize, df_best_fixed, df_best_ind))
gen_solved_problems_plots_split_tol(
    df, f'{MIPLIB_STR}_stepsize', len(miplib_instances), False)
gen_total_solved_problems_table_split_tol(df, f'{MIPLIB_STR}_stepsize', PAR)

######################################################################

# PDLP vs mp vs scs on MIPLIB (JOIN PDHG/MP WITH SCS)
df_pdhg_mp = pd.read_csv(os.path.join(CSV_DIR, 'miplib_pdhg_mp_1h.csv'))
df_pdhg_mp = fill_in_missing_problems(df_pdhg_mp, miplib_instances)
df_scs = pd.read_csv(os.path.join(CSV_DIR, 'miplib_scs_1h.csv'))
df_scs = fill_in_missing_problems(df_scs, miplib_instances)
df_pdhg_vanilla = pd.read_csv(os.path.join(
    CSV_DIR, 'miplib_pdhg_vanilla_1h.csv'))
df_pdhg_vanilla = fill_in_missing_problems(df_pdhg_vanilla, miplib_instances)
df = pd.concat((df_pdhg_mp, df_pdhg_vanilla, df_scs))
gen_solved_problems_plots_split_tol(
    df, f'{MIPLIB_STR}_baselines', len(miplib_instances))
gen_total_solved_problems_table_split_tol(df, f'{MIPLIB_STR}_baselines', PAR)

df_pdhg_scs_dir = pd.concat(
    (df_pdhg_mp[df_pdhg_mp['method'] == 'pdhg'], df_scs[df_scs['method'] == 'scs-direct']))
df_pdhg_scs_indir = pd.concat(
    (df_pdhg_mp[df_pdhg_mp['method'] == 'pdhg'], df_scs[df_scs['method'] == 'scs-indirect']))
gen_ratio_histograms_split_tol(df_pdhg_mp, f'{MIPLIB_STR}', PAR)
gen_ratio_histograms_split_tol(df_pdhg_scs_indir, f'{MIPLIB_STR}', PAR)
gen_ratio_histograms_split_tol(df_pdhg_scs_dir, f'{MIPLIB_STR}', PAR)

######################################################################

# PDLP vs mp vs scs on MITTELMANN (JOIN PDHG/MP WITH SCS)
df_pdhg_mp = pd.read_csv(os.path.join(CSV_DIR, 'mittelmann_pdhg_mp_1h.csv'))
df_pdhg_mp = fill_in_missing_problems(df_pdhg_mp, mittelmann_instances)
df_pdhg_vanilla = pd.read_csv(os.path.join(
    CSV_DIR, 'mittelmann_pdhg_vanilla_1h.csv'))
df_pdhg_vanilla = fill_in_missing_problems(df_pdhg_vanilla, mittelmann_instances)
df_scs = pd.read_csv(os.path.join(CSV_DIR, 'mittelmann_scs_1h.csv'))
df_scs = fill_in_missing_problems(df_scs, mittelmann_instances)
df = pd.concat((df_pdhg_mp, df_pdhg_vanilla, df_scs))
gen_solved_problems_plots_split_tol(
    df,
    f'{MITTELMANN_STR}_baselines',
    len(mittelmann_instances))
gen_total_solved_problems_table_split_tol(
    df, f'{MITTELMANN_STR}_baselines', PAR)

df_pdhg_scs_dir = pd.concat(
    (df_pdhg_mp[df_pdhg_mp['method'] == 'pdhg'], df_scs[df_scs['method'] == 'scs-direct']))
df_pdhg_scs_indir = pd.concat(
    (df_pdhg_mp[df_pdhg_mp['method'] == 'pdhg'], df_scs[df_scs['method'] == 'scs-indirect']))
gen_ratio_histograms_split_tol(df_pdhg_mp, f'{MITTELMANN_STR}', PAR)
gen_ratio_histograms_split_tol(df_pdhg_scs_indir, f'{MITTELMANN_STR}', PAR)
gen_ratio_histograms_split_tol(df_pdhg_scs_dir, f'{MITTELMANN_STR}', PAR)

######################################################################

# PDLP vs mp vs scs on NETLIB (JOIN PDHG/MP WITH SCS)
df_pdhg_mp = pd.read_csv(os.path.join(CSV_DIR, 'netlib_pdhg_mp_1h.csv'))
df_pdhg_mp = fill_in_missing_problems(df_pdhg_mp, netlib_instances)
df_pdhg_vanilla = pd.read_csv(os.path.join(
    CSV_DIR, 'netlib_pdhg_vanilla_1h.csv'))
df_pdhg_vanilla = fill_in_missing_problems(df_pdhg_vanilla, netlib_instances)
df_scs = pd.read_csv(os.path.join(CSV_DIR, 'netlib_scs_1h.csv'))
df_scs = fill_in_missing_problems(df_scs, netlib_instances)
df = pd.concat((df_pdhg_mp, df_pdhg_vanilla, df_scs))
gen_solved_problems_plots_split_tol(
    df,
    f'{NETLIB_STR}_baselines',
    len(netlib_instances))
gen_total_solved_problems_table_split_tol(
    df, f'{NETLIB_STR}_baselines', PAR)

df_pdhg_scs_dir = pd.concat(
    (df_pdhg_mp[df_pdhg_mp['method'] == 'pdhg'], df_scs[df_scs['method'] == 'scs-direct']))
df_pdhg_scs_indir = pd.concat(
    (df_pdhg_mp[df_pdhg_mp['method'] == 'pdhg'], df_scs[df_scs['method'] == 'scs-indirect']))
gen_ratio_histograms_split_tol(df_pdhg_mp, f'{NETLIB_STR}', PAR)
gen_ratio_histograms_split_tol(df_pdhg_scs_indir, f'{NETLIB_STR}', PAR)
gen_ratio_histograms_split_tol(df_pdhg_scs_dir, f'{NETLIB_STR}', PAR)

######################################################################

# PDLP presolve vs no presolve (JOIN DEFAULT)
df = pd.read_csv(os.path.join(CSV_DIR, 'miplib_nopresolve_100k.csv'))
df = pd.concat((df_default, df))
gen_solved_problems_plots_split_tol(
    df, f'{MIPLIB_STR}_presolve', len(miplib_instances))
gen_total_solved_problems_table_split_tol(df, f'{MIPLIB_STR}_presolve', PAR)

######################################################################

# PDLP scaling vs no scaling (NO JOIN DEFAULT)
df = pd.read_csv(os.path.join(CSV_DIR, 'miplib_scaling_100k.csv'))
df = fill_in_missing_problems(df, miplib_instances)

# Pull out best performing scaling for each instance / tolerance:
df_best_per = df[df['termination_reason'] == OPT].reset_index()
best_idxs = df_best_per.groupby(['instance_name', 'tolerance'])[
    'cumulative_kkt_matrix_passes'].idxmin()
df_best_per = df_best_per.loc[best_idxs]

for t in df_best_per['tolerance'].unique():
    # rename the experiment label
    df_best_per.loc[df_best_per['tolerance'] == t, 'experiment_label'] = \
        f'scaling {_BEST_STR} {t}'

df_best_per = fill_in_missing_problems(df_best_per, miplib_instances)
# filter out un-needed scaling experiments:
df = pd.concat(df[df['experiment_label'].str.contains(e)]
               for e in SCALING_EXPS_TO_USE)
gen_solved_problems_plots_split_tol(
    df, f'{MIPLIB_STR}_scaling', len(miplib_instances))
gen_total_solved_problems_table_split_tol(df, f'{MIPLIB_STR}_scaling', PAR)

df = pd.concat((df, df_best_per))
gen_solved_problems_plots_split_tol(
    df, f'{MIPLIB_STR}_scaling_with_best_per', len(miplib_instances))
gen_total_solved_problems_table_split_tol(
    df, f'{MIPLIB_STR}_scaling_with_best_per', PAR)

######################################################################

# PDLP restart vs no restart (NO JOIN DEFAULT)
df = pd.read_csv(os.path.join(CSV_DIR, 'miplib_restarts_100k.csv'))
df = fill_in_missing_problems(df, miplib_instances)
gen_solved_problems_plots_split_tol(
    df, f'{MIPLIB_STR}_restarts', len(miplib_instances))
gen_total_solved_problems_table_split_tol(df, f'{MIPLIB_STR}_restarts', PAR)

######################################################################

# PDLP primalweight (NO JOIN DEFAULT)
df = pd.read_csv(os.path.join(CSV_DIR, 'miplib_primalweight_100k.csv'))
df = fill_in_missing_problems(df, miplib_instances)
df_fixed = df[df['experiment_label'].str.contains('Fixed')]

pw_solved = df_fixed[df_fixed['termination_reason'] == OPT] \
    .groupby(['experiment_label', 'tolerance'])['experiment_label'] \
    .agg('count') \
    .pipe(pd.DataFrame) \
    .rename(columns={'experiment_label': 'solved'}) \
    .reset_index()
dfs = []
for t in df_fixed['tolerance'].unique():
    _df = pw_solved[pw_solved['tolerance'] == t]
    best_mp_run = _df.loc[_df['solved'].idxmax()]['experiment_label']
    dfs.append(df_fixed[df_fixed['experiment_label'] == best_mp_run])
df_best_ind = fill_in_missing_problems(pd.concat(dfs), miplib_instances)

for t in df_best_fixed['tolerance'].unique():
    # rename the experiment label
    df_best_ind.loc[df_best_ind['tolerance'] == t, 'experiment_label'] = \
        f'primalweight {_BEST_FIXED} {t}'

# Pull out best performing fixed weight for each instance / tolerance:
df_best_fixed = df_fixed[df_fixed['termination_reason'] == OPT].reset_index()
best_idxs = df_best_fixed.groupby(['instance_name', 'tolerance'])[
    'cumulative_kkt_matrix_passes'].idxmin()
df_best_fixed = df_best_fixed.loc[best_idxs]

for t in df_best_fixed['tolerance'].unique():
    # rename the experiment label
    df_best_fixed.loc[df_best_fixed['tolerance'] == t, 'experiment_label'] = \
        f'primalweight {_BEST_STR} {t}'
df_best_fixed = fill_in_missing_problems(df_best_fixed, miplib_instances)

df = pd.concat(df[df['experiment_label'].str.contains(e)]
               for e in PRIMALWEIGHT_EXPS_TO_USE)
df = pd.concat((df, df_best_fixed, df_best_ind))
gen_solved_problems_plots_split_tol(
    df, f'{MIPLIB_STR}_primalweight', len(miplib_instances), False)
gen_total_solved_problems_table_split_tol(
    df, f'{MIPLIB_STR}_primalweight', PAR)


######################################################################

# MIPLIB PDLP ablate improvements (JOIN DEFAULT)
df = pd.read_csv(os.path.join(CSV_DIR, 'miplib_improvements_100k.csv'))
df_pdlp = df_default.copy()
for t in df_pdlp['tolerance'].unique():
    df_pdlp.loc[df_pdlp['tolerance'] == t,
                'experiment_label'] = f'pdlp_final_improvements_{t}'
df = pd.concat((df, df_pdlp.reset_index()))
df = fill_in_missing_problems(df, miplib_instances)
gen_solved_problems_plots_split_tol(
    df, f'{MIPLIB_STR}_improvements', len(miplib_instances), True)
outputs = gen_total_solved_problems_table_split_tol(
    df, f'{MIPLIB_STR}_improvements', PAR)

gen_all_improvement_plots(outputs, f'{MIPLIB_STR}_improvements')

######################################################################

# MITTELMAN PDLP ablate improvements (JOIN DEFAULT)
df_default_mittelmann = pd.read_csv(
    os.path.join(
        CSV_DIR,
        'mittelmann_pdhg_enhanced_100k.csv'))
df_default_mittelmann = fill_in_missing_problems(
    df_default_mittelmann, mittelmann_instances)

df = pd.read_csv(os.path.join(CSV_DIR, 'mittelmann_improvements_100k.csv'))
df_pdlp = df_default_mittelmann.copy()
for t in df_pdlp['tolerance'].unique():
    df_pdlp.loc[df_pdlp['tolerance'] == t,
                'experiment_label'] = f'pdlp_final_improvements_{t}'
df = pd.concat((df, df_pdlp.reset_index()))
df = fill_in_missing_problems(df, mittelmann_instances)
gen_solved_problems_plots_split_tol(
    df,
    f'{MITTELMANN_STR}_improvements',
    len(mittelmann_instances),
    True)
outputs = gen_total_solved_problems_table_split_tol(
    df, f'{MITTELMANN_STR}_improvements', PAR)

for df in outputs.values():
    df['rank'] = df['Experiment'].map(IMPROVEMENTS_ORDER_IDX)
    df.sort_values('rank', inplace=True)
    df.drop('rank', 1, inplace=True)

gen_all_improvement_plots(outputs, f'{MITTELMANN_STR}_improvements')

######################################################################

# NETLIB PDLP ablate improvements (JOIN DEFAULT)
df_default_netlib = pd.read_csv(
    os.path.join(
        CSV_DIR,
        'netlib_pdhg_enhanced_100k.csv'))
df_default_netlib = fill_in_missing_problems(
    df_default_netlib, netlib_instances)

df = pd.read_csv(os.path.join(CSV_DIR, 'netlib_improvements_100k.csv'))
df_pdlp = df_default_netlib.copy()
for t in df_pdlp['tolerance'].unique():
    df_pdlp.loc[df_pdlp['tolerance'] == t,
                'experiment_label'] = f'pdlp_final_improvements_{t}'
df = pd.concat((df, df_pdlp.reset_index()))
df = fill_in_missing_problems(df, netlib_instances)
gen_solved_problems_plots_split_tol(
    df,
    f'{NETLIB_STR}_improvements',
    len(netlib_instances),
    True)
outputs = gen_total_solved_problems_table_split_tol(
    df, f'{NETLIB_STR}_improvements', PAR)

for df in outputs.values():
    df['rank'] = df['Experiment'].map(IMPROVEMENTS_ORDER_IDX)
    df.sort_values('rank', inplace=True)
    df.drop('rank', 1, inplace=True)

gen_all_improvement_plots(outputs, f'{NETLIB_STR}_improvements')
