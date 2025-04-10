#plotting rewards (mean and cumulative mean with SD); actions (C or D) and actions types (in response to state) 
import numpy as np
import pandas as pd
import matplotlib 
import matplotlib.pyplot as plt 
import seaborn as sns 
from matplotlib.pyplot import figure
import dataframe_image as dfi
import os 
import glob 
from collections import Counter 
import pickle 
#import pickle5 as pickle #for earlier versions of python
from ast import literal_eval as make_tuple
#from ordered_set import OrderedSet

import matplotlib as mpl
mpl.rcParams['agg.path.chunksize'] = 10000

from statistics import mean 
import networkx as nx
from math import isnan
from collections import Counter

sns.set_theme(font_scale=3)

SAVE_FIGURES_PATH = 'MyFigures'

#################################
#### New plots for LLM Study ####
#################################

def process_CD_tokens(run_idxs, num_episodes, PART, extra, moral_type=None):
    colnames = ['C|C', 'C|D', 'D|C', 'D|D', 'illegal|C', 'illegal|D', 'C|illegal', 'D|illegal', 'illegal|illegal']
    results_allruns = pd.DataFrame(index=range(num_episodes), columns=colnames)

    for run_idx in run_idxs: 
        if PART == '2then3': 
            during_PT2 = pd.read_csv(f'run{run_idx}/During FT PART2 (Game rewards).csv', index_col=0)
            during_PT3 = pd.read_csv(f'run{run_idx}/During FT PART3 ({moral_type} rewards).csv', index_col=0)
            during_PT3['episode'] = during_PT3['episode'] + num_episodes/2
            df = pd.concat([during_PT2, during_PT3], ignore_index=True)
        else:
            during_PTx = pd.read_csv(f'run{run_idx}/During FT PART{PART} {extra}.csv', index_col=0)
            df = during_PTx.copy()
        results_1run = pd.DataFrame(index=range(num_episodes), columns=colnames)

        #split up the 'C&D strings' column into two, if it is there 
        if 'C & D strings' in df.columns:
            temp = df['C & D strings'].str.split(',', expand=True)
            for col in temp:
                temp[col] = temp[col].str.strip('[]')
                temp[col] = temp[col].str.replace("'", "")
            df[['C_str', 'D_str']] = temp[[0, 1]]

        for row_idx in range(len(df['action_M'])):
            if df['action_M'][row_idx] == df['C_str'][row_idx]:
                df['action_M'][row_idx] = df['action_M'][row_idx].replace(df['C_str'][row_idx], 'C')
            elif df['action_M'][row_idx] == df['D_str'][row_idx]:
                df['action_M'][row_idx] = df['action_M'][row_idx].replace(df['D_str'][row_idx], 'D')

        for row_idx in range(len(df['prev_move_O'])):
            if df['prev_move_O'][row_idx] == df['C_str'][row_idx]:
                df['prev_move_O'][row_idx] = df['prev_move_O'][row_idx].replace(df['C_str'][row_idx], 'C')
            elif df['prev_move_O'][row_idx] == df['D_str'][row_idx]:
                df['prev_move_O'][row_idx] = df['prev_move_O'][row_idx].replace(df['D_str'][row_idx], 'D')

        #relabel all illegal tokens as 'illegal' 
        df['action_M_clean'] = df['action_M'].apply(lambda x: 'illegal' if x not in ['C', 'D'] else x)
        df['prev_move_O_clean'] = df['prev_move_O'].apply(lambda x: 'illegal' if x not in ['C', 'D'] else x)

        #combine two str columns into a third column and add a '|' between them 
        df['action_M | state'] = [str(df['action_M_clean'][i]) + '|' + str(df['prev_move_O_clean'][i]) for i in range(len(df))]

        #aggregate per episode 
        for episode in range(num_episodes):
            episode_result = df[df['episode']==episode]['action_M | state'].value_counts()
            for combination in episode_result.index:
                results_1run.loc[episode, combination] = episode_result[combination]

        results_allruns = results_allruns.add(results_1run, fill_value=0)

    results_allruns.to_csv(f'actions_given_state_PT{PART}.csv')




def plot_C_tokens_per_episode(df_original, title):
    df = df_original.copy()
    df['action_M'].value_counts()

    df['action_M_C'] = df.apply(lambda row: str(row['action_M']).strip()=='X', axis=1)

    df_grouped = df.groupby('episode').agg({'action_M_C': 'sum', 'action_M':'count'})
    
    perc_C = df_grouped['action_M_C'] * 100 / df_grouped['action_M']

    plt.plot(perc_C, label='cooperation by player M', color='darkgreen', linewidth=0.3)
    
    plt.title(f'Cooperation by player M \n {title}') 
    plt.gca().set_ylim([0, 100])
    plt.ylabel(r'% cooperating per episode') #\n (mean over '+str(n_runs)+r' runs +- CI)')
    plt.xlabel('Episode')



def plot_action_types_per_episode(num_episodes, run_idx, PART, ax=None):
    #do the below in a separate function now 
    if False: 
        df=df_original.copy()
        colnames=['C|C', 'C|D', 'D|C', 'D|D', 'illegal|C', 'illegal|D', 'C|illegal', 'D|illegal', 'illegal|illegal']
        results = pd.DataFrame(index=range(num_episodes), columns=colnames)

        df['action_M'].replace(C_symbol, 'C', inplace=True)
        df['action_M'].replace(D_symbol, 'D', inplace=True)
        df['prev_move_O'].replace(C_symbol, 'C', inplace=True)
        df['prev_move_O'].replace(D_symbol, 'D', inplace=True)

        #relabel all illegal tokens as 'illegal' 
        df['action_M_clean'] = df['action_M'].apply(lambda x: 'illegal' if x not in ['C', 'D'] else x)
        df['prev_move_O_clean'] = df['prev_move_O'].apply(lambda x: 'illegal' if x not in ['C', 'D'] else x)

        #combine two str columns into a third column and add a '|' between them 
        df['action_M | state'] = [str(df['action_M_clean'][i]) + '|' + str(df['prev_move_O_clean'][i]) for i in range(len(df))]

        for episode in range(num_episodes):
            episode_result = df[df['episode']==episode]['action_M | state'].value_counts()
            for combination in episode_result.index:
                results.loc[episode, combination] = episode_result[combination]
            
    results = pd.read_csv(f'actions_given_state_PT{PART}.csv', index_col=0)

    #results_counts = results.transpose().apply(pd.value_counts).transpose()[1:] 
    #plot after first episode, as in the first apisode they are reacting to default state=0
    results.dropna(axis=1, how='all', inplace=True)

    #plt.figure(dpi=80, figsize=(10, 8))
    #plt.rcParams.update({'font.size':10})
    #plt.figure(figsize=(20, 15), dpi=100)
    results.plot.area(stacked=True, ylabel = '# times M took this type of action', rot=45, linewidth=0.4, alpha=0.8,
        xlabel='Episode', color={'C|C':'#28641E', 'C|D':'#B0DC82', 'D|C':'#FBE6F1', 'D|D':'#8E0B52', 
                                 'illegal|C':'#A9A9A9', 'illegal|D':'#A9A9A9', 'C|illegal':'#A9A9A9', 'D|illegal':'#A9A9A9', 'illegal|illegal':'#A9A9A9'}, #EAED4
        #colormap='PiYG_r',
        #color={'C | (C,C)':'lightblue', 'C | (C,D)':'deepskyblue', 'C | (D,C)':'cyan', 'C | (D,D)':'royalblue', 'D | (C,C)':'orange', 'D | (C,D)':'yellow', 'D | (D,C)':'peru', 'D | (D,D)':'dimgrey'}, 
        #title='Types of actions over time: \n ' + 
        title = f'\n Run {run_idx}', 
        ax=ax
        ).legend(fontsize=9, loc='center') #Pairs of simultaneous actions over time:
    #plt.savefig(f'{destination_folder}/plots/action_types_area_player1.png', bbox_inches='tight')

def plot_action_types_perepisode_aggruns(run_idxs, PART, opponent, extra, addition=None, legend=False):
    if addition: 
        addition = '\n' + addition
    else: 
        addition = ''

    title=f"LLM's actions during \n {extra} fine-tuning \n vs {opponent} opponent"
    #do the below in a separate function now 
    results_allruns = pd.read_csv(f'actions_given_state_PT{PART}.csv', index_col=0)          
    if len(results_allruns) == 500:
        xticks = [0, 250, 500]
    elif len(results_allruns) == 1000:
        xticks = [0, 500, 1000]
    #results_counts = results.transpose().apply(pd.value_counts).transpose()[1:] 
    #plot after first episode, as in the first apisode they are reacting to default state=0
    results_allruns.dropna(axis=1, how='all', inplace=True)

    if legend=='outside':
        plt.figure(dpi=80, figsize=(8, 4))
    else: 
        plt.figure(dpi=80, figsize=(10, 4))
    #plt.rcParams.update({'font.size':10})
    #plt.figure(figsize=(20, 15), dpi=100)
    results_allruns.plot.area(stacked=True, 
                              ylabel = f"M's action | O's prev. move \n (sum over {len(run_idxs)} runs)", 
                              rot=0, linewidth=0.4, alpha=0.8,
                              xlabel='Episode', color={'C|C':'#28641E', 'C|D':'#B0DC82', 'D|C':'#FBE6F1', 'D|D':'#8E0B52', 
                                 'illegal|C':'#A9A9A9', 'illegal|D':'#A9A9A9', 'C|illegal':'#A9A9A9', 'D|illegal':'#A9A9A9', 'illegal|illegal':'#A9A9A9'}, #EAED4
        #colormap='PiYG_r',
        #color={'C | (C,C)':'lightblue', 'C | (C,D)':'deepskyblue', 'C | (D,C)':'cyan', 'C | (D,D)':'royalblue', 'D | (C,C)':'orange', 'D | (C,D)':'yellow', 'D | (D,C)':'peru', 'D | (D,D)':'dimgrey'}, 
        title= title + f'{addition}')
    #.legend(fontsize=10, loc='center') #Pairs of simultaneous actions over time:
    plt.xticks(xticks)
    if legend=='outside':
        plt.legend(fontsize=25, loc='right', bbox_to_anchor=(1.6, 0.5))
        legendloc='legoutside'
    elif legend=='inside':
        plt.legend(fontsize=15, loc='center')
        legendloc='leginside'
    elif legend==False: 
        #do not plot legend 
        plt.legend().set_visible(False)
        legendloc='noleg'

    if not os.path.isdir('plots'):
        os.makedirs('plots') 
    plt.savefig(f'plots/action_types_area_PT{PART}{extra}_opp{opponent}{addition.strip()}{legendloc}.png', bbox_inches='tight')
    plt.show()

def plot_action_types_perepisode_separateruns(run_idxs, num_episodes, opponent, do_PT1, do_PT2, do_PT3, do_PT4):

    if do_PT2:
        fig, axs = plt.subplots(nrows=1, ncols=len(run_idxs), sharey=True, sharex=True, figsize=(4*len(run_idxs),2.5))
        plt.suptitle(f'Action types (action | opp_prev_move) over time on each run  \n vs {opponent} opponent (during PT2 of fine-tuning)', y=1.4, fontsize=29)
        plt.xlabel('Episode')

        for run_idx in run_idxs:
            ax = axs[run_idx-1]
            ax.set_label(f'Run {run_idx}')
            #PART 1
            #during_PT2 = pd.read_csv(f'run{run_idx}/During FT PART2 (IPD rewards).csv', index_col=0)
            plot_action_types_per_episode(num_episodes, run_idx, C_symbol, D_symbol, ax)
        plt.savefig(f'plots/action_types_area_separateruns_PT2.png', bbox_inches='tight')

    if do_PT3:
        fig, axs = plt.subplots(nrows=1, ncols=len(run_idxs), sharey=True, sharex=True, figsize=(4*len(run_idxs),2.5))
        plt.suptitle(f'Action types (action | opp_prev_move) over time on each run  \n vs {opponent} opponent (during PT3 of fine-tuning)', y=1.4, fontsize=29)
        plt.xlabel('Episode')

        for run_idx in run_idxs:
            ax = axs[run_idx-1]
            ax.set_label(f'Run {run_idx}')
            #PART 3
            #during_PT3 = pd.read_csv(f'run{run_idx}/During FT PART3 (De rewards).csv', index_col=0)
            plot_action_types_per_episode(num_episodes, run_idx, PART=3, ax=ax)
        plt.savefig(f'plots/action_types_area_separateruns_PT3.png', bbox_inches='tight')

    if do_PT4:
        fig, axs = plt.subplots(nrows=1, ncols=len(run_idxs), sharey=True, sharex=True, figsize=(4*len(run_idxs),2.5))
        plt.suptitle(f'Action types (action | opp_prev_move) over time on each run  \n vs {opponent} opponent (during PT3 of fine-tuning)', y=1.4, fontsize=29)
        plt.xlabel('Episode')

        for run_idx in run_idxs:
            ax = axs[run_idx-1]
            ax.set_label(f'Run {run_idx}')
            #PART 3
            #during_PT3 = pd.read_csv(f'run{run_idx}/During FT PART3 (De rewards).csv', index_col=0)
            plot_action_types_per_episode(num_episodes, run_idx, PART=4, ax=ax)
        plt.savefig(f'plots/action_types_area_separateruns_PT4.png', bbox_inches='tight')



def plot_reward_per_episode_playerM(run_idxs, num_episodes, opponent, PART, extra, with_CI=True):
    '''explore the Intrinsic Reward obtained by the learning LLM agent at every episode - what percentage are cooperating? '''
    
    #combine each player's actions into one df: 
    result_df = pd.DataFrame(index=range(num_episodes)) 
    n_runs = len(run_idxs)
    for run_idx in run_idxs:
        run_df = pd.read_csv(f'run{run_idx}/During FT PART{PART} {extra}.csv', index_col=0)[['episode', 'reward_M']]

        #group by episode 
        run_grouped = run_df.groupby('episode').agg({'reward_M': 'mean'})

        result_df[f'run{run_idx}'] = run_grouped['reward_M']


    if with_CI:
        means = result_df.mean(axis=1)
        sds = result_df.std(axis=1)
        ci = 1.96 * sds/np.sqrt(n_runs)
        result_df['R_mean'] = means
        result_df['R_ci'] = ci
    else: 
        result_df['R_mean'] = result_df.mean(axis=1)

    if PART ==2: 
        title = 'Extrinsic (IPD Game) reward'
    elif PART == 3: 
        title = 'Intrinsic (Moral) reward '
    elif PART == 4: 
        title = 'Combined Reward (IPD + Moral reward)'
   
    #plot results 
    plt.figure(dpi=80, figsize=(7, 5)) 
    if with_CI:
        plt.plot(result_df.index[:], result_df['R_mean'], label=f'vs {opponent}', color='blue', linewidth=0.5)
        plt.fill_between(result_df.index[:], result_df['R_mean']-result_df['R_ci'], result_df['R_mean']+result_df['R_ci'], facecolor='lightblue', linewidth=0.04, alpha=0.4)
    else: 
        plt.plot(result_df.index[:], result_df['R_mean'], label=f'vs {opponent}', color='purple', linewidth=0.05)

    plt.title(f'{title} \n obtained by LLM agent \n learning vs {opponent} opponent ') 
    #plt.gca().set_ylim([-11, 5])
    if with_CI:
        plt.ylabel(r'Intrinsic Reward'+ '  \n (mean over '+str(n_runs)+r' runs +- CI)')
    else: 
        plt.ylabel(r'Intrinsic Reward'+ '  \n (mean over '+str(n_runs)+r' runs)')
    plt.xlabel('Episode')

    if not os.path.isdir(f'plots'):
        os.makedirs(f'plots')

    plt.savefig(f'plots/episode_reward_during_learning.png', bbox_inches='tight')

def plot_bar_reward_after_playerM(run_idxs, opponent, PART, extra):
    #bs_eval=10, num_episodes=1
    '''assume only one episode was played on the eval stage '''
    if PART ==2: 
        title = 'Extrinsic (IPD Game) reward'
        header1 = 'rewards_Game (before)'
        header2 = 'rewards_Game (after)'
    elif PART == 3: 
        title = 'Intrinsic (Moral) reward'
        header1 = 'rewards_De (before)'
        header2 = 'rewards_De (after)'
    elif PART == 4: 
        title = 'Combined (IPD + Moral) reward'
        header1 = 'rewards_De&Game (before)'
        header2 = 'rewards_De&Game (after)'
        extra = '(De & IPD rewards)'
    result_df = pd.DataFrame(index=['Ref \n model', 'Fine-tuned \n model'], columns=[f'run{r}' for r in run_idxs]) 
    
    n_runs = len(run_idxs)
    for run_idx in runs:
        run_df = pd.read_csv(f'run{run_idx}/After FT PART{PART} {extra}.csv', index_col=0)#[['episode', 'reward_M']]

        #group by episode 
        #run_grouped = run_df.groupby('episode').agg({'reward_M': 'mean'})
        result_df[f'run{run_idx}']['Ref \n model'] = run_df[str(header1)].mean()
        result_df[f'run{run_idx}']['Fine-tuned \n model'] = run_df[str(header2)].mean()
 
    means = result_df.mean(axis=1)
    sds = result_df.std(axis=1)
    ci = 1.96 * sds/np.sqrt(n_runs)

    plt.figure(dpi=80, figsize=(5, 4)) 
    result_df.mean(axis=1).plot(kind='bar', ylim=[-12, 5], color='lightblue', linewidth=0.5, rot=0)
    #plt.bar(result_df.index, result_df.mean(axis=1))
    plt.errorbar(result_df.index, result_df.mean(axis=1), yerr=ci, fmt=",", color="b")
    plt.title(f'{title} \n obtained by fine-tuned LLM agent \n (fine-tuned vs {opponent} opponent) \n vs reference model ') 
    plt.ylabel(f'{title}'+ '\n (mean over '+str(n_runs)+r' runs +- CI)')
    #plt.xlabel('')

    plt.savefig(f'plots/reward_after.png', bbox_inches='tight')

def plot_bar_subfigs_reward_after(opponent): ## NOTE this might be unfiished 
    '''assume only one episode was played on the eval stage '''

    fig, axs = plt.subplots(nrows=1, ncols=3, sharey=False, sharex=False, figsize=(9, 3))
    fig.tight_layout() 
    plt.suptitle(f'LLM agent fine-tuned vs {opponent} opponent, compared to reference model ', fontsize=18, y=1.25)

    idx = -1 
    for PART in [2,3,4]:
        idx += 1 
        if PART ==2: 
            title = 'Fine-tuned on \n Extrinsic (IPD Game) \n reward'
            header1 = 'rewards_Game (before)'
            header2 = 'rewards_Game (after)'
            PARTs_detail = '_PT2'
            #extra = '(Game rewards)'
            extra = '(IPD rewards)'
            run_idxs = [1,2,3,5,6]
        elif PART == 3: 
            title = 'Fine-tuned on \n Intrinsic (Moral) \n reward'
            header1 = 'rewards_De (before)'
            header2 = 'rewards_De (after)'
            PARTs_detail = '_PT3'
            extra = '(De rewards)'
            run_idxs = [1,2,3,5,6]
        elif PART == 4: 
            title = 'Fine-tuned on \n Combined (IPD + Moral) \n reward'
            header1 = 'rewards_De&Game (before)'
            header2 = 'rewards_De&Game (after)'
            extra = '(De & IPD rewards)'
            PARTs_detail = '_PT4'
            #extra = '(De&Game rewards)'
            run_idxs = [1,2,3,5,6]

        os.chdir(f'../gemma-2-2b-it_FT_{PARTs_detail}_opp{opponent}')
        result_df = pd.DataFrame(index=['Ref \n model', 'Fine-tuned \n model'], columns=[f'run{r}' for r in run_idxs]) 
        
        n_runs = len(run_idxs)
        for run_idx in run_idxs:
            run_df = pd.read_csv(f'run{run_idx}/After FT PART{PART} {extra}.csv', index_col=0)#[['episode', 'reward_M']]

            #group by episode 
            result_df[f'run{run_idx}']['Ref \n model'] = run_df[str(header1)].mean()
            result_df[f'run{run_idx}']['Fine-tuned \n model'] = run_df[str(header2)].mean()
    
        means = result_df.mean(axis=1)
        sds = result_df.std(axis=1)
        ci = 1.96 * sds/np.sqrt(n_runs)


        result_df.mean(axis=1).plot(ax=axs[idx], kind='bar', ylim=[-12, 5], color='lightblue', linewidth=0.5, rot=0, fontsize=15)
        #plt.bar(result_df.index, result_df.mean(axis=1))
        axs[idx].errorbar(result_df.index, result_df.mean(axis=1), yerr=ci, fmt="o", color="b")
        axs[idx].set_title(f'{title}', fontsize=15) 
        #axs[idx].set_title(f'{title} \n obtained by fine-tuned LLM agent \n (fine-tuned vs {opponent} opponent) \n vs reference model ') 
        axs[idx].set_ylabel(f'{title}'.replace('\n','').replace('Fine-tuned on ', '')+ '\n (mean over '+str(n_runs)+r' runs +- CI)', fontsize=15)

    if not os.path.isdir(f'../plots'):
        os.makedirs(f'../plots')
    plt.savefig(f'../plots/reward_after_subfigs_all3parts_{opponent}opp.png', bbox_inches='tight')




def caluclate_regret_after(game, PARTs_detail, opponent, option, num_episodes_trained, r_toplot, moral_max, label, extra='', moral_min=None):
    allruns_singlepart = pd.DataFrame(columns=[f'run{run_idx}' for run_idx in [1,2,3,5,6]], index=[f'{label} Moral Regret'])
    os.chdir(f'../gemma-2-2b-it_FT_{PARTs_detail}_opp{opponent}_{num_episodes_trained}ep_{option}')
    for run_idx in [1,2,3,5,6]:
        run_df = pd.read_csv(f'run{run_idx}{extra}/EVAL After FT {PARTs_detail} - independent eval {game}.csv', index_col=0)
        run_episodes = run_df[['episode', f'{r_toplot} (after)']].groupby('episode').mean()
        run_episodes_regret = moral_max - run_episodes.reset_index()[f'{r_toplot} (after)']
        if label == 'Utilitarian': run_episodes_regret = run_episodes_regret /  (moral_max - moral_min)
        #store this run's value for Moral Regret (after)
        allruns_singlepart[f'run{run_idx}'][f'{label} Moral Regret'] = run_episodes_regret.mean()
    return allruns_singlepart.mean(axis=1)[f'{label} Moral Regret'], 1.96 * allruns_singlepart.std(axis=1)[f'{label} Moral Regret'] / np.sqrt(5)


def plot_regret_allgames(r_toplot, label, opponent, options, extra, include_NoFT, include_Prompted=False):
    games = ['IPD', 'ISH', 'ICN', 'BOS', 'ICD']
    columns = ['No fine-tuning', 'Game payoffs', 'Deontological', 'Utilitarian', 'Game + \nDeontological', 'Game, then \nDeontological', 'Game, then \nUtilitarian']
    if include_Prompted == True: 
        columns.insert(1, 'No fine-tuning, \nDeont. prompted')
        columns.insert(2, 'No fine-tuning, \nUtilit. prompted')
    all_runs = pd.DataFrame(index=games, columns=columns)
    all_runs_ci = pd.DataFrame(index=games, columns=columns)


    if label=='Deontological': 
        moral_max = {'IPD': 0, 'ISH': 0, 'ICN':0, 'BOS':0, 'ICD':0} 
        moral_min = {'IPD': -6, 'ISH': -6, 'ICN':-6, 'BOS':-6, 'ICD':-6}
    elif label == 'Utilitarian':
        moral_max = {'IPD': 6, 'ISH': 8, 'ICN':5, 'BOS':5, 'ICD':8}
        moral_min = {'IPD': -6, 'ISH': -6, 'ICN':-6, 'BOS':-6, 'ICD':-6}

    ylab = 'No fine-tuning'
    PARTs_detail = '_PT2'
    num_episodes_trained = 1000
    game = 'IPD'
    allruns_singlepart = pd.DataFrame(columns=[f'run{run_idx}' for run_idx in [1,2,3,5,6]], index=[f'{label} Moral Regret'])
    try: 
        os.chdir(f'gemma-2-2b-it_FT_{PARTs_detail}_opp{opponent}_{num_episodes_trained}ep_{options[0]}')
    except:
        os.chdir(f'../gemma-2-2b-it_FT_{PARTs_detail}_opp{opponent}_{num_episodes_trained}ep_{options[0]}')
    for run_idx in [1,2,3,5,6]:
        run_df = pd.read_csv(f'run{run_idx}{extra}/EVAL After FT {PARTs_detail} - independent eval {game}.csv', index_col=0)
        run_episodes = run_df[['episode', f'{r_toplot} (before)']].groupby('episode').mean()
        run_episodes_regret = moral_max[game] - run_episodes.reset_index()[f'{r_toplot} (before)']
        if label == 'Utilitarian': run_episodes_regret = run_episodes_regret /  (moral_max[game] - moral_min[game])
        #store this run's value for Moral Regret (before)
        allruns_singlepart[f'run{run_idx}'][f'{label} Moral Regret'] = run_episodes_regret.mean()
    all_runs.loc['IPD'][ylab] = allruns_singlepart.mean(axis=1)[f'{label} Moral Regret']
    all_runs_ci.loc['IPD'][ylab] = 1.96 * allruns_singlepart.std(axis=1)[f'{label} Moral Regret'] / np.sqrt(5)

    game = 'ISH'
    allruns_singlepart = pd.DataFrame(columns=[f'run{run_idx}' for run_idx in [1,2,3,5,6]], index=[f'{label} Moral Regret'])
    os.chdir(f'../gemma-2-2b-it_FT_{PARTs_detail}_opp{opponent}_{num_episodes_trained}ep_{options[0]}')
    for run_idx in [1,2,3,5,6]:
        run_df = pd.read_csv(f'run{run_idx}{extra}/EVAL After FT {PARTs_detail} - independent eval {game}.csv', index_col=0)
        run_episodes = run_df[['episode', f'{r_toplot} (before)']].groupby('episode').mean()
        run_episodes_regret = moral_max[game] - run_episodes.reset_index()[f'{r_toplot} (before)']
        if label == 'Utilitarian': run_episodes_regret = run_episodes_regret /  (moral_max[game] - moral_min[game])
        allruns_singlepart[f'run{run_idx}'][f'{label} Moral Regret'] = run_episodes_regret.mean()
    all_runs.loc['ISH'][ylab] = allruns_singlepart.mean(axis=1)[f'{label} Moral Regret']
    all_runs_ci.loc['ISH'][ylab] = 1.96 * allruns_singlepart.std(axis=1)[f'{label} Moral Regret'] / np.sqrt(5)

    game = 'ICN'
    allruns_singlepart = pd.DataFrame(columns=[f'run{run_idx}' for run_idx in [1,2,3,5,6]], index=[f'{label} Moral Regret'])
    os.chdir(f'../gemma-2-2b-it_FT_{PARTs_detail}_opp{opponent}_{num_episodes_trained}ep_{options[0]}')
    for run_idx in [1,2,3,5,6]:
        run_df = pd.read_csv(f'run{run_idx}{extra}/EVAL After FT {PARTs_detail} - independent eval {game}.csv', index_col=0)
        run_episodes = run_df[['episode', f'{r_toplot} (before)']].groupby('episode').mean()
        run_episodes_regret = moral_max[game] - run_episodes.reset_index()[f'{r_toplot} (before)']
        if label == 'Utilitarian': run_episodes_regret = run_episodes_regret /  (moral_max[game] - moral_min[game])
        allruns_singlepart[f'run{run_idx}'][f'{label} Moral Regret'] = run_episodes_regret.mean()
    all_runs.loc['ICN'][ylab] = allruns_singlepart.mean(axis=1)[f'{label} Moral Regret']
    all_runs_ci.loc['ICN'][ylab] = 1.96 * allruns_singlepart.std(axis=1)[f'{label} Moral Regret'] / np.sqrt(5)

    game = 'BOS'
    allruns_singlepart = pd.DataFrame(columns=[f'run{run_idx}' for run_idx in [1,2,3,5,6]], index=[f'{label} Moral Regret'])
    os.chdir(f'../gemma-2-2b-it_FT_{PARTs_detail}_opp{opponent}_{num_episodes_trained}ep_{options[0]}')
    for run_idx in [1,2,3,5,6]:
        run_df = pd.read_csv(f'run{run_idx}{extra}/EVAL After FT {PARTs_detail} - independent eval {game}.csv', index_col=0)
        run_episodes = run_df[['episode', f'{r_toplot} (before)']].groupby('episode').mean()
        run_episodes_regret = moral_max[game] - run_episodes.reset_index()[f'{r_toplot} (before)']
        if label == 'Utilitarian': run_episodes_regret = run_episodes_regret /  (moral_max[game] - moral_min[game])
        allruns_singlepart[f'run{run_idx}'][f'{label} Moral Regret'] = run_episodes_regret.mean()
    all_runs.loc['BOS'][ylab] = allruns_singlepart.mean(axis=1)[f'{label} Moral Regret']
    all_runs_ci.loc['BOS'][ylab] = 1.96 * allruns_singlepart.std(axis=1)[f'{label} Moral Regret'] / np.sqrt(5)

    game = 'ICD'
    allruns_singlepart = pd.DataFrame(columns=[f'run{run_idx}' for run_idx in [1,2,3,5,6]], index=[f'{label} Moral Regret'])
    os.chdir(f'../gemma-2-2b-it_FT_{PARTs_detail}_opp{opponent}_{num_episodes_trained}ep_{options[0]}')
    for run_idx in [1,2,3,5,6]:
        run_df = pd.read_csv(f'run{run_idx}{extra}/EVAL After FT {PARTs_detail} - independent eval {game}.csv', index_col=0)
        run_episodes = run_df[['episode', f'{r_toplot} (before)']].groupby('episode').mean()
        run_episodes_regret = moral_max[game] - run_episodes.reset_index()[f'{r_toplot} (before)']
        if label == 'Utilitarian': run_episodes_regret = run_episodes_regret /  (moral_max[game] - moral_min[game])
        allruns_singlepart[f'run{run_idx}'][f'{label} Moral Regret'] = run_episodes_regret.mean()
    all_runs.loc['ICD'][ylab] = allruns_singlepart.mean(axis=1)[f'{label} Moral Regret']
    all_runs_ci.loc['ICD'][ylab] = 1.96 * allruns_singlepart.std(axis=1)[f'{label} Moral Regret'] / np.sqrt(5)


    if include_Prompted == True: 
        game = 'IPD'
        for game in games:
            ylab = 'No fine-tuning, \nDeont. prompted'
            all_runs.loc[game][ylab], all_runs_ci.loc['IPD'][ylab] =  mini_calculateregret_new_baselines(game, r_toplot, moral_max, moral_min, label, value='De')
            ylab = 'No fine-tuning, \nUtilit. prompted'
            all_runs.loc[game][ylab], all_runs_ci.loc['IPD'][ylab] =  mini_calculateregret_new_baselines(game, r_toplot, moral_max, moral_min, label, value='Ut')



    game='IPD'
    all_runs.loc['IPD']['Game payoffs'], all_runs_ci.loc['IPD']['Game payoffs'] = caluclate_regret_after(game=game, PARTs_detail = '_PT2', opponent=opponent, option = options[0], num_episodes_trained=1000, r_toplot=r_toplot, moral_max=moral_max[game], label=label, extra=extra,  moral_min=moral_min[game])
    all_runs.loc['IPD']['Deontological'], all_runs_ci.loc['IPD']['Deontological'] = caluclate_regret_after(game=game, PARTs_detail = '_PT3', opponent=opponent, option = options[0], num_episodes_trained=1000, r_toplot=r_toplot, moral_max=moral_max[game], label=label, extra=extra,  moral_min=moral_min[game])
    try:
        all_runs.loc['IPD']['Utilitarian'], all_runs_ci.loc['IPD']['Utilitarian'] = caluclate_regret_after(game=game, PARTs_detail = '_PT3', opponent=opponent, option = options[1], num_episodes_trained=1000, r_toplot=r_toplot, moral_max=moral_max[game], label=label, extra=extra,  moral_min=moral_min[game])
        all_runs.loc['IPD']['Game + \nDeontological'], all_runs_ci.loc['IPD']['Game + \nDeontological'] = caluclate_regret_after(game=game, PARTs_detail = '_PT4', opponent=opponent, option = options[0], num_episodes_trained=1000, r_toplot=r_toplot, moral_max=moral_max[game], label=label, extra=extra,  moral_min=moral_min[game])
        all_runs.loc['IPD']['Game, then \nDeontological'], all_runs_ci.loc['IPD']['Game, then \nDeontological'] = caluclate_regret_after(game=game, PARTs_detail = '_PT3after2', opponent=opponent, option = options[0], num_episodes_trained=500, r_toplot=r_toplot, moral_max=moral_max[game], label=label, extra=extra,  moral_min=moral_min[game])
        all_runs.loc['IPD']['Game, then \nUtilitarian'], all_runs_ci.loc['IPD']['Game, then \nUtilitarian']= caluclate_regret_after(game=game, PARTs_detail = '_PT3after2', opponent=opponent, option = options[1], num_episodes_trained=500, r_toplot=r_toplot, moral_max=moral_max[game], label=label, extra=extra,  moral_min=moral_min[game])
    except: pass

    game='ISH'
    all_runs.loc['ISH']['Game payoffs'], all_runs_ci.loc['ISH']['Game payoffs'] = caluclate_regret_after(game=game, PARTs_detail = '_PT2', opponent=opponent, option = options[0], num_episodes_trained=1000, r_toplot=r_toplot, moral_max=moral_max[game], label=label, extra=extra,  moral_min=moral_min[game])
    all_runs.loc['ISH']['Deontological'], all_runs_ci.loc['ISH']['Deontological'] = caluclate_regret_after(game=game, PARTs_detail = '_PT3', opponent=opponent, option = options[0], num_episodes_trained=1000, r_toplot=r_toplot, moral_max=moral_max[game], label=label, extra=extra,  moral_min=moral_min[game])
    try: 
        all_runs.loc['ISH']['Utilitarian'], all_runs_ci.loc['ISH']['Utilitarian'] = caluclate_regret_after(game=game, PARTs_detail = '_PT3', opponent=opponent, option = options[1], num_episodes_trained=1000, r_toplot=r_toplot, moral_max=moral_max[game], label=label, extra=extra,  moral_min=moral_min[game])
        all_runs.loc['ISH']['Game + \nDeontological'],  all_runs_ci.loc['ISH']['Game + \nDeontological'] = caluclate_regret_after(game=game, PARTs_detail = '_PT4', opponent=opponent, option = options[0], num_episodes_trained=1000, r_toplot=r_toplot, moral_max=moral_max[game], label=label, extra=extra,  moral_min=moral_min[game])
        all_runs.loc['ISH']['Game, then \nDeontological'], all_runs_ci.loc['ISH']['Game, then \nDeontological'] = caluclate_regret_after(game=game, PARTs_detail = '_PT3after2', opponent=opponent, option = options[0], num_episodes_trained=500, r_toplot=r_toplot, moral_max=moral_max[game], label=label, extra=extra,  moral_min=moral_min[game])
        all_runs.loc['ISH']['Game, then \nUtilitarian'], all_runs_ci.loc['ISH']['Game, then \nUtilitarian'] = caluclate_regret_after(game=game, PARTs_detail = '_PT3after2', opponent=opponent, option = options[1], num_episodes_trained=500, r_toplot=r_toplot, moral_max=moral_max[game],  label=label, extra=extra,  moral_min=moral_min[game])
    except: pass

    game='ICN'
    all_runs.loc['ICN']['Game payoffs'], all_runs_ci.loc['ICN']['Game payoffs'] = caluclate_regret_after(game=game, PARTs_detail = '_PT2', opponent=opponent, option = options[0], num_episodes_trained=1000, r_toplot=r_toplot, moral_max=moral_max[game], label=label, extra=extra,  moral_min=moral_min[game])
    all_runs.loc['ICN']['Deontological'], all_runs_ci.loc['ICN']['Deontological'] = caluclate_regret_after(game=game, PARTs_detail = '_PT3', opponent=opponent, option = options[0], num_episodes_trained=1000, r_toplot=r_toplot, moral_max=moral_max[game],  label=label, extra=extra,  moral_min=moral_min[game])
    try: 
        all_runs.loc['ICN']['Utilitarian'], all_runs_ci.loc['ICN']['Utilitarian'] = caluclate_regret_after(game=game, PARTs_detail = '_PT3', opponent=opponent, option = options[1], num_episodes_trained=1000, r_toplot=r_toplot, moral_max=moral_max[game],  label=label, extra=extra,  moral_min=moral_min[game])
        all_runs.loc['ICN']['Game + \nDeontological'], all_runs_ci.loc['ICN']['Game + \nDeontological'] = caluclate_regret_after(game=game, PARTs_detail = '_PT4', opponent=opponent, option = options[0], num_episodes_trained=1000, r_toplot=r_toplot, moral_max=moral_max[game],  label=label, extra=extra,  moral_min=moral_min[game])
        all_runs.loc['ICN']['Game, then \nDeontological'], all_runs_ci.loc['ICN']['Game, then \nDeontological'] = caluclate_regret_after(game=game, PARTs_detail = '_PT3after2', opponent=opponent, option = options[0], num_episodes_trained=500, r_toplot=r_toplot, moral_max=moral_max[game],  label=label, extra=extra,  moral_min=moral_min[game])
        all_runs.loc['ICN']['Game, then \nUtilitarian'] , all_runs_ci.loc['ICN']['Game, then \nUtilitarian'] = caluclate_regret_after(game=game, PARTs_detail = '_PT3after2', opponent=opponent, option = options[1], num_episodes_trained=500, r_toplot=r_toplot, moral_max=moral_max[game],  label=label, extra=extra,  moral_min=moral_min[game])
    except: pass 

    game='BOS'
    all_runs.loc['BOS']['Game payoffs'], all_runs_ci.loc['BOS']['Game payoffs'] = caluclate_regret_after(game=game, PARTs_detail = '_PT2', opponent=opponent, option = options[0], num_episodes_trained=1000, r_toplot=r_toplot, moral_max=moral_max[game], label=label, extra=extra,  moral_min=moral_min[game])
    all_runs.loc['BOS']['Deontological'], all_runs_ci.loc['BOS']['Deontological'] = caluclate_regret_after(game=game, PARTs_detail = '_PT3', opponent=opponent, option = options[0], num_episodes_trained=1000, r_toplot=r_toplot, moral_max=moral_max[game],  label=label, extra=extra,  moral_min=moral_min[game])
    try: 
        all_runs.loc['BOS']['Utilitarian'], all_runs_ci.loc['BOS']['Utilitarian'] = caluclate_regret_after(game=game, PARTs_detail = '_PT3', opponent=opponent, option = options[1], num_episodes_trained=1000, r_toplot=r_toplot, moral_max=moral_max[game],  label=label, extra=extra,  moral_min=moral_min[game])
        all_runs.loc['BOS']['Game + \nDeontological'], all_runs_ci.loc['BOS']['Game + \nDeontological'] = caluclate_regret_after(game=game, PARTs_detail = '_PT4', opponent=opponent, option = options[0], num_episodes_trained=1000, r_toplot=r_toplot, moral_max=moral_max[game],  label=label, extra=extra,  moral_min=moral_min[game])
        all_runs.loc['BOS']['Game, then \nDeontological'], all_runs_ci.loc['BOS']['Game, then \nDeontological'] = caluclate_regret_after(game=game, PARTs_detail = '_PT3after2', opponent=opponent, option = options[0], num_episodes_trained=500, r_toplot=r_toplot, moral_max=moral_max[game],  label=label, extra=extra,  moral_min=moral_min[game])
        all_runs.loc['BOS']['Game, then \nUtilitarian'] , all_runs_ci.loc['BOS']['Game, then \nUtilitarian'] = caluclate_regret_after(game=game, PARTs_detail = '_PT3after2', opponent=opponent, option = options[1], num_episodes_trained=500, r_toplot=r_toplot, moral_max=moral_max[game],  label=label, extra=extra,  moral_min=moral_min[game])
    except: pass 


    game='ICD'
    all_runs.loc['ICD']['Game payoffs'], all_runs_ci.loc['ICD']['Game payoffs'] = caluclate_regret_after(game=game, PARTs_detail = '_PT2', opponent=opponent, option = options[0], num_episodes_trained=1000, r_toplot=r_toplot, moral_max=moral_max[game], label=label, extra=extra,  moral_min=moral_min[game])
    all_runs.loc['ICD']['Deontological'], all_runs_ci.loc['ICD']['Deontological'] = caluclate_regret_after(game=game, PARTs_detail = '_PT3', opponent=opponent, option = options[0], num_episodes_trained=1000, r_toplot=r_toplot, moral_max=moral_max[game],  label=label, extra=extra,  moral_min=moral_min[game])
    try: 
        all_runs.loc['ICD']['Utilitarian'], all_runs_ci.loc['ICD']['Utilitarian'] = caluclate_regret_after(game=game, PARTs_detail = '_PT3', opponent=opponent, option = options[1], num_episodes_trained=1000, r_toplot=r_toplot, moral_max=moral_max[game],  label=label, extra=extra,  moral_min=moral_min[game])
        all_runs.loc['ICD']['Game + \nDeontological'], all_runs_ci.loc['ICD']['Game + \nDeontological'] = caluclate_regret_after(game=game, PARTs_detail = '_PT4', opponent=opponent, option = options[0], num_episodes_trained=1000, r_toplot=r_toplot, moral_max=moral_max[game],  label=label, extra=extra,  moral_min=moral_min[game])
        all_runs.loc['ICD']['Game, then \nDeontological'], all_runs_ci.loc['ICD']['Game, then \nDeontological'] = caluclate_regret_after(game=game, PARTs_detail = '_PT3after2', opponent=opponent, option = options[0], num_episodes_trained=500, r_toplot=r_toplot, moral_max=moral_max[game],  label=label, extra=extra,  moral_min=moral_min[game])
        all_runs.loc['ICD']['Game, then \nUtilitarian'] , all_runs_ci.loc['ICD']['Game, then \nUtilitarian'] = caluclate_regret_after(game=game, PARTs_detail = '_PT3after2', opponent=opponent, option = options[1], num_episodes_trained=500, r_toplot=r_toplot, moral_max=moral_max[game],  label=label, extra=extra,  moral_min=moral_min[game])
    except: pass 

    all_runs.index=["Iterated Prisoner's Dilemma", "Iterated Stag Hunt", "Iterated Chicken", "Iterated Bach-or-Stravinsky", "Iterated Defective Coordination"]
    all_runs_ci.index=["Iterated Prisoner's Dilemma", "Iterated Stag Hunt", "Iterated Chicken", "Iterated Bach-or-Stravinsky", "Iterated Defective Coordination"]

    if include_NoFT == False:
        all_runs = all_runs.drop(columns='No fine-tuning')
        all_runs_ci = all_runs_ci.drop(columns='No fine-tuning')
    #all_runs.T.plot(
    #    kind='bar', ylabel=f' {label}\n Moral Regret', 
    #    title = f'Test time performance on three matrix games \n (after fine-tuning on the IPD vs {opponent} opponent)',
    #    #legend = False,
    #    xlabel = 'Fine-tuned models \n (by reward type used)',
    #    color=['turquoise', 'darkorchid', 'goldenrod'], figsize=(17,7), fontsize=40)

    # Create the figure and axis objects
    fig, ax = plt.subplots(figsize=(14, 10))

    # Get the number of groups and bars per group
    n_groups = all_runs.shape[1]
    n_bars = all_runs.shape[0]

    # Set the width of each bar and the gap between groups
    bar_width = 0.88 / n_bars  # Adjust this value to change bar width
    group_gap = 0.12  # Adjust this value to change gap between groups

    # Create x-coordinates for the bars
    index = np.arange(n_groups) * (1 + group_gap)

    # Plot the data
    for i in range(n_bars):
        ax.bar(index + i * bar_width, all_runs.iloc[i], bar_width,
            color=['turquoise', 'darkorchid', 'goldenrod', 'green', 'cornflowerblue'][i],
            label=all_runs.index[i])

    # Add error bars
    for i in range(n_bars):
        ax.errorbar(index + i * bar_width, all_runs.iloc[i], yerr=all_runs_ci.iloc[i], fmt=',', color='black', lw=3)

    # Set the font size for all elements
    fontsize = 35  # You can adjust this value to your preferred size

    # Set the title
    ax.set_title(f'Test time performance on five matrix games\n(after fine-tuning on the IPD vs {opponent} opponent)', fontsize=fontsize, y=1.05)

    # Set the x-axis label
    ax.set_xlabel('Fine-tuned models (by reward type used)', fontsize=fontsize)

    # Set the y-axis label
    ax.set_ylabel(f'{label} Moral Regret', fontsize=fontsize)
    ax.set_ylim(ymin=0)
    
    # Set x-ticks in the middle of the groups
    ax.set_xticks(index + bar_width * (n_bars - 1) / 2)
    ax.set_xticklabels(all_runs.columns, fontsize=fontsize, rotation=90)

    # Adjust tick label font sizes
    ax.tick_params(axis='both', which='major', labelsize=fontsize,)

    # Adjust legend font size
    ax.legend(fontsize=25)

    # Adjust layout to prevent clipping of labels
    plt.tight_layout()

    if include_Prompted == True:
        extratitle='with2baselines'
    else: 
        extratitle=''
    # Save the plot 
    plt.savefig(f'{SAVE_FIGURES_PATH}/{EVALS_dir}/regret_{label}_allgames_opp{opponent}_includeNoFT{include_NoFT}_{extratitle}.png', tight_layout=True)

    # Show the plot
    plt.show()


def mini_calculateregret_new_baselines(game, r_toplot, moral_max, moral_min, label, value):
    ylab = 'No fine-tuning, \nDeont. prompted'
    PARTs_detail = '_PT2'
    num_episodes_trained = 1000
    game = 'IPD'
    allruns_singlepart = pd.DataFrame(columns=[f'run{run_idx}' for run_idx in [1,2,3,5,6]], index=[f'{label} Moral Regret'])
    try: 
        os.chdir(f'gemma-2-2b-it_FT_{PARTs_detail}_opp{opponent}_{num_episodes_trained}ep_{options[0]}')
    except:
        os.chdir(f'../gemma-2-2b-it_FT_{PARTs_detail}_opp{opponent}_{num_episodes_trained}ep_{options[0]}')
    for run_idx in [1,2,3,5,6]:
        run_df = pd.read_csv(f'run{run_idx}{extra}/EVAL BEFORE VALUE {value} PROMPTED {PARTs_detail} - independent eval {game}.csv', index_col=0)
        run_episodes = run_df[['episode', f'{r_toplot} (before)']].groupby('episode').mean()
        run_episodes_regret = moral_max[game] - run_episodes.reset_index()[f'{r_toplot} (before)']
        if label == 'Utilitarian': run_episodes_regret = run_episodes_regret /  (moral_max[game] - moral_min[game])
        #store this run's value for Moral Regret (before)
        allruns_singlepart[f'run{run_idx}'][f'{label} Moral Regret'] = run_episodes_regret.mean()
    return allruns_singlepart.mean(axis=1)[f'{label} Moral Regret'], 1.96 * allruns_singlepart.std(axis=1)[f'{label} Moral Regret'] / np.sqrt(5)


def plot_regret_during_allgames(opponent, num_episodes, extra, options):
    all_runs = pd.DataFrame(index=range(num_episodes),
                            #index = [f'{r_toplot} (before)', f'{r_toplot} (after)', 'Moral Regret (before)', 'Moral Regret (after)'], 
                            columns = ['Game payoffs', 'Deontological', 'Utilitarian', 'Game + Deontological', 'Game, then Deontological', 'Game, then Utilitarian'])
    all_runs_ci = pd.DataFrame(index=range(num_episodes),
                                columns = ['Game payoffs', 'Deontological', 'Utilitarian', 'Game + Deontological', 'Game, then Deontological', 'Game, then Utilitarian'])

    PARTs_detail_1 = '_PT2'
    PARTs_detail_2 = 'PART2'
    ylab = 'Game payoffs'
    option = options[0]
    label = 'Game'
    moral_max = 4
    moral_min = -6
    allruns_singlepart = pd.DataFrame(columns=[f'run{run_idx}' for run_idx in [1,2,3,5,6]], index=range(num_episodes))
    try: 
        os.chdir(f'gemma-2-2b-it_FT_{PARTs_detail_1}_opp{opponent}_{option}')
    except: 
        os.chdir(f'../gemma-2-2b-it_FT_{PARTs_detail_1}_opp{opponent}_{option}')
    for run_idx in [1,2,3,5,6]:
        run_df = pd.read_csv(f'run{run_idx}{extra}/During FT {PARTs_detail_2} ({label} rewards).csv', index_col=0)[['episode', 'reward_M']]
        run_episodes = run_df.groupby('episode').mean()
        run_episodes_regret = moral_max - run_episodes.reset_index()['reward_M']
        run_episodes_regret_normalised = (run_episodes_regret - moral_min) / (moral_max - moral_min)
        allruns_singlepart[f'run{run_idx}'] = run_episodes_regret_normalised
    all_runs[ylab] = allruns_singlepart.mean(axis=1)
    all_runs_ci[ylab] = allruns_singlepart.std(axis=1) / np.sqrt(5) * 1.96

    PARTs_detail_1 = '_PT3'
    PARTs_detail_2 = 'PART3'
    ylab = 'Deontological'
    option = options[0]
    label = 'De'
    moral_max = 0
    moral_min = -6
    allruns_singlepart = pd.DataFrame(columns=[f'run{run_idx}' for run_idx in [1,2,3,5,6]], index=range(num_episodes))
    os.chdir(f'../gemma-2-2b-it_FT_{PARTs_detail_1}_opp{opponent}_{option}')
    for run_idx in [1,2,3,5,6]:
        run_df = pd.read_csv(f'run{run_idx}{extra}/During FT {PARTs_detail_2} ({label} rewards).csv', index_col=0)[['episode', 'reward_M']]
        run_episodes = run_df.groupby('episode').mean()
        run_episodes_regret = moral_max - run_episodes.reset_index()['reward_M']
        run_episodes_regret_normalised = (run_episodes_regret - moral_min) / (moral_max - moral_min)
        allruns_singlepart[f'run{run_idx}'] = run_episodes_regret_normalised
    all_runs[ylab] = allruns_singlepart.mean(axis=1)
    all_runs_ci[ylab] = allruns_singlepart.std(axis=1) / np.sqrt(5) * 1.96

    PARTs_detail_1 = '_PT3'
    PARTs_detail_2 = 'PART3'
    ylab = 'Utilitarian'
    option = options[1]
    label = 'Ut'
    moral_max = 6
    moral_min = -6
    allruns_singlepart = pd.DataFrame(columns=[f'run{run_idx}' for run_idx in [1,2,3,5,6]], index=range(num_episodes))
    os.chdir(f'../gemma-2-2b-it_FT_{PARTs_detail_1}_opp{opponent}_{option}')
    for run_idx in [1,2,3,5,6]:
        run_df = pd.read_csv(f'run{run_idx}{extra}/During FT {PARTs_detail_2} ({label} rewards).csv', index_col=0)[['episode', 'reward_M']]
        run_episodes = run_df.groupby('episode').mean()
        run_episodes_regret = moral_max - run_episodes.reset_index()['reward_M']
        run_episodes_regret_normalised = (run_episodes_regret - moral_min) / (moral_max - moral_min)
        allruns_singlepart[f'run{run_idx}'] = run_episodes_regret_normalised
    all_runs[ylab] = allruns_singlepart.mean(axis=1)
    all_runs_ci[ylab] = allruns_singlepart.std(axis=1) / np.sqrt(5) * 1.96

    PARTs_detail_1 = '_PT4'
    PARTs_detail_2 = 'PART4'
    ylab = 'Game + Deontological'
    option = options[0]
    label = 'De&Game'
    moral_max = 4
    moral_min = -6
    allruns_singlepart = pd.DataFrame(columns=[f'run{run_idx}' for run_idx in [1,2,3,5,6]], index=range(num_episodes))
    os.chdir(f'../gemma-2-2b-it_FT_{PARTs_detail_1}_opp{opponent}_{option}')
    for run_idx in [1,2,3,5,6]:
        run_df = pd.read_csv(f'run{run_idx}{extra}/During FT {PARTs_detail_2} ({label} rewards).csv', index_col=0)[['episode', 'reward_M']]
        run_episodes = run_df.groupby('episode').mean()
        run_episodes_regret = moral_max - run_episodes.reset_index()['reward_M']
        run_episodes_regret_normalised = (run_episodes_regret - moral_min) / (moral_max - moral_min)
        allruns_singlepart[f'run{run_idx}'] = run_episodes_regret_normalised
    all_runs[ylab] = allruns_singlepart.mean(axis=1)
    all_runs_ci[ylab] = allruns_singlepart.std(axis=1) / np.sqrt(5) * 1.96

    PARTs_detail_1 = '_PT2_PT3'
    PARTs_detail_2_1, PARTs_detail_2_2 = 'PART2', 'PART3'
    ylab = 'Game, then Deontological'
    option = options[0]
    label1, label2 = 'Game', 'De'
    moral_max1, moral_min1 = 4, -6
    moral_max2, moral_min2 = 0, -6    
    allruns_singlepart = pd.DataFrame(columns=[f'run{run_idx}' for run_idx in [1,2,3,5,6]], index=range(num_episodes))
    os.chdir(f'../gemma-2-2b-it_FT_{PARTs_detail_1}_opp{opponent}_{option}')
    for run_idx in [1,2,3,5,6]:
        run_df_1 = pd.read_csv(f'run{run_idx}{extra}/During FT {PARTs_detail_2_1} ({label1} rewards).csv', index_col=0)[['episode', 'reward_M']]
        run_df_1['regret_M'] = moral_max1 - run_df_1['reward_M']
        run_df_1['regret_normalised'] = (run_df_1['regret_M'] - moral_min1) / (moral_max1 - moral_min1)
        run_df_2 = pd.read_csv(f'run{run_idx}{extra}/During FT {PARTs_detail_2_2} ({label2} rewards).csv', index_col=0)[['episode', 'reward_M']]
        run_df_2['regret_M'] = moral_max2 - run_df_2['reward_M']
        run_df_2['regret_normalised'] = (run_df_2['regret_M'] - moral_min2) / (moral_max2 - moral_min2)
        run_df_2['episode'] = run_df_2['episode'] + num_episodes/2
        run_df = pd.concat([run_df_1[['episode', 'regret_normalised']], run_df_2[['episode', 'regret_normalised']]], ignore_index=True)
        run_episodes = run_df.groupby('episode').mean()
        run_episodes_regret_normalised = run_episodes.reset_index()['regret_normalised']
        allruns_singlepart[f'run{run_idx}'] = run_episodes_regret_normalised
    all_runs[ylab] = allruns_singlepart.mean(axis=1)
    all_runs_ci[ylab] = allruns_singlepart.std(axis=1) / np.sqrt(5) * 1.96

    PARTs_detail_1 = '_PT2_PT3'
    PARTs_detail_2_1, PARTs_detail_2_2 = 'PART2', 'PART3'
    ylab = 'Game, then Utilitarian'
    option = options[1]
    label1, label2 = 'Game', 'Ut'
    moral_max1, moral_min1 = 4, -6
    moral_max2, moral_min2 = 6, -6
    allruns_singlepart = pd.DataFrame(columns=[f'run{run_idx}' for run_idx in [1,2,3,5,6]], index=range(num_episodes))
    os.chdir(f'../gemma-2-2b-it_FT_{PARTs_detail_1}_opp{opponent}_{option}')
    for run_idx in [1,2,3,5,6]:
        run_df_1 = pd.read_csv(f'run{run_idx}{extra}/During FT {PARTs_detail_2_1} ({label1} rewards).csv', index_col=0)[['episode', 'reward_M']]
        run_df_1['regret_M'] = moral_max1 - run_df_1['reward_M']
        run_df_1['regret_normalised'] = (run_df_1['regret_M'] - moral_min1) / (moral_max1 - moral_min1)
        run_df_2 = pd.read_csv(f'run{run_idx}{extra}/During FT {PARTs_detail_2_2} ({label2} rewards).csv', index_col=0)[['episode', 'reward_M']]
        run_df_2['regret_M'] = moral_max2 - run_df_2['reward_M']
        run_df_2['regret_normalised'] = (run_df_2['regret_M'] - moral_min2) / (moral_max2 - moral_min2)
        run_df_2['episode'] = run_df_2['episode'] + num_episodes/2
        run_df = pd.concat([run_df_1[['episode', 'regret_normalised']], run_df_2[['episode', 'regret_normalised']]], ignore_index=True)
        run_episodes = run_df.groupby('episode').mean()
        run_episodes_regret_normalised = run_episodes.reset_index()['regret_normalised']
        allruns_singlepart[f'run{run_idx}'] = run_episodes_regret_normalised
    all_runs[ylab] = allruns_singlepart.mean(axis=1)
    all_runs_ci[ylab] = allruns_singlepart.std(axis=1) / np.sqrt(5) * 1.96



    #all_runs.plot(
    #    ylabel=f' Moral Regret \n(min-max normalised)', 
    #    title = f'Moral regret \n during various types of fine-tuning \n (vs {opponent} opponent)',
    #    #legend = False,
    #    xlabel = 'Episode',
        #    color=['turquoise', 'darkorchid', 'goldenrod']
    #    figsize=(10,7), 
    #    linewidth=1.5
    #    )


    fig = plt.figure(figsize=(10,7))
    colors = ['darkgoldenrod', 'darkorchid', 'teal', 'brown', 'red', 'dodgerblue']
    i=-1
    for col in all_runs.columns:
        i+=1
        color = colors[i]
        plt.plot(all_runs.index, all_runs[col], label=col, linewidth=1.4, alpha=0.8, color=color)
        plt.fill_between(all_runs.index, all_runs[col]+all_runs_ci[col], all_runs[col]-all_runs_ci[col], alpha=0.15, color=color)
    leg = plt.legend(title = 'Fine-tuning type', loc='center right', bbox_to_anchor=(2.0, 0.5), fontsize=32)
    # set the linewidth of each legend object
    for legobj in leg.legendHandles:
        legobj.set_linewidth(10.0)
    plt.title(f'Moral regret \n during various types of fine-tuning \n (vs {opponent} opponent)')
    plt.ylabel(f' Moral Regret \n(min-max normalised)')
    plt.xlabel('Episode')

    # Save the plot 
    plt.savefig(f'{SAVE_FIGURES_PATH}/RESULTS/regret_DURING_opp{opponent}.pdf', tight_layout=True)

    # Show the plot
    plt.show()



def plot_reward_during_allgames(opponent, num_episodes, extra, options):
    all_runs = pd.DataFrame(index=range(num_episodes),
                            #index = [f'{r_toplot} (before)', f'{r_toplot} (after)', 'Moral Regret (before)', 'Moral Regret (after)'], 
                            columns = ['Game payoffs', 'Deontological', 'Utilitarian', 'Game + Deontological', 'Game, then Deontological', 'Game, then Utilitarian'])
    all_runs_ci = pd.DataFrame(index=range(num_episodes),
                                columns = ['Game payoffs', 'Deontological', 'Utilitarian', 'Game + Deontological', 'Game, then Deontological', 'Game, then Utilitarian'])

    PARTs_detail_1 = '_PT2'
    PARTs_detail_2 = 'PART2'
    ylab = 'Game payoffs'
    option = options[0]
    label = 'Game'
    moral_max = 4
    moral_min = -6
    allruns_singlepart = pd.DataFrame(columns=[f'run{run_idx}' for run_idx in [1,2,3,5,6]], index=range(num_episodes))
    try: 
        os.chdir(f'gemma-2-2b-it_FT_{PARTs_detail_1}_opp{opponent}_{option}')
    except: 
        os.chdir(f'../gemma-2-2b-it_FT_{PARTs_detail_1}_opp{opponent}_{option}')
    for run_idx in [1,2,3,5,6]:
        run_df = pd.read_csv(f'run{run_idx}{extra}/During FT {PARTs_detail_2} ({label} rewards).csv', index_col=0)[['episode', 'reward_M']]
        run_episodes = run_df.groupby('episode').mean()
        run_episodes_reward = run_episodes.reset_index()['reward_M']
        run_episodes_reward_normalised = (run_episodes_reward - moral_min) / (moral_max - moral_min)
        allruns_singlepart[f'run{run_idx}'] = run_episodes_reward_normalised
    all_runs[ylab] = allruns_singlepart.mean(axis=1)
    all_runs_ci[ylab] = allruns_singlepart.std(axis=1) / np.sqrt(5) * 1.96

    PARTs_detail_1 = '_PT3'
    PARTs_detail_2 = 'PART3'
    ylab = 'Deontological'
    option = options[0]
    label = 'De'
    moral_max = 0
    moral_min = -6
    allruns_singlepart = pd.DataFrame(columns=[f'run{run_idx}' for run_idx in [1,2,3,5,6]], index=range(num_episodes))
    os.chdir(f'../gemma-2-2b-it_FT_{PARTs_detail_1}_opp{opponent}_{option}')
    for run_idx in [1,2,3,5,6]:
        run_df = pd.read_csv(f'run{run_idx}{extra}/During FT {PARTs_detail_2} ({label} rewards).csv', index_col=0)[['episode', 'reward_M']]
        run_episodes = run_df.groupby('episode').mean()
        run_episodes_reward = run_episodes.reset_index()['reward_M']
        run_episodes_reward_normalised = (run_episodes_reward - moral_min) / (moral_max - moral_min)
        allruns_singlepart[f'run{run_idx}'] = run_episodes_reward_normalised
    all_runs[ylab] = allruns_singlepart.mean(axis=1)
    all_runs_ci[ylab] = allruns_singlepart.std(axis=1) / np.sqrt(5) * 1.96

    PARTs_detail_1 = '_PT3'
    PARTs_detail_2 = 'PART3'
    ylab = 'Utilitarian'
    option = options[1]
    label = 'Ut'
    moral_max = 6
    moral_min = -6
    allruns_singlepart = pd.DataFrame(columns=[f'run{run_idx}' for run_idx in [1,2,3,5,6]], index=range(num_episodes))
    os.chdir(f'../gemma-2-2b-it_FT_{PARTs_detail_1}_opp{opponent}_{option}')
    for run_idx in [1,2,3,5,6]:
        run_df = pd.read_csv(f'run{run_idx}{extra}/During FT {PARTs_detail_2} ({label} rewards).csv', index_col=0)[['episode', 'reward_M']]
        run_episodes = run_df.groupby('episode').mean()
        run_episodes_reward = run_episodes.reset_index()['reward_M']
        run_episodes_reward_normalised = (run_episodes_reward - moral_min) / (moral_max - moral_min)
        allruns_singlepart[f'run{run_idx}'] = run_episodes_reward_normalised
    all_runs[ylab] = allruns_singlepart.mean(axis=1)
    all_runs_ci[ylab] = allruns_singlepart.std(axis=1) / np.sqrt(5) * 1.96

    PARTs_detail_1 = '_PT4'
    PARTs_detail_2 = 'PART4'
    ylab = 'Game + Deontological'
    option = options[0]
    label = 'De&Game'
    moral_max = 4
    moral_min = -6
    allruns_singlepart = pd.DataFrame(columns=[f'run{run_idx}' for run_idx in [1,2,3,5,6]], index=range(num_episodes))
    os.chdir(f'../gemma-2-2b-it_FT_{PARTs_detail_1}_opp{opponent}_{option}')
    for run_idx in [1,2,3,5,6]:
        run_df = pd.read_csv(f'run{run_idx}{extra}/During FT {PARTs_detail_2} ({label} rewards).csv', index_col=0)[['episode', 'reward_M']]
        run_episodes = run_df.groupby('episode').mean()
        run_episodes_reward = run_episodes.reset_index()['reward_M']
        run_episodes_reward_normalised = (run_episodes_reward - moral_min) / (moral_max - moral_min)
        allruns_singlepart[f'run{run_idx}'] = run_episodes_reward_normalised
    all_runs[ylab] = allruns_singlepart.mean(axis=1)
    all_runs_ci[ylab] = allruns_singlepart.std(axis=1) / np.sqrt(5) * 1.96

    PARTs_detail_1 = '_PT2_PT3'
    PARTs_detail_2_1, PARTs_detail_2_2 = 'PART2', 'PART3'
    ylab = 'Game, then Deontological'
    option = options[0]
    label1, label2 = 'Game', 'De'
    moral_max1, moral_min1 = 4, -6
    moral_max2, moral_min2 = 0, -6
    allruns_singlepart = pd.DataFrame(columns=[f'run{run_idx}' for run_idx in [1,2,3,5,6]], index=range(num_episodes))
    os.chdir(f'../gemma-2-2b-it_FT_{PARTs_detail_1}_opp{opponent}_{option}')
    for run_idx in [1,2,3,5,6]:
        run_df_1 = pd.read_csv(f'run{run_idx}{extra}/During FT {PARTs_detail_2_1} ({label1} rewards).csv', index_col=0)[['episode', 'reward_M']]
        run_df_1['reward_normalised'] = (run_df_1['reward_M'] - moral_min1) / (moral_max1 - moral_min1)
        run_df_2 = pd.read_csv(f'run{run_idx}{extra}/During FT {PARTs_detail_2_2} ({label2} rewards).csv', index_col=0)[['episode', 'reward_M']]
        run_df_2['reward_normalised'] = (run_df_2['reward_M'] - moral_min2) / (moral_max2 - moral_min2)
        run_df_2['episode'] = run_df_2['episode'] + num_episodes/2
        run_df = pd.concat([run_df_1[['episode', 'reward_normalised']], run_df_2[['episode', 'reward_normalised']]], ignore_index=True)
        run_episodes = run_df.groupby('episode').mean()
        run_episodes_reward_normalised = run_episodes.reset_index()['reward_normalised']
        allruns_singlepart[f'run{run_idx}'] = run_episodes_reward_normalised
    all_runs[ylab] = allruns_singlepart.mean(axis=1)
    all_runs_ci[ylab] = allruns_singlepart.std(axis=1) / np.sqrt(5) * 1.96

    PARTs_detail_1 = '_PT2_PT3'
    PARTs_detail_2_1, PARTs_detail_2_2 = 'PART2', 'PART3'
    ylab = 'Game, then Utilitarian'
    option = options[1]
    label1, label2 = 'Game', 'Ut'
    moral_max1, moral_min1 = 4, -6
    moral_max2, moral_min2 = 6, -6
    allruns_singlepart = pd.DataFrame(columns=[f'run{run_idx}' for run_idx in [1,2,3,5,6]], index=range(num_episodes))
    os.chdir(f'../gemma-2-2b-it_FT_{PARTs_detail_1}_opp{opponent}_{option}')
    for run_idx in [1,2,3,5,6]:
        run_df_1 = pd.read_csv(f'run{run_idx}{extra}/During FT {PARTs_detail_2_1} ({label1} rewards).csv', index_col=0)[['episode', 'reward_M']]
        run_df_1['reward_normalised'] = (run_df_1['reward_M'] - moral_min1) / (moral_max1 - moral_min1)
        run_df_2 = pd.read_csv(f'run{run_idx}{extra}/During FT {PARTs_detail_2_2} ({label2} rewards).csv', index_col=0)[['episode', 'reward_M']]
        run_df_2['reward_normalised'] = (run_df_2['reward_M'] - moral_min2) / (moral_max2 - moral_min2)
        run_df_2['episode'] = run_df_2['episode'] + num_episodes/2
        run_df = pd.concat([run_df_1[['episode', 'reward_normalised']], run_df_2[['episode', 'reward_normalised']]], ignore_index=True)
        run_episodes = run_df.groupby('episode').mean()
        run_episodes_reward_normalised = run_episodes.reset_index()['reward_normalised']
        allruns_singlepart[f'run{run_idx}'] = run_episodes_reward_normalised
    all_runs[ylab] = allruns_singlepart.mean(axis=1)
    all_runs_ci[ylab] = allruns_singlepart.std(axis=1) / np.sqrt(5) * 1.96

    #convert all_runs to movingaverage
    all_runs = all_runs.rolling(window=10).mean()
    all_runs_ci = all_runs_ci.rolling(window=10).mean()


    fig = plt.figure(figsize=(10,7))
    colors = ['darkgoldenrod', 'darkorchid', 'green', 'darkorange', 'red', 'dodgerblue']
        #legend = False,
    i=-1
    for col in all_runs.columns:
        i=i+1
        color = colors[i]
        plt.plot(all_runs.index, all_runs[col], label=col, linewidth=1.6, alpha=1, color=color)
        plt.fill_between(all_runs.index, all_runs[col]+all_runs_ci[col], all_runs[col]-all_runs_ci[col], alpha=0.15, color=color)
    leg = plt.legend(title = 'Fine-tuning type', loc='center right', bbox_to_anchor=(2.0, 0.5), fontsize=32)
    # set the linewidth of each legend object
    for legobj in leg.legendHandles:
        legobj.set_linewidth(10.0)
    plt.title(f'Moral reward \n during various types of fine-tuning \n (vs {opponent} opponent)')
    plt.ylabel(f' Moral Reward \n(min-max normalised)')
    plt.xlabel('Episode')

    # Save the plot 
    plt.savefig(f'{SAVE_FIGURES_PATH}/RESULTS/reward_DURING_opp{opponent}.pdf', bbox_inches='tight', dpi=300) #tight_layout=True, dpi=300)

    # Show the plot
    #plt.show()


os.getcwd()
#os.chdir('{SAVE_FIGURES_PATH}/RESULTS/EVALaction21')


def calculate_eval_responses_unrelated(PARTs_detail, opponent, num_episodes_trained, option, extra, C_str, D_str, columns):
    index_actions = [f'{C_str}', f'{D_str}', 'other']
    allruns_singlepart = pd.DataFrame(columns=columns, index=index_actions)
    allruns_singlepart.index = allruns_singlepart.index.rename('index') 
    try: 
        os.chdir(f'gemma-2-2b-it_FT_{PARTs_detail}_opp{opponent}_{num_episodes_trained}ep_{option}')
    except: 
        os.chdir(f'../gemma-2-2b-it_FT_{PARTs_detail}_opp{opponent}_{num_episodes_trained}ep_{option}')
             
    for run_idx in [1,2,3,5,6]: #NOTE add RUN 6 @!!! 
        run_df = pd.read_csv(f'run{run_idx}{extra}/EVAL After FT {PARTs_detail} - independent eval 4 unrelated queries.csv', index_col=0)[columns] #[['episode'] + columns]
        for col in run_df[columns]:
            run_df[col] = run_df[col].str.strip().apply(lambda x: x if x in [C_str, D_str] else 'other')

        run_counts = pd.DataFrame(index=index_actions, columns=columns)
        #count every value in the cells of run_df, separately for each column
        for col in run_df[columns]:
            run_counts[col] = run_df[col].value_counts().to_frame()
        run_counts.index = run_counts.index.rename('index')
        #convert numbers to percentages
        run_percent = run_counts * 100 / run_counts.sum(axis=0)
        concat_df = pd.concat([allruns_singlepart.reset_index(), run_percent.reset_index()])
        allruns_singlepart = concat_df.groupby(by='index').agg({col:'mean' for col in columns})
        #scale once again to fit on the 1-100 scale
        allruns_singlepart = allruns_singlepart * 100 / allruns_singlepart.sum(axis=0)
        return allruns_singlepart[['response (after) - game', 'response (after) - question', 'response (after) - moral']].rename(
            columns={'response (after) - question': 'Action only', 'response (after) - game': 'Action+Game', 'response (after) - moral': 'Action+Game+State'}).T




def visualise_unrelated_eval(opponent, extra, options, C_str, D_str):
    index_actions = [f'{C_str}', f'{D_str}', 'other']

    iterables = [['No fine-tuning', 'Game payoffs', 'Deontological', 'Utilitarian', 'Game + \nDeontological', 'Game, then \nDeontological', 'Game, then \nUtilitarian'],
                  index_actions]
    multiindex = pd.MultiIndex.from_product(iterables, names=["PART", "prompt_type"])
    #    Action-only, Action+game, Action+game+state, explicit IPD
    all_runs = pd.DataFrame(index=['Action only', 'Action+Game', 'Action+Game+State'],
                 columns=multiindex)
    #all_runs = pd.DataFrame(index=['Game - no payoffs', 'Question - no payoffs', 'Question - reciprocity', 'Game - explicit IPD'],
    #                        #index = [f'{r_toplot} (before)', f'{r_toplot} (after)', 'Moral Regret (before)', 'Moral Regret (after)'], 
    #                        columns = ['No fine-tuning', 'Game payoffs', 'Deontological', 'Utilitarian', 'Game + \nDeontological', 'Game, then \nDeontological', 'Game, then \nUtilitarian'])
    
    columns = ['response (before) - question', 'response (after) - question','response (before) - game', 'response (after) - game', 'response (before) - moral', 'response (after) - moral']
    num_episodes_trained = 1000

    ylab = 'Game payoffs' #we will do the \No fine-tuning part in parallel with this one
    PARTs_detail = '_PT2'
    allruns_singlepart = pd.DataFrame(columns=columns, index=index_actions)
    allruns_singlepart.index = allruns_singlepart.index.rename('index') 
    try: 
        os.chdir(f'gemma-2-2b-it_FT_{PARTs_detail}_opp{opponent}_{num_episodes_trained}ep_{options[0]}')
    except:
        os.chdir(f'../gemma-2-2b-it_FT_{PARTs_detail}_opp{opponent}_{num_episodes_trained}ep_{options[0]}')
    for run_idx in [1,2,3,5,6]: 
    #for run_idx in [1,2,3,5,]: 
        run_df = pd.read_csv(f'run{run_idx}{extra}/EVAL After FT {PARTs_detail} - independent eval 4 unrelated queries.csv', index_col=0)[columns] #[['episode'] + columns]
        for col in run_df[columns]:
            run_df[col] = run_df[col].str.strip().apply(lambda x: x if x in [C_str, D_str] else 'other')

        run_counts = pd.DataFrame(index=index_actions, columns=columns)
        #count every value in the cells of run_df, separately for each column
        for col in run_df[columns]:
            run_counts[col] = run_df[col].value_counts().to_frame()
        run_counts.index = run_counts.index.rename('index')
        #convert numbers to percentages
        run_percent = run_counts * 100 / run_counts.sum(axis=0)
        concat_df = pd.concat([allruns_singlepart.reset_index(), run_percent.reset_index()])
        allruns_singlepart = concat_df.groupby(by='index').agg({col:'mean' for col in columns})
        #scale once again to fit on the 1-100 scale
        allruns_singlepart = allruns_singlepart * 100 / allruns_singlepart.sum(axis=0)

        all_runs['No fine-tuning'] = allruns_singlepart[['response (before) - question', 'response (before) - game', 'response (before) - moral']].rename(
            columns={'response (before) - question': 'Action only', 'response (before) - game': 'Action+Game', 'response (before) - moral': 'Action+Game+State'}).T
        try: 
            all_runs[ylab] = allruns_singlepart[['response (after) - game', 'response (after) - question', 'response (after) - moral']].rename(
            columns={'response (after) - question': 'Action only', 'response (after) - game': 'Action+Game', 'response (after) - moral': 'Action+Game+State'}).T
        except: 
            pass

    ylab = 'Deontological'
    PARTs_detail = '_PT3'
    option = options[0]
    all_runs[ylab] = calculate_eval_responses_unrelated(PARTs_detail, opponent, num_episodes_trained, option, extra, C_str, D_str, columns)
    
    ylab = 'Utilitarian'
    PARTs_detail = '_PT3'
    option = options[1]
    all_runs[ylab] = calculate_eval_responses_unrelated(PARTs_detail, opponent, num_episodes_trained, option, extra, C_str, D_str, columns)
    
    ylab = 'Game + \nDeontological'
    PARTs_detail = '_PT4'
    option = options[0]
    try:  
        all_runs[ylab] = calculate_eval_responses_unrelated(PARTs_detail, opponent, num_episodes_trained, option, extra, C_str, D_str, columns)
    except: 
        pass
    
    ylab = 'Game, then \nDeontological'
    PARTs_detail = '_PT3after2' #'_PT2_PT3'
    option = options[0]
    all_runs[ylab] = calculate_eval_responses_unrelated(PARTs_detail, opponent, 500, option, extra, C_str, D_str, columns)

    ylab = 'Game, then \nUtilitarian'
    PARTs_detail = '_PT3after2' #'_PT2_PT3'
    option = options[1]
    all_runs[ylab] = calculate_eval_responses_unrelated(PARTs_detail, opponent, 500, option, extra, C_str, D_str, columns)

    fig, axs = plt.subplots(1, 7, figsize=(55, 7), sharey=True)
    fontsize=70
    colors = ['blue', 'red', 'grey']
    all_runs['No fine-tuning'].plot(kind='bar', stacked=True, ax=axs[0], legend=False, color=colors, fontsize=fontsize)
    axs[0].set_title('No fine-tuning\n', fontsize=fontsize)
    all_runs['Game payoffs'].plot(kind='bar', stacked=True, ax=axs[1], legend=False, color=colors, fontsize=fontsize)
    axs[1].set_title('Game \npayoffs', fontsize=fontsize)
    all_runs['Deontological'].plot(kind='bar', stacked=True, ax=axs[2], legend=False, color=colors, fontsize=fontsize)
    axs[2].set_title('Deontological\n', fontsize=fontsize)
    all_runs['Utilitarian'].plot(kind='bar', stacked=True, ax=axs[3], legend=False, color=colors, fontsize=fontsize)
    axs[3].set_title('Utilitarian\n', fontsize=fontsize)
    all_runs['Game + \nDeontological'].plot(kind='bar', stacked=True, ax=axs[4], legend=False, color=colors, fontsize=fontsize)
    axs[4].set_title('Game + \nDeontological', fontsize=fontsize)
    all_runs['Game, then \nDeontological'].plot(kind='bar', stacked=True, ax=axs[5], legend=False, color=colors, fontsize=fontsize)
    axs[5].set_title('Game, then \nDeontological', fontsize=fontsize)
    all_runs['Game, then \nUtilitarian'].plot(kind='bar', stacked=True, ax=axs[6], title='Game, then \nUtilitarian', legend=False, color=colors, fontsize=fontsize)
    axs[6].set_title('Game, then \nUtilitarian', fontsize=fontsize)
    #plt.legend(bbox_to_anchor=(1.9, 0.5), loc='center', fontsize=fontsize, labels=[f'{C_str} = C', f'{D_str} = D', 'other'])
    plt.legend(bbox_to_anchor=(1.9, 0.5), loc='center', fontsize=fontsize, labels=[f'{C_str}', f'{D_str}', 'other'])
    #plt.xlabel('Prompt type')
    axs[0].set_ylabel(f'Action choices \n (% of test time responses)', fontsize=fontsize)
    axs[3].set_xlabel('Test Prompt type', fontsize=fontsize)
    plt.suptitle(f'Action choices on unrelated prompts \n (all models trained vs {opponent} opponent)', fontsize=fontsize+10, y=1.49)



    if False: 
        # Set the font size for all elements
        fontsize = 35  # You can adjust this value to your preferred size

        # Set the title
        ax.set_title(f'Test time performance on three matrix games\n(after fine-tuning on the IPD vs {opponent} opponent)', fontsize=fontsize)

        # Set x-ticks in the middle of the groups
        ax.set_xticks(index + bar_width * (n_bars - 1) / 2)
        ax.set_xticklabels(all_runs.columns, fontsize=fontsize, rotation=90)

        # Adjust tick label font sizes
        ax.tick_params(axis='both', which='major', labelsize=fontsize,)

    # Adjust layout to prevent clipping of labels
    plt.tight_layout()

    # Save the plot 
    plt.savefig(f'{SAVE_FIGURES_PATH}/RESULTS/{EVALS_dir}/unrelated_opp{opponent}_noexplicitIPD.pdf', bbox_inches='tight', dpi=300)

    # Show the plot
    plt.show()


def calculate_eval_reciprocity(PARTs_detail, opponent, num_episodes_trained, option, extra, C_str, D_str, action_types):
    allruns_singlepart = pd.DataFrame(columns=['action | state'], index=action_types)
    allruns_singlepart.index = allruns_singlepart.index.rename('index') 
    try: 
        os.chdir(f'gemma-2-2b-it_FT_{PARTs_detail}_opp{opponent}_{num_episodes_trained}ep_{option}')
    except FileNotFoundError: 
        os.chdir(f'../gemma-2-2b-it_FT_{PARTs_detail}_opp{opponent}_{num_episodes_trained}ep_{option}')
        
    
    for run_idx in [1,2,3,5,6]: 
        #for run_idx in [1,2,3,5]: 
        columns = ['response (before) - moral', 'response (after) - moral', 'opp_prev_move (for moral eval only)']
        run_df = pd.read_csv(f'run{run_idx}{extra}/EVAL After FT {PARTs_detail} - independent eval 4 unrelated queries.csv', index_col=0)[columns] #[['episode'] + columns]
        for col in run_df[columns]:
            run_df[col] = run_df[col].str.strip().apply(lambda x: x if x in [C_str, D_str] else 'other')
            run_df[col].replace({C_str: 'C', D_str:'D'}, inplace=True)
        run_df['action | state'] = run_df['response (after) - moral'] + ' | ' + run_df['opp_prev_move (for moral eval only)']

        run_df['action (ref model) | state'] = run_df['response (before) - moral'] + ' | ' + run_df['opp_prev_move (for moral eval only)']

        run_counts = pd.DataFrame(index=action_types, columns=['action | state'])
        #count every value in the cells of run_df, separately for each column
        run_counts['action | state'] = run_df['action | state'].value_counts().to_frame()
        run_counts.index = run_counts.index.rename('index')
        #convert numbers to percentages
        run_percent = run_counts * 100 / run_counts.sum(axis=0)
        concat_df = pd.concat([allruns_singlepart.reset_index(), run_percent.reset_index()])
        allruns_singlepart = concat_df.groupby(by='index').agg({'action | state':'mean'})
        #scale once again to fit on the 1-100 scale
        allruns_singlepart = allruns_singlepart * 100 / allruns_singlepart.sum(axis=0)

        return allruns_singlepart['action | state']

def visualise_reciprocity_eval(opponent, extra, options, C_str, D_str, stacked=False):
    #iterables = [['No fine-tuning', 'Game payoffs', 'Deontological', 'Utilitarian', 'Game + \nDeontological', 'Game, then \nDeontological', 'Game, then \nUtilitarian'],
    #              ['C | C','C | D,' 'D | C', 'D | D', 'other | C', 'other | D' ]]
    #multiindex = pd.MultiIndex.from_product(iterables, names=["PART", "Action type"])
    #    Action-only, Action+game, Action+game+state, explicit IPD
    num_episodes_trained = 1000

    action_types = ['C | C','C | D', 'D | C', 'D | D', 'other | C', 'other | D' ]
    all_runs = pd.DataFrame(columns = ['No fine-tuning', 'Game payoffs', 'Deontological', 'Utilitarian', 'Game + \nDeontological', 'Game, then \nDeontological', 'Game, then \nUtilitarian'],
                            index = action_types)
    
    ylab = 'Game payoffs'
    PARTs_detail = '_PT2'
    allruns_singlepart = pd.DataFrame(columns=['action | state'], index=action_types)
    allruns_singlepart.index = allruns_singlepart.index.rename('index') 
    try: 
        os.chdir(f'gemma-2-2b-it_FT_{PARTs_detail}_opp{opponent}_{num_episodes_trained}ep_{options[0]}')
    except:
        os.chdir(f'../gemma-2-2b-it_FT_{PARTs_detail}_opp{opponent}_{num_episodes_trained}ep_{options[0]}')
    for run_idx in [1,2,3,5,6]: 
#    for run_idx in [1,2,3,5,]: 
        columns = ['response (before) - moral', 'response (after) - moral', 'opp_prev_move (for moral eval only)']
        run_df = pd.read_csv(f'run{run_idx}{extra}/EVAL After FT {PARTs_detail} - independent eval 4 unrelated queries.csv', index_col=0)[columns] #[['episode'] + columns]
        for col in run_df[columns]:
            run_df[col] = run_df[col].str.strip().apply(lambda x: x if x in [C_str, D_str] else 'other')
            run_df[col].replace({C_str: 'C', D_str:'D'}, inplace=True)
        run_df['action | state'] = run_df['response (before) - moral'] + ' | ' + run_df['opp_prev_move (for moral eval only)']
        run_counts = pd.DataFrame(index=action_types, columns=['action | state'])
        #count every value in the cells of run_df, separately for each column
        run_counts['action | state'] = run_df['action | state'].value_counts().to_frame()
        run_counts.index = run_counts.index.rename('index')
        #convert numbers to percentages
        run_percent = run_counts * 100 / run_counts.sum(axis=0)
        concat_df = pd.concat([allruns_singlepart.reset_index(), run_percent.reset_index()])
        allruns_singlepart = concat_df.groupby(by='index').agg({'action | state':'mean'})
        #scale once again to fit on the 1-100 scale
        allruns_singlepart = allruns_singlepart * 100 / allruns_singlepart.sum(axis=0)
    all_runs['No fine-tuning'] = allruns_singlepart['action | state']

    ylab = 'Game payoffs'
    PARTs_detail = '_PT2'
    option = options[0]
    all_runs[ylab] = calculate_eval_reciprocity(PARTs_detail, opponent, num_episodes_trained, option, extra, C_str, D_str, action_types)

    ylab = 'Deontological'
    PARTs_detail = '_PT3'
    option = options[0]
    all_runs[ylab] = calculate_eval_reciprocity(PARTs_detail, opponent, num_episodes_trained, option, extra, C_str, D_str, action_types)
    
    ylab = 'Utilitarian'
    PARTs_detail = '_PT3'
    option = options[1]
    all_runs[ylab] = calculate_eval_reciprocity(PARTs_detail, opponent, num_episodes_trained, option, extra, C_str, D_str, action_types)
    
    ylab = 'Game + \nDeontological'
    PARTs_detail = '_PT4'
    option = options[0]
    try: 
        all_runs[ylab] = calculate_eval_reciprocity(PARTs_detail, opponent, num_episodes_trained, option, extra, C_str, D_str, action_types)
    except: 
        pass
    
    ylab = 'Game, then \nDeontological'
    PARTs_detail = '_PT3after2'
    option = options[0]
    all_runs[ylab] = calculate_eval_reciprocity(PARTs_detail, opponent, 500, option, extra, C_str, D_str, action_types)

    ylab = 'Game, then \nUtilitarian'
    PARTs_detail = '_PT3after2'
    option = options[1]
    all_runs[ylab] = calculate_eval_reciprocity(PARTs_detail, opponent, 500, option, extra, C_str, D_str, action_types)


    fontsize=50
    colors={'C | C':'#28641E', 'C | D':'#B0DC82', 'D | C':'#FBE6F1', 'D | D':'#8E0B52', 
                                 'other | C':'#A9A9A9', 'other | D':'#A9A9A9'}
    if stacked == True: 
        figsize = (15, 7)
    else:
        figsize = (20, 7)
    all_runs.T.plot(kind='bar', stacked=stacked, figsize=figsize, fontsize=fontsize, color=colors)
    plt.plot()
    #plot horizontal dahsed line with a label on top of this plot 
    if stacked == False:
        plt.axhline(y=25, color='black', linestyle='--', linewidth=1.5, label='Chance level \nof reciprocity')
    #add text in the center-right of the plot 
    #plt.text(0.01, 26, 'Chance level \nof reciprocity', fontsize=fontsize, color='black')
    plt.legend(bbox_to_anchor=(1.3, 0.5), loc='center', fontsize=fontsize, title='Action type')
    plt.ylabel(f" M's action | O's prev. move \n (% of test time responses)", fontsize=fontsize)
    plt.xlabel('Fine-tuned models (by R type used)', fontsize=fontsize)
    plt.title(f'Action choices on an unrelated prompt \ncontaining Action+Game+State \n (all models trained vs {opponent} opponent)', fontsize=fontsize, y=1.08)

    # Adjust layout to prevent clipping of labels
    plt.tight_layout()

    # Save the plot 
    plt.savefig(f'/{SAVE_FIGURES_PATH}/RESULTS/{EVALS_dir}/reciprocity_opp{opponent}_stacked{stacked}.pdf', bbox_inches='tight', dpi=300)

    # Show the plot
    plt.show()



def calculate_eval_reciprocity_IPDonly(PARTs_detail, opponent, num_episodes_trained, option, extra, C_str, D_str, action_types):
    allruns_singlepart = pd.DataFrame(columns=['action | state'], index=action_types)
    allruns_singlepart.index = allruns_singlepart.index.rename('index') 
    try: 
        os.chdir(f'gemma-2-2b-it_FT_{PARTs_detail}_opp{opponent}_{num_episodes_trained}ep_{option}')
    except FileNotFoundError: 
        os.chdir(f'../gemma-2-2b-it_FT_{PARTs_detail}_opp{opponent}_{num_episodes_trained}ep_{option}')
        
    
    for run_idx in [1,2,3,5,6]: 
        #for run_idx in [1,2,3,5]: 
        columns = ['query', 'response (before)', 'response (after)']
        run_df = pd.read_csv(f'run{run_idx}{extra}/EVAL After FT {PARTs_detail} - independent eval IPD.csv', index_col=0)[columns] #[['episode'] + columns]
        run_df['opp_prev_move'] = run_df['query'].apply(extract_prev_action_from_query)
        run_df['opp_prev_move'] = run_df['opp_prev_move'].str.replace(C_str, 'C').replace(D_str, 'D')
        for col in run_df[columns]:
            run_df[col] = run_df[col].str.strip().apply(lambda x: x if x in [C_str, D_str] else 'other')
            run_df[col].replace({C_str: 'C', D_str:'D'}, inplace=True)
        run_df['action | state'] = run_df['response (after)'] + ' | ' + run_df['opp_prev_move']

        run_df['action (ref model) | state'] = run_df['response (before)'] + ' | ' + run_df['opp_prev_move']

        run_counts = pd.DataFrame(index=action_types, columns=['action | state'])
        #count every value in the cells of run_df, separately for each column
        run_counts['action | state'] = run_df['action | state'].value_counts().to_frame()
        run_counts.index = run_counts.index.rename('index')
        #convert numbers to percentages
        run_percent = run_counts * 100 / run_counts.sum(axis=0)
        concat_df = pd.concat([allruns_singlepart.reset_index(), run_percent.reset_index()])
        allruns_singlepart = concat_df.groupby(by='index').agg({'action | state':'mean'})
        #scale once again to fit on the 1-100 scale
        allruns_singlepart = allruns_singlepart * 100 / allruns_singlepart.sum(axis=0)

        return allruns_singlepart['action | state']


def visualise_reciprocity_eval_IPDonly(opponent, extra, options, C_str, D_str, stacked=False):
    #iterables = [['No fine-tuning', 'Game payoffs', 'Deontological', 'Utilitarian', 'Game + \nDeontological', 'Game, then \nDeontological', 'Game, then \nUtilitarian'],
    #              ['C | C','C | D,' 'D | C', 'D | D', 'other | C', 'other | D' ]]
    #multiindex = pd.MultiIndex.from_product(iterables, names=["PART", "Action type"])
    #    Action-only, Action+game, Action+game+state, explicit IPD
    num_episodes_trained = 1000

    action_types = ['C | C','C | D', 'D | C', 'D | D', 'other | C', 'other | D' ]
    all_runs = pd.DataFrame(columns = ['No fine-tuning', 'Game payoffs', 'Deontological', 'Utilitarian', 'Game + \nDeontological', 'Game, then \nDeontological', 'Game, then \nUtilitarian'],
                            index = action_types)
    
    ylab = 'Game payoffs'
    PARTs_detail = '_PT2'
    allruns_singlepart = pd.DataFrame(columns=['action | state'], index=action_types)
    allruns_singlepart.index = allruns_singlepart.index.rename('index') 
    try: 
        os.chdir(f'gemma-2-2b-it_FT_{PARTs_detail}_opp{opponent}_{num_episodes_trained}ep_{options[0]}')
    except:
        os.chdir(f'../gemma-2-2b-it_FT_{PARTs_detail}_opp{opponent}_{num_episodes_trained}ep_{options[0]}')
    for run_idx in [1,2,3,5,6]: 
#    for run_idx in [1,2,3,5,]: 
        columns = ['query', 'response (before)', 'response (after)']
        run_df = pd.read_csv(f'run{run_idx}{extra}/EVAL After FT {PARTs_detail} - independent eval IPD.csv', index_col=0)[columns] #[['episode'] + columns]
        run_df['opp_prev_move'] = run_df['query'].apply(extract_prev_action_from_query)
        run_df['opp_prev_move'] = run_df['opp_prev_move'].str.replace(C_str, 'C').replace(D_str, 'D')
        for col in run_df[columns]:
            run_df[col] = run_df[col].str.strip().apply(lambda x: x if x in [C_str, D_str] else 'other')
            run_df[col].replace({C_str: 'C', D_str:'D'}, inplace=True)
        run_df['action | state'] = run_df['response (before)'] + ' | ' + run_df['opp_prev_move']
        run_counts = pd.DataFrame(index=action_types, columns=['action | state'])
        #count every value in the cells of run_df, separately for each column
        run_counts['action | state'] = run_df['action | state'].value_counts().to_frame()
        run_counts.index = run_counts.index.rename('index')
        #convert numbers to percentages
        run_percent = run_counts * 100 / run_counts.sum(axis=0)
        concat_df = pd.concat([allruns_singlepart.reset_index(), run_percent.reset_index()])
        allruns_singlepart = concat_df.groupby(by='index').agg({'action | state':'mean'})
        #scale once again to fit on the 1-100 scale
        allruns_singlepart = allruns_singlepart * 100 / allruns_singlepart.sum(axis=0)
    all_runs['No fine-tuning'] = allruns_singlepart['action | state']

    ylab = 'Game payoffs'
    PARTs_detail = '_PT2'
    option = options[0]
    all_runs[ylab] = calculate_eval_reciprocity_IPDonly(PARTs_detail, opponent, num_episodes_trained, option, extra, C_str, D_str, action_types)

    ylab = 'Deontological'
    PARTs_detail = '_PT3'
    option = options[0]
    all_runs[ylab] = calculate_eval_reciprocity_IPDonly(PARTs_detail, opponent, num_episodes_trained, option, extra, C_str, D_str, action_types)
    
    ylab = 'Utilitarian'
    PARTs_detail = '_PT3'
    option = options[1]
    all_runs[ylab] = calculate_eval_reciprocity_IPDonly(PARTs_detail, opponent, num_episodes_trained, option, extra, C_str, D_str, action_types)
    
    ylab = 'Game + \nDeontological'
    PARTs_detail = '_PT4'
    option = options[0]
    try: 
        all_runs[ylab] = calculate_eval_reciprocity_IPDonly(PARTs_detail, opponent, num_episodes_trained, option, extra, C_str, D_str, action_types)
    except: 
        pass
    
    ylab = 'Game, then \nDeontological'
    PARTs_detail = '_PT3after2'
    option = options[0]
    all_runs[ylab] = calculate_eval_reciprocity_IPDonly(PARTs_detail, opponent, 500, option, extra, C_str, D_str, action_types)

    ylab = 'Game, then \nUtilitarian'
    PARTs_detail = '_PT3after2'
    option = options[1]
    all_runs[ylab] = calculate_eval_reciprocity_IPDonly(PARTs_detail, opponent, 500, option, extra, C_str, D_str, action_types)


    fontsize=50
    colors={'C | C':'#28641E', 'C | D':'#B0DC82', 'D | C':'#FBE6F1', 'D | D':'#8E0B52', 
                                 'other | C':'#A9A9A9', 'other | D':'#A9A9A9'}
    if stacked == True: 
        figsize = (15, 7)
    else:
        figsize = (20, 7)
    all_runs.T.plot(kind='bar', stacked=stacked, figsize=figsize, fontsize=fontsize, color=colors)
    plt.plot()
    #plot horizontal dahsed line with a label on top of this plot 
    if stacked == False:
        plt.axhline(y=25, color='black', linestyle='--', linewidth=1.5, label='Chance level \nof reciprocity')
    #add text in the center-right of the plot 
    #plt.text(0.01, 26, 'Chance level \nof reciprocity', fontsize=fontsize, color='black')
    plt.legend(bbox_to_anchor=(1.3, 0.5), loc='center', fontsize=fontsize, title='Action type')
    plt.ylabel(f" M's action | O's prev. move \n (% of test time responses)", fontsize=fontsize)
    plt.xlabel('Fine-tuned models (by R type used)', fontsize=fontsize)
    plt.title(f'Action choices on the IPD (at test time) \n (all models trained vs {opponent} opponent) \n', fontsize=fontsize, y=1.08)

    # Adjust layout to prevent clipping of labels
    plt.tight_layout()

    # Save the plot 
    plt.savefig(f'{SAVE_FIGURES_PATH}/RESULTS/{EVALS_dir}/reciprocity_IPD_opp{opponent}_stacked{stacked}.pdf', bbox_inches='tight', dpi=300)

    # Show the plot
    plt.show()




def extract_prev_action_from_query(query):
    '''per value'''
    return query.split('they played ')[1].split(', so you got')[0]

def recode_actions(col, C_str, D_str):
    '''per entire column'''
    col = col.str.strip().apply(lambda x: x if x in [C_str, D_str] else 'illegal')
    col.replace({C_str: 'C', D_str:'D'}, inplace=True)
    return col

def process_actions_eval(PARTs_detail, opponent, num_episodes_trained, option, game, extra, C_str, D_str, num_episodes_eval, before_or_after, options=''):
    options = f' - {options}' if options is not '' else ''
    allruns_singlepart = pd.DataFrame(columns=[f'run{run_idx}' for run_idx in [1,2,3,5,6]], index=[['C | C', 'C | D', 'D | C', 'D | D', 'illegal | C', 'illegal | D']])
    try: 
        os.chdir(f'gemma-2-2b-it_FT_{PARTs_detail}_opp{opponent}_{num_episodes_trained}ep_{option}')
    except:
        os.chdir(f'../gemma-2-2b-it_FT_{PARTs_detail}_opp{opponent}_{num_episodes_trained}ep_{option}')
    for run_idx in [1,2,3,5,6]:
        run_df = pd.read_csv(f'run{run_idx}{extra}/EVAL After FT {PARTs_detail} - independent eval {game}.csv', index_col=0)
        run_df[f'response ({before_or_after}){options}'] = recode_actions(run_df[f'response ({before_or_after}){options}'], C_str, D_str)
        try: 
            run_df['opp_prev_move'] = run_df['query'].apply(extract_prev_action_from_query)
        except: 
            pass 
        run_df['opp_prev_move'] = run_df['opp_prev_move'].str.replace(C_str, 'C').replace(D_str, 'D')
        run_df['action | state'] = run_df[f'response ({before_or_after}){options}'] + ' | ' + run_df['opp_prev_move']
        #aggregate per episode 
        results_1run = pd.DataFrame(index=range(num_episodes_eval), columns=['C | C', 'C | D', 'D | C', 'D | D', 'illegal | C', 'illegal | D'])
        for episode in range(num_episodes_eval):
            episode_result = run_df[run_df['episode']==episode]['action | state'].value_counts()
            for combination in episode_result.index:
                results_1run.loc[episode, combination] = episode_result[combination]
        #insert into overall results df
        allruns_singlepart[f'run{run_idx}'] = results_1run.sum(axis=0).values
    return allruns_singlepart

#NEW#

def process_actions_eval_valuepromptedbaseline(PARTs_detail, opponent, num_episodes_trained, option, game, extra, C_str, D_str, num_episodes_eval, before_or_after, value, options=''):
    options = f' - {options}' if options is not '' else ''
    allruns_singlepart = pd.DataFrame(columns=[f'run{run_idx}' for run_idx in [1,2,3,5,6]], index=[['C | C', 'C | D', 'D | C', 'D | D', 'illegal | C', 'illegal | D']])
    try: 
        os.chdir(f'gemma-2-2b-it_FT_{PARTs_detail}_opp{opponent}_{num_episodes_trained}ep_{option}')
    except:
        os.chdir(f'../gemma-2-2b-it_FT_{PARTs_detail}_opp{opponent}_{num_episodes_trained}ep_{option}')
    for run_idx in [1,2,3,5,6]:
        run_df = pd.read_csv(f'run{run_idx}{extra}/EVAL Before VALUE {value} PROMPTED {PARTs_detail} - independent eval {game}.csv', index_col=0)
        run_df[f'response ({before_or_after}){options}'] = recode_actions(run_df[f'response ({before_or_after}){options}'], C_str, D_str)
        try: 
            run_df['opp_prev_move'] = run_df['query'].apply(extract_prev_action_from_query)
        except: 
            pass 
        run_df['opp_prev_move'] = run_df['opp_prev_move'].str.replace(C_str, 'C').replace(D_str, 'D')
        run_df['action | state'] = run_df[f'response ({before_or_after}){options}'] + ' | ' + run_df['opp_prev_move']
        #aggregate per episode 
        results_1run = pd.DataFrame(index=range(num_episodes_eval), columns=['C | C', 'C | D', 'D | C', 'D | D', 'illegal | C', 'illegal | D'])
        for episode in range(num_episodes_eval):
            episode_result = run_df[run_df['episode']==episode]['action | state'].value_counts()
            for combination in episode_result.index:
                results_1run.loc[episode, combination] = episode_result[combination]
        #insert into overall results df
        allruns_singlepart[f'run{run_idx}'] = results_1run.sum(axis=0).values
    return allruns_singlepart

def visualise_othergames_eval(opponent, extra, options, C_str, D_str, include_Prompted=False):

    iterables = [['No fine-tuning', 'Game payoffs', 'Deontological', 'Utilitarian', 'Game + \nDeontological', 'Game, then \nDeontological', 'Game, then \nUtilitarian'],
                  ['C | C', 'C | D', 'D | C', 'D | D', 'illegal | C', 'illegal | D']]
    if include_Prompted:
        iterables[0].insert(1, 'No fine-tuning, \nDeont. prompted')
        iterables[0].insert(1, 'No fine-tuning, \nUtilit. prompted')
    multiindex = pd.MultiIndex.from_product(iterables, names=["PART", "Action Types"])
    games = ['IPD', 'ISH', 'ICN', 'BOS', 'ICD']
    all_runs = pd.DataFrame(index=games,
                            columns = multiindex)
    
    num_episodes_eval = 10 

    ylab = 'No fine-tuning'
    for game in games:
        allruns_singlepart = process_actions_eval('_PT2', opponent, 1000, options[0], game, extra, C_str, D_str, num_episodes_eval, before_or_after='before')
        all_runs.loc[game][ylab] = allruns_singlepart.mean(axis=1).values

    if include_Prompted: 
        ylab = 'No fine-tuning, \nDeont. prompted'
        for game in games:
            allruns_singlepart = process_actions_eval_valuepromptedbaseline('_PT2', opponent, 1000, options[0], game, extra, C_str, D_str, num_episodes_eval, value='De', before_or_after='before')
            all_runs.loc[game][ylab] = allruns_singlepart.mean(axis=1).values
   
        ylab = 'No fine-tuning, \nUtilit. prompted'
        for game in games:
            allruns_singlepart = process_actions_eval_valuepromptedbaseline('_PT2', opponent, 1000, options[0], game, extra, C_str, D_str, num_episodes_eval, value='Ut', before_or_after='before')
            all_runs.loc[game][ylab] = allruns_singlepart.mean(axis=1).values

            
    for game in games:
        all_runs.loc[game]['Game payoffs'] = process_actions_eval('_PT2', opponent, 1000, options[0], game, extra, C_str, D_str, num_episodes_eval, before_or_after='after').mean(axis=1).values
        all_runs.loc[game]['Deontological'] = process_actions_eval('_PT3', opponent, 1000, options[0], game, extra, C_str, D_str, num_episodes_eval, before_or_after='after').mean(axis=1).values
        all_runs.loc[game]['Utilitarian'] = process_actions_eval('_PT3', opponent, 1000, options[1], game, extra, C_str, D_str, num_episodes_eval, before_or_after='after').mean(axis=1).values
        all_runs.loc[game]['Game + \nDeontological'] = process_actions_eval('_PT4', opponent, 1000, options[0], game, extra, C_str, D_str, num_episodes_eval, before_or_after='after').mean(axis=1).values
        all_runs.loc[game]['Game, then \nDeontological'] = process_actions_eval('_PT3after2', opponent, 500, options[0], game, extra, C_str, D_str, num_episodes_eval, before_or_after='after').mean(axis=1).values
        all_runs.loc[game]['Game, then \nUtilitarian'] = process_actions_eval('_PT3after2', opponent, 500, options[1], game, extra, C_str, D_str, num_episodes_eval, before_or_after='after').mean(axis=1).values

    all_runs.index=["Iterated Prisoner's Dilemma", "Iterated Stag Hunt", "Iterated Chicken", "Iterated Bach-or-Stravinsky", "Iterated Defective Coordination"]


    if include_Prompted:
        fig, axs = plt.subplots(1, 9, figsize=(85, 7), sharey=True)
    else: 
        fig, axs = plt.subplots(1, 7, figsize=(65, 7), sharey=True) 
    fontsize=70
    #colors = {'C|C':'#28641E', 'C|D':'#B0DC82', 'D|C':'#FBE6F1', 'D|D':'#8E0B52', 
    #          'illegal|C':'#A9A9A9', 'illegal|D':'#A9A9A9', 'C|illegal':'#A9A9A9', 'D|illegal':'#A9A9A9', 'illegal|illegal':'#A9A9A9'}
    colors = ['#28641E', '#B0DC82', '#FBE6F1', '#8E0B52', 
                '#A9A9A9', '#A9A9A9', '#A9A9A9', '#A9A9A9', '#A9A9A9']
    axs_counter = 0
    all_runs['No fine-tuning'].plot(kind='bar', stacked=True, ax=axs[0], legend=False, color=colors, fontsize=fontsize)
    axs[axs_counter].set_title('No fine-tuning\n', fontsize=fontsize)
    if include_Prompted:
        axs_counter += 1
        all_runs['No fine-tuning, \nDeont. prompted'].plot(kind='bar', stacked=True, ax=axs[axs_counter], legend=False, color=colors, fontsize=fontsize)
        axs[axs_counter].set_title('No fine-tuning, \nDeont. prompted', fontsize=fontsize)
        axs_counter += 1
        all_runs['No fine-tuning, \nUtilit. prompted'].plot(kind='bar', stacked=True, ax=axs[axs_counter], legend=False, color=colors, fontsize=fontsize)
        axs[axs_counter].set_title('No fine-tuning, \nUtilit. prompted', fontsize=fontsize)
    axs_counter += 1
    all_runs['Game payoffs'].plot(kind='bar', stacked=True, ax=axs[axs_counter], legend=False, color=colors, fontsize=fontsize)
    axs[axs_counter].set_title('Game \npayoffs', fontsize=fontsize)
    axs_counter += 1 
    all_runs['Deontological'].plot(kind='bar', stacked=True, ax=axs[axs_counter], legend=False, color=colors, fontsize=fontsize)
    axs[axs_counter].set_title('Deontological\n', fontsize=fontsize)
    axs_counter += 1 
    all_runs['Utilitarian'].plot(kind='bar', stacked=True, ax=axs[axs_counter], legend=False, color=colors, fontsize=fontsize)
    axs[axs_counter].set_title('Utilitarian\n', fontsize=fontsize)
    axs_counter += 1 
    all_runs['Game + \nDeontological'].plot(kind='bar', stacked=True, ax=axs[axs_counter], legend=False, color=colors, fontsize=fontsize)
    axs[axs_counter].set_title('Game + \nDeontological', fontsize=fontsize)
    axs_counter += 1
    all_runs['Game, then \nDeontological'].plot(kind='bar', stacked=True, ax=axs[axs_counter], legend=False, color=colors, fontsize=fontsize)
    axs[axs_counter].set_title('Game, then \nDeontological', fontsize=fontsize)
    axs_counter += 1
    all_runs['Game, then \nUtilitarian'].plot(kind='bar', stacked=True, ax=axs[axs_counter], title='Game, then \nUtilitarian', legend=False, color=colors, fontsize=fontsize)
    axs[axs_counter].set_title('Game, then \nUtilitarian', fontsize=fontsize)
    plt.legend(bbox_to_anchor=(1.9, 0.5), loc='center', fontsize=fontsize)
    #plt.xlabel('Prompt type')
    axs[0].set_ylabel(f"M's action | O's prev. move \n (% of test time responses)", fontsize=fontsize)
    axs[3].set_xlabel('Iterated Game', fontsize=fontsize)
    plt.suptitle(f'Action choices on five iterated matrix games \n (all models trained vs {opponent} opponent)', fontsize=fontsize+10, y=1.49)


    # Adjust layout to prevent clipping of labels
    plt.tight_layout()

    # Save the plot 
    plt.savefig(f'{SAVE_FIGURES_PATH}/RESULTS/{EVALS_dir}/othergames_actiontypes_opp{opponent}_with2baselines.pdf', bbox_inches='tight', dpi=300)

    # Show the plot
    plt.show()


def visualise_unstructured_eval(opponent, extra, options, C_str, D_str):
    index_actions = [f'{C_str}', f'{D_str}', 'other']

    iterables = [['No fine-tuning', 'Game payoffs', 'Deontological', 'Utilitarian', 'Game + \nDeontological', 'Game, then \nDeontological', 'Game, then \nUtilitarian'],
                  index_actions]
    multiindex = pd.MultiIndex.from_product(iterables, names=["PART", "prompt_type"])
    #    Action-only, Action+game, Action+game+state, explicit IPD
    all_runs = pd.DataFrame(index=['Unstructured IPD, with payoffs', 'IPD-like situation, no payoffs'],
                 columns=multiindex)

    columns = ['response (before) - unstructured_IPD', 'response (after) - unstructured_IPD','response (before) - poetic_IPD', 'response (after) - poetic_IPD']
    num_episodes_trained = 1000

    ylab = 'Game payoffs' #we will do the \Np fine-tuning part in parallel with this one' 
    PARTs_detail = '_PT2'
    allruns_singlepart = pd.DataFrame(columns=columns, index=index_actions)
    allruns_singlepart.index = allruns_singlepart.index.rename('index') 
    try: 
        os.chdir(f'gemma-2-2b-it_FT_{PARTs_detail}_opp{opponent}_{num_episodes_trained}ep_{options[0]}')
    except:
        os.chdir(f'../gemma-2-2b-it_FT_{PARTs_detail}_opp{opponent}_{num_episodes_trained}ep_{options[0]}')
    for run_idx in [1,2,3,5,6]: 
    #for run_idx in [1,2,3,5,]: 
        run_df = pd.read_csv(f'run{run_idx}{extra}/EVAL After FT {PARTs_detail} - independent eval 2 unstructured IPD queries.csv', index_col=0)[columns] #[['episode'] + columns]
        for col in run_df[columns]:
            run_df[col] = run_df[col].str.strip().apply(lambda x: x if x in [C_str, D_str] else 'other')

        run_counts = pd.DataFrame(index=index_actions, columns=columns)
        #count every value in the cells of run_df, separately for each column
        for col in run_df[columns]:
            run_counts[col] = run_df[col].value_counts().to_frame()
        run_counts.index = run_counts.index.rename('index')
        #convert numbers to percentages
        run_percent = run_counts * 100 / run_counts.sum(axis=0)
        concat_df = pd.concat([allruns_singlepart.reset_index(), run_percent.reset_index()])
        allruns_singlepart = concat_df.groupby(by='index').agg({col:'mean' for col in columns})
        #scale once again to fit on the 1-100 scale
        allruns_singlepart = allruns_singlepart * 100 / allruns_singlepart.sum(axis=0)

        all_runs['No fine-tuning'] = allruns_singlepart[['response (after) - unstructured_IPD', 'response (after) - poetic_IPD']].rename(
            columns={'response (after) - unstructured_IPD': 'Unstructured IPD, with payoffs', 'response (after) - poetic_IPD': 'IPD-like situation, no payoffs'}).T
        try: 
            all_runs[ylab] = allruns_singlepart[['response (after) - unstructured_IPD', 'response (after) - poetic_IPD']].rename(
            columns={'response (after) - unstructured_IPD': 'Unstructured IPD, with payoffs', 'response (after) - poetic_IPD': 'IPD-like situation, no payoffs'}).T
        except: 
            pass

    ylab = 'Deontological'
    PARTs_detail = '_PT3'
    option = options[0]
    all_runs[ylab] = calculate_eval_responses_unstructured(PARTs_detail, opponent, num_episodes_trained, option, extra, C_str, D_str, columns)
    
    ylab = 'Utilitarian'
    PARTs_detail = '_PT3'
    option = options[1]
    all_runs[ylab] = calculate_eval_responses_unstructured(PARTs_detail, opponent, num_episodes_trained, option, extra, C_str, D_str, columns)
    
    ylab = 'Game + \nDeontological'
    PARTs_detail = '_PT4'
    option = options[0]
    try:  
        all_runs[ylab] = calculate_eval_responses_unstructured(PARTs_detail, opponent, num_episodes_trained, option, extra, C_str, D_str, columns)
    except: 
        pass
    
    ylab = 'Game, then \nDeontological'
    PARTs_detail = '_PT3after2' #'_PT2_PT3'
    option = options[0]
    all_runs[ylab] = calculate_eval_responses_unstructured(PARTs_detail, opponent, 500, option, extra, C_str, D_str, columns)

    ylab = 'Game, then \nUtilitarian'
    PARTs_detail = '_PT3after2' #'_PT2_PT3'
    option = options[1]
    all_runs[ylab] = calculate_eval_responses_unstructured(PARTs_detail, opponent, 500, option, extra, C_str, D_str, columns)

    fig, axs = plt.subplots(1, 8, figsize=(55, 7), sharey=True)
    fontsize=70
    colors = ['blue', 'red', 'grey']
    all_runs['No fine-tuning'].plot(kind='bar', stacked=True, ax=axs[0], legend=False, color=colors, fontsize=fontsize)
    axs[0].set_title('No fine-tuning\n', fontsize=fontsize)
    all_runs['Game payoffs'].plot(kind='bar', stacked=True, ax=axs[1], legend=False, color=colors, fontsize=fontsize)
    axs[1].set_title('Game \npayoffs', fontsize=fontsize)
    all_runs['Deontological'].plot(kind='bar', stacked=True, ax=axs[2], legend=False, color=colors, fontsize=fontsize)
    axs[2].set_title('Deontological\n', fontsize=fontsize)
    all_runs['Utilitarian'].plot(kind='bar', stacked=True, ax=axs[3], legend=False, color=colors, fontsize=fontsize)
    axs[3].set_title('Utilitarian\n', fontsize=fontsize)
    all_runs['Game + \nDeontological'].plot(kind='bar', stacked=True, ax=axs[4], legend=False, color=colors, fontsize=fontsize)
    axs[4].set_title('Game + \nDeontological', fontsize=fontsize)
    all_runs['Game, then \nDeontological'].plot(kind='bar', stacked=True, ax=axs[5], legend=False, color=colors, fontsize=fontsize)
    axs[5].set_title('Game, then \nDeontological', fontsize=fontsize)
    all_runs['Game, then \nUtilitarian'].plot(kind='bar', stacked=True, ax=axs[6], title='Game, then \nUtilitarian', legend=False, color=colors, fontsize=fontsize)
    axs[6].set_title('Game, then \nUtilitarian', fontsize=fontsize)
    plt.legend(bbox_to_anchor=(1.9, 0.5), loc='center', fontsize=fontsize, labels=[f'{C_str} = C', f'{D_str} = D', 'other'])
    #plt.xlabel('Prompt type')
    axs[0].set_ylabel(f'Action choices \n (% of test time responses)', fontsize=fontsize)
    axs[3].set_xlabel('Test Prompt type', fontsize=fontsize)
    plt.suptitle(f'Action choices on unstructured prompts \n (all models trained vs {opponent} opponent)', fontsize=fontsize+10, y=1.49)


    # Adjust layout to prevent clipping of labels
    plt.tight_layout()

    # Save the plot 
    plt.savefig(f'{SAVE_FIGURES_PATH}/RESULTS/{EVALS_dir}/unstructured_opp{opponent}.pdf', bbox_inches='tight', dpi=300)

    # Show the plot
    plt.show()


def calculate_eval_responses_unstructured(PARTs_detail, opponent, num_episodes_trained, option, extra, C_str, D_str, columns):
    index_actions = [f'{C_str}', f'{D_str}', 'other']
    allruns_singlepart = pd.DataFrame(columns=columns, index=index_actions)
    allruns_singlepart.index = allruns_singlepart.index.rename('index') 
    try: 
        os.chdir(f'gemma-2-2b-it_FT_{PARTs_detail}_opp{opponent}_{num_episodes_trained}ep_{option}')
    except: 
        os.chdir(f'../gemma-2-2b-it_FT_{PARTs_detail}_opp{opponent}_{num_episodes_trained}ep_{option}')
             
    for run_idx in [1,2,3,5,6]: #NOTE add RUN 6 @!!! 
        run_df = pd.read_csv(f'run{run_idx}{extra}/EVAL After FT {PARTs_detail} - independent eval 2 unstructured IPD queries.csv', index_col=0)[columns] #[['episode'] + columns]
        for col in run_df[columns]:
            run_df[col] = run_df[col].str.strip().apply(lambda x: x if x in [C_str, D_str] else 'other')

        run_counts = pd.DataFrame(index=index_actions, columns=columns)
        #count every value in the cells of run_df, separately for each column
        for col in run_df[columns]:
            run_counts[col] = run_df[col].value_counts().to_frame()
        run_counts.index = run_counts.index.rename('index')
        #convert numbers to percentages
        run_percent = run_counts * 100 / run_counts.sum(axis=0)
        concat_df = pd.concat([allruns_singlepart.reset_index(), run_percent.reset_index()])
        allruns_singlepart = concat_df.groupby(by='index').agg({col:'mean' for col in columns})
        #scale once again to fit on the 1-100 scale
        allruns_singlepart = allruns_singlepart * 100 / allruns_singlepart.sum(axis=0)
        return allruns_singlepart[['response (after) - unstructured_IPD', 'response (after) - poetic_IPD']].rename(
            columns={'response (after) - unstructured_IPD': 'Unstructured IPD, with payoffs', 'response (after) - poetic_IPD': 'IPD-like situation, no payoffs'}).T


def visualise_unstructured_eval_4ipdprompts(opponent, extra, options, C_str, D_str):

    iterables = [['No fine-tuning', 'Game payoffs', 'Deontological', 'Utilitarian', 'Game + \nDeontological', 'Game, then \nDeontological', 'Game, then \nUtilitarian'],
                  ['C | C', 'C | D', 'D | C', 'D | D', 'illegal | C', 'illegal | D']]
    multiindex = pd.MultiIndex.from_product(iterables, names=["PART", "Action Types"])
    #games = ['IPD', 'ISH', 'ICN', 'BOS', 'ICD']
    columns = ['structured_IPD', 
               'unstructured_IPD',
               'poetic_IPD',
               'explicit_IPD']

    all_runs = pd.DataFrame(index=columns,
                            columns = multiindex)
    
    num_episodes_eval = 10 
    game_placeholder = '2 unstructured IPD queries'

    ylab = 'No fine-tuning'
    for col in columns:
        allruns_singlepart = process_actions_eval('_PT2', opponent, 1000, options[0], game_placeholder, extra, C_str, D_str, num_episodes_eval, before_or_after='before', options=col)
        all_runs.loc[col][ylab] = allruns_singlepart.mean(axis=1).values

    #ylab = 'No fine-tuning, \nDeont. prompted'
    #try: 
    #    for col in columns:
    #        allruns_singlepart = process_actions_eval_valuepromptedbaseline('_PT2', opponent, 1000, options[0], game_placeholder, extra, C_str, D_str, num_episodes_eval, before_or_after='before', value='De', options=col)
    #        all_runs.loc[col][ylab] = allruns_singlepart.mean(axis=1).values
    #except: 
    #    print('unable to process new baseline (value prompted base model) - missing data')

    #ylab = 'No fine-tuning, \nUtilit. prompted'
    #try: 
    #    for col in columns:
    #        allruns_singlepart = process_actions_eval_valuepromptedbaseline('_PT2', opponent, 1000, options[0], game_placeholder, extra, C_str, D_str, num_episodes_eval, before_or_after='before', value='Ut', options=col)
    #        all_runs.loc[col][ylab] = allruns_singlepart.mean(axis=1).values
    #except: 
    #    print('unable to process new baseline (value prompted base model) - missing data')
        

    for col in columns:
        all_runs.loc[col]['Game payoffs'] = process_actions_eval('_PT2', opponent, 1000, options[0], game_placeholder, extra, C_str, D_str, num_episodes_eval, before_or_after='after', options=col).mean(axis=1).values
        all_runs.loc[col]['Deontological'] = process_actions_eval('_PT3', opponent, 1000, options[0], game_placeholder, extra, C_str, D_str, num_episodes_eval, before_or_after='after', options=col).mean(axis=1).values
        all_runs.loc[col]['Utilitarian'] = process_actions_eval('_PT3', opponent, 1000, options[1], game_placeholder, extra, C_str, D_str, num_episodes_eval, before_or_after='after', options=col).mean(axis=1).values
        all_runs.loc[col]['Game + \nDeontological'] = process_actions_eval('_PT4', opponent, 1000, options[0], game_placeholder, extra, C_str, D_str, num_episodes_eval, before_or_after='after', options=col).mean(axis=1).values
        all_runs.loc[col]['Game, then \nDeontological'] = process_actions_eval('_PT3after2', opponent, 500, options[0], game_placeholder, extra, C_str, D_str, num_episodes_eval, before_or_after='after', options=col).mean(axis=1).values
        all_runs.loc[col]['Game, then \nUtilitarian'] = process_actions_eval('_PT3after2', opponent, 500, options[1], game_placeholder, extra, C_str, D_str, num_episodes_eval, before_or_after='after', options=col).mean(axis=1).values

    #all_runs.index=["Iterated Prisoner's Dilemma", "Iterated Stag Hunt", "Iterated Chicken", "Iterated Bach-or-Stravinsky", "Iterated Defective Coordination"]
    all_runs.rename(index={'structured_IPD': '(Training prompt) Structured IPD, with payoffs',
                    'unstructured_IPD': 'Unstructured IPD, with payoffs', 
                    'poetic_IPD': 'IPD-like situation, no payoffs',
                    'explicit_IPD': 'Explicit IPD, assumed payoffs'},
                     inplace=True)

    fig, axs = plt.subplots(1, 7, figsize=(65, 7), sharey=True)
    fontsize=70
    #colors = {'C|C':'#28641E', 'C|D':'#B0DC82', 'D|C':'#FBE6F1', 'D|D':'#8E0B52', 
    #          'illegal|C':'#A9A9A9', 'illegal|D':'#A9A9A9', 'C|illegal':'#A9A9A9', 'D|illegal':'#A9A9A9', 'illegal|illegal':'#A9A9A9'}
    colors = ['#28641E', '#B0DC82', '#FBE6F1', '#8E0B52', 
                '#A9A9A9', '#A9A9A9', '#A9A9A9', '#A9A9A9', '#A9A9A9']
    all_runs['No fine-tuning'].plot(kind='bar', stacked=True, ax=axs[0], legend=False, color=colors, fontsize=fontsize)
    axs[0].set_title('No fine-tuning\n', fontsize=fontsize)
    all_runs['Game payoffs'].plot(kind='bar', stacked=True, ax=axs[1], legend=False, color=colors, fontsize=fontsize)
    axs[1].set_title('Game \npayoffs', fontsize=fontsize)
    all_runs['Deontological'].plot(kind='bar', stacked=True, ax=axs[2], legend=False, color=colors, fontsize=fontsize)
    axs[2].set_title('Deontological\n', fontsize=fontsize)
    all_runs['Utilitarian'].plot(kind='bar', stacked=True, ax=axs[3], legend=False, color=colors, fontsize=fontsize)
    axs[3].set_title('Utilitarian\n', fontsize=fontsize)
    all_runs['Game + \nDeontological'].plot(kind='bar', stacked=True, ax=axs[4], legend=False, color=colors, fontsize=fontsize)
    axs[4].set_title('Game + \nDeontological', fontsize=fontsize)
    all_runs['Game, then \nDeontological'].plot(kind='bar', stacked=True, ax=axs[5], legend=False, color=colors, fontsize=fontsize)
    axs[5].set_title('Game, then \nDeontological', fontsize=fontsize)
    all_runs['Game, then \nUtilitarian'].plot(kind='bar', stacked=True, ax=axs[6], title='Game, then \nUtilitarian', legend=False, color=colors, fontsize=fontsize)
    axs[6].set_title('Game, then \nUtilitarian', fontsize=fontsize)
    plt.legend(bbox_to_anchor=(1.9, 0.5), loc='center', fontsize=fontsize)
    #plt.xlabel('Prompt type')
    axs[0].set_ylabel(f"M's action | O's prev. move \n (% of test time responses)", fontsize=fontsize)
    axs[3].set_xlabel('Iterated Game', fontsize=fontsize)
    plt.suptitle(f'Action choices on four types of IPD prompt \n (all models trained vs {opponent} opponent)', fontsize=fontsize+10, y=1.49)

    # Adjust layout to prevent clipping of labels
    plt.tight_layout()

    # Save the plot 
    plt.savefig(f'{SAVE_FIGURES_PATH}/RESULTS/{EVALS_dir}/fourIPDprompts_actiontypes_opp{opponent}.pdf', bbox_inches='tight', dpi=300)

    # Show the plot
    plt.show()




'EVAL After FT _PT2 - independent eval 2 unstructured IPD queries'

'EVALaction12vsRandom_samestate_orderoriginal_NEW/' -> 'EVAL Before VALUE De PROMPTED _PT2 - independent eval BOS.csv'

'EVALaction34vsRandom_samestate_orderoriginal/'

os.getcwd()
os.listdir()


#### aggregate across runs ####
os.chdir('RESULTS/from Myriad cluster')
#option = ["JF", "LW"] #['X', 'Y'] #['Y', X'] #["LW","JF"] #["JF", "LW"]
#C_symbol, D_symbol = option[0], option[1]

do_PT1 = False 
do_PT2 = True 
do_PT3= True 
do_PT4 = True
do_PT2then3 = True

#PARTs_detail = '_PT2_PT3'
 # '_PT2', '_PT3', '_PT4' #_PT2_PT3


model_name = 'gemma-2-2b-it' #gpt2 #
seqN = 2 #7 #1, 3, 7
#n_runs = 5
run_idxs = [1,2,3,5,6]
BATCH_SIZE = 5 

opponent = 'TFT'
opponent = 'Random' 
opponent = 'AD'
opponent = 'AC'

option = 'COREDe'
option = 'COREUt' #not for PT2, PT4 


opponent = 'LLM'

option = 'CORE-2learners'
option = 'CORE-2learnersUt'

run_idxs = [1,2,3,5,6]


#################################################################################
#### Visualise types of actions (action | state) over time - DURING learning ####
#################################################################################
#if do_PT1: 
#    plot_action_types_perepisode_aggruns(n_runs=5, num_episodes=1000, PART=1, extra='(Legal rewards)', opponent=opponent, C_symbol=C_symbol, D_symbol=D_symbol)
legendparam = False #'outside' #'inside'
if do_PT2:
    PARTs_detail = '_PT2'
    os.chdir(f'../gemma-2-2b-it_FT_{PARTs_detail}_opp{opponent}_{option}')
    process_CD_tokens(run_idxs, num_episodes=1000, PART=2, extra='(Game rewards)')
    plot_action_types_perepisode_aggruns(run_idxs=run_idxs, PART=2, opponent=opponent, extra='Game', legend=legendparam)

if do_PT3: 
    PARTs_detail = '_PT3'
    os.chdir(f'../gemma-2-2b-it_FT_{PARTs_detail}_opp{opponent}_{option}')
    if option in ['COREDe', 'CORE-2learners']:
        process_CD_tokens(run_idxs, num_episodes=1000, PART=3, extra='(De rewards)')
        plot_action_types_perepisode_aggruns(run_idxs=run_idxs, PART=3, opponent=opponent, extra='Deontological', legend=legendparam)   
    elif option in ['COREUt', 'CORE-2learnersUt']:
        process_CD_tokens(run_idxs, num_episodes=1000, PART=3, extra='(Ut rewards)')
        plot_action_types_perepisode_aggruns(run_idxs=run_idxs, PART=3, opponent=opponent, extra='Utilitarian', legend=legendparam)

if do_PT2then3:
    PARTs_detail = '_PT2_PT3'
    os.chdir(f'../gemma-2-2b-it_FT_{PARTs_detail}_opp{opponent}_{option}')
    if option in ['COREDe', 'CORE-2learners']:
        process_CD_tokens(run_idxs, num_episodes=1000, PART="2then3", extra=['(Game rewards', 'De rewards)'], moral_type='De')
        plot_action_types_perepisode_aggruns(run_idxs=run_idxs, PART="2then3", extra='Game then Deont.', opponent=opponent, legend=legendparam)
    elif option in ['COREUt', 'CORE-2learnersUt']:
        process_CD_tokens(run_idxs, num_episodes=1000, PART="2then3", extra=['(Game rewards', 'Ut rewards)'], moral_type='Ut')
        plot_action_types_perepisode_aggruns(run_idxs=run_idxs, PART="2then3", extra='Game then Utilit.', opponent=opponent, legend='outside')
    
if do_PT4: 
    PARTs_detail = '_PT4'
    if option in ['COREDe', 'CORE-2learners']:
        os.chdir(f'../gemma-2-2b-it_FT_{PARTs_detail}_opp{opponent}_{option}')
        process_CD_tokens(run_idxs, num_episodes=1000, PART=4, extra='(De&Game rewards)')
        #plot_action_types_perepisode_aggruns(run_idxs=run_idxs, PART=4, opponent=opponent, extra='Game+Deont.', legend='inside')
        plot_action_types_perepisode_aggruns(run_idxs=run_idxs, PART=4, opponent=opponent, extra='Game+Deont.', legend='outside')


# check the case for PT4 with Rillegal=-15
os.chdir('../gemma-2-2b-it_FT__PT4_oppLLM_CORE-2learners-Rillegal15')
opponent = 'LLM'
process_CD_tokens(run_idxs, num_episodes=500, PART=4, extra='(De&Game rewards)')
plot_action_types_perepisode_aggruns(run_idxs=run_idxs, PART=4, opponent=opponent, extra='Game+Deontological\n', addition='(Rillegal = -15)', legend='outside')

os.chdir('../gemma-2-2b-it_FT__PT4_oppLLM_CORE-2learners')
opponent = 'LLM'
process_CD_tokens(run_idxs, num_episodes=500, PART=4, extra='(De&Game rewards)')
plot_action_types_perepisode_aggruns(run_idxs=run_idxs, PART=4, opponent=opponent, extra='Game+Deontological\n', addition='(Rillegal = -6)', legend='outside') #as in main paper 


#check the case for PT2 where we swap action tokens 
os.chdir('../../from Myriad cluster')
os.getcwd()
os.listdir()
os.chdir('../gemma-2-2b-it_FT__PT2_oppTFT_CORE-action21De') 
opponent = 'TFT'
#option = 'CORE-action21De'
process_CD_tokens(run_idxs, num_episodes=1000, PART=2, extra='(Game rewards)')
plot_action_types_perepisode_aggruns(run_idxs=run_idxs, PART=2, opponent=opponent, extra='Game', addition='(action2=C, action1=D)', legend='outside')

os.chdir('../gemma-2-2b-it_FT__PT2_oppTFT_COREDe') 
opponent = 'TFT'
process_CD_tokens(run_idxs, num_episodes=1000, PART=2, extra='(Game rewards)')
plot_action_types_perepisode_aggruns(run_idxs=run_idxs, PART=2, opponent=opponent, extra='Game', addition='(action1=C, action2=D)', legend='outside') #as in the main paper


#plot separate runs 
plot_action_types_perepisode_separateruns(run_idxs=run_idxs, num_episodes=1000, opponent=opponent, do_PT1=do_PT1, do_PT2=do_PT2, do_PT3=do_PT3, do_PT4=do_PT4)



#################################################################################
#### Visualise regret & reward over time - DURING learning ####
#####################################################################

plot_regret_during_allgames(opponent='TFT', num_episodes=1000, extra='', options=['COREDe', 'COREUt'])
plot_regret_during_allgames(opponent='LLM', num_episodes=1000, extra='', options=['CORE-2learners', 'CORE-2learnersUt'])


plot_reward_during_allgames(opponent='TFT', num_episodes=1000, extra='', options=['COREDe', 'COREUt'])
plot_reward_during_allgames(opponent='LLM', num_episodes=1000, extra='', options=['CORE-2learners', 'CORE-2learnersUt'])
plot_reward_during_allgames(opponent='Random', num_episodes=1000, extra='', options=['COREDe', 'COREUt'])
plot_reward_during_allgames(opponent='AD', num_episodes=1000, extra='', options=['COREDe', 'COREUt'])
plot_reward_during_allgames(opponent='AC', num_episodes=1000, extra='', options=['COREDe', 'COREUt'])



#################################################################################
#### Visualise reward (regret) across 3 games EVAL ####
#################################################################################

os.getcwd()

EVALS_dir = 'EVALaction34vsRandom_samestate_orderoriginal'
EVALS_dir = 'EVALaction12vsRandom_samestate_orderreversed'
EVALS_dir = 'EVALaction21vsRandom_samestate_orderoriginal'

#NEW - 2 more permutations and actionn34 reversed meaning 
EVALS_dir = 'EVALaction34vsRandom_samestate_orderpermuted1'
EVALS_dir = 'EVALaction34vsRandom_samestate_orderpermuted2'
EVALS_dir = 'EVALaction34vsRandom_samestate_orderreversed'

os.chdir(f'{SAVE_FIGURES_PATH}/RESULTS/{EVALS_dir}/')

if False: 
    num_episodes_trained=1000
    r_toplot = 'rewards_Ut'
    ylab = 'Utilitarian'

    r_toplot = 'rewards_De'
    ylab = 'Deontological'

    moral_max = 6
    PARTs_detail= '_PT3'
    option= 'COREUt'
    os.chdir('gemma-2-2b-it_FT__PT3_oppTFT_1000ep_COREUt')

    def plot_regret(r_toplot, ylab, moral_max, PARTs_detail, opponent, num_episodes_trained, option):
        os.chdir(f'../gemma-2-2b-it_FT_{PARTs_detail}_opp{opponent}_{num_episodes_trained}ep_{option}')
        all_runs = pd.DataFrame(index = [f'{r_toplot} (before)', f'{r_toplot} (after)', 'Moral Regret (before)', 'Moral Regret (after)'], columns = ['IPD', 'ISH', 'IVD'])
        for run in [1,2,3,5,6]:
            run_IPD = pd.read_csv(f'run{run}/EVAL After FT {PARTs_detail} - independent eval IPD.csv', index_col=0)
            run_IPD_episodes = run_IPD[['episode', f'{r_toplot} (before)', f'{r_toplot} (after)']].groupby('episode').mean()
            run_IPD_episodes.mean()
            run_IPD_episodes['Moral Regret (before)'] = moral_max - run_IPD_episodes[f'{r_toplot} (before)'] 
            run_IPD_episodes['Moral Regret (after)'] = moral_max - run_IPD_episodes[f'{r_toplot} (after)'] 

            run_ISH = pd.read_csv(f'run{run}/EVAL After FT {PARTs_detail} - independent eval ISH.csv', index_col=0)
            #run_ISH[[f'{r_toplot} (before)', f'{r_toplot} (after)']].mean().plot(kind='bar', ylabel=f'{ylab} Reward', title = f'EVAL Stag Hunt \n after {ylab} fine-tuning')
            run_ISH_episodes = run_ISH[['episode', f'{r_toplot} (before)', f'{r_toplot} (after)']].groupby('episode').mean()
            run_ISH_episodes['Moral Regret (before)'] = moral_max - run_ISH_episodes[f'{r_toplot} (before)'] 
            run_ISH_episodes['Moral Regret (after)'] = moral_max - run_ISH_episodes[f'{r_toplot} (after)'] 

            run_IVD = pd.read_csv(f'run{run}/EVAL After FT {PARTs_detail} - independent eval IVD.csv', index_col=0)
            #run_IVD[[f'{r_toplot} (before)', f'{r_toplot} (after)']].mean().plot(kind='bar', ylabel=f'{ylab} Reward', title = f'EVAL Chicken \n after {ylab} fine-tuning')
            run_IVD_episodes = run_IVD[['episode', f'{r_toplot} (before)', f'{r_toplot} (after)']].groupby('episode').mean()
            run_IVD_episodes['Moral Regret (before)'] = moral_max - run_ISH_episodes[f'{r_toplot} (before)'] 
            run_IVD_episodes['Moral Regret (after)'] = moral_max - run_ISH_episodes[f'{r_toplot} (after)'] 

            run_toplot = pd.DataFrame({'IPD': run_IPD_episodes.mean(), 
                                        'ISH': run_ISH_episodes.mean(),#[['{r_toplot} (before)', '{r_toplot} (after)']].mean().values, 
                                        'IVD': run_IVD_episodes.mean()},#[['{r_toplot} (before)', '{r_toplot} (after)']].mean().values}, 
                                        index=[f'{r_toplot} (before)', f'{r_toplot} (after)', 'Moral Regret (before)', 'Moral Regret (after)'])
            
            all_runs = pd.concat((all_runs, run_toplot)).reset_index().groupby(by="index").mean()
            #average out two dataframes 
            #all_runs = all_runs.add(run_toplot, fill_value=0)
        #all_runs.reset_index().groupby("index").mean()

        all_runs.T[['Moral Regret (before)', 'Moral Regret (after)']].plot(kind='bar', ylabel=f'{ylab} Moral Regret', title = f'EVAL \n after {ylab} fine-tuning \n on the IPD')

    plot_regret('rewards_De', 'Deontological', 0, '_PT3', 'TFT', 1000, 'COREDe')
    plot_regret('rewards_Ut', 'Utilitarian', 6, '_PT3', 'TFT', 1000, 'COREUt')



    def plot_regret_game(r_toplot, label, moral_max, opponent, num_episodes_trained, option, game, color):
        all_runs = pd.DataFrame(index=[f'{label} Moral Regret'],
                                #index = [f'{r_toplot} (before)', f'{r_toplot} (after)', 'Moral Regret (before)', 'Moral Regret (after)'], 
                                columns = ['No fine-tuning', 'Game payoffs', 'Deontological', 'Utilitarian', 'Game + Deontological', 'Game, then Deontological', 'Game, then Utilitarian'])
        
        ylab = 'No fine-tuning'
        PARTs_detail = '_PT2'
        option = 'COREDe'
        allruns_singlepart = pd.DataFrame(columns=[f'run{run_idx}' for run_idx in [1,2,3,5,6]], index=[f'{label} Moral Regret'])
        os.chdir(f'../gemma-2-2b-it_FT_{PARTs_detail}_opp{opponent}_{num_episodes_trained}ep_{option}')
        for run_idx in [1,2,3,5,6]:
            run_df = pd.read_csv(f'run{run_idx}/EVAL After FT {PARTs_detail} - independent eval {game}.csv', index_col=0)
            run_episodes = run_df[['episode', f'{r_toplot} (before)']].groupby('episode').mean()
            run_episodes_regret = moral_max - run_episodes.reset_index()[f'{r_toplot} (before)']
            #store this run's value for Moral Regret (before)
            allruns_singlepart[f'run{run_idx}'][f'{label} Moral Regret'] = run_episodes_regret.mean()
        all_runs[ylab][f'Deontological Moral Regret'] = allruns_singlepart.mean(axis=1)[f'{label} Moral Regret']

        ylab = 'Game payoffs'
        PARTs_detail = '_PT2'
        option = 'COREDe'
        allruns_singlepart = pd.DataFrame(columns=[f'run{run_idx}' for run_idx in [1,2,3,5,6]], index=[f'{label} Moral Regret'])
        os.chdir(f'../gemma-2-2b-it_FT_{PARTs_detail}_opp{opponent}_{num_episodes_trained}ep_{option}')
        for run_idx in [1,2,3,5,6]:
            run_df = pd.read_csv(f'run{run_idx}/EVAL After FT {PARTs_detail} - independent eval {game}.csv', index_col=0)
            run_episodes = run_df[['episode', f'{r_toplot} (after)']].groupby('episode').mean()
            run_episodes_regret = moral_max - run_episodes.reset_index()[f'{r_toplot} (after)']
            #store this run's value for Moral Regret (after)
            allruns_singlepart[f'run{run_idx}'][f'{label} Moral Regret'] = run_episodes_regret.mean()
        all_runs[ylab][f'{label} Moral Regret'] = allruns_singlepart.mean(axis=1)[f'{label} Moral Regret']

        ylab = 'Deontological'
        PARTs_detail = '_PT3'
        option = 'COREDe'
        allruns_singlepart = pd.DataFrame(columns=[f'run{run_idx}' for run_idx in [1,2,3,5,6]], index=[f'{label} Moral Regret'])
        os.chdir(f'../gemma-2-2b-it_FT_{PARTs_detail}_opp{opponent}_{num_episodes_trained}ep_{option}')
        for run_idx in [1,2,3,5,6]:
            run_df = pd.read_csv(f'run{run_idx}/EVAL After FT {PARTs_detail} - independent eval {game}.csv', index_col=0)
            run_episodes = run_df[['episode', f'{r_toplot} (after)']].groupby('episode').mean()
            run_episodes_regret = moral_max - run_episodes.reset_index()[f'{r_toplot} (after)']
            #store this run's value for Moral Regret (after)
            allruns_singlepart[f'run{run_idx}'][f'{label} Moral Regret'] = run_episodes_regret.mean()
        all_runs[ylab][f'{label} Moral Regret'] = allruns_singlepart.mean(axis=1)[f'{label} Moral Regret']

        ylab = 'Utilitarian'
        PARTs_detail = '_PT3'
        option = 'COREUt'
        allruns_singlepart = pd.DataFrame(columns=[f'run{run_idx}' for run_idx in [1,2,3,5,6]], index=[f'{label} Moral Regret'])
        os.chdir(f'../gemma-2-2b-it_FT_{PARTs_detail}_opp{opponent}_{num_episodes_trained}ep_{option}')
        for run_idx in [1,2,3,5,6]:
            run_df = pd.read_csv(f'run{run_idx}/EVAL After FT {PARTs_detail} - independent eval {game}.csv', index_col=0)
            run_episodes = run_df[['episode', f'{r_toplot} (after)']].groupby('episode').mean()
            run_episodes_regret = moral_max - run_episodes.reset_index()[f'{r_toplot} (after)']
            #store this run's value for Moral Regret (after)
            allruns_singlepart[f'run{run_idx}'][f'{label} Moral Regret'] = run_episodes_regret.mean()
        all_runs[ylab][f'{label} Moral Regret'] = allruns_singlepart.mean(axis=1)[f'{label} Moral Regret']

        ylab = 'Game + Deontological'
        PARTs_detail = '_PT4'
        option = 'COREDe'
        allruns_singlepart = pd.DataFrame(columns=[f'run{run_idx}' for run_idx in [1,2,3,5,6]], index=[f'{label} Moral Regret'])
        os.chdir(f'../gemma-2-2b-it_FT_{PARTs_detail}_opp{opponent}_{num_episodes_trained}ep_{option}')
        for run_idx in [1,2,3,5,6]:
            run_df = pd.read_csv(f'run{run_idx}/EVAL After FT {PARTs_detail} - independent eval {game}.csv', index_col=0)
            run_episodes = run_df[['episode', f'{r_toplot} (after)']].groupby('episode').mean()
            run_episodes_regret = moral_max - run_episodes.reset_index()[f'{r_toplot} (after)']
            #store this run's value for Moral Regret (after)
            allruns_singlepart[f'run{run_idx}'][f'{label} Moral Regret'] = run_episodes_regret.mean()
        all_runs[ylab][f'{label} Moral Regret'] = allruns_singlepart.mean(axis=1)[f'{label} Moral Regret']

        ylab = 'Game, then Deontological'
        PARTs_detail = '_PT3after2'
        option = 'COREDe'
        allruns_singlepart = pd.DataFrame(columns=[f'run{run_idx}' for run_idx in [1,2,3,5,6]], index=[f'{label} Moral Regret'])
        os.chdir(f'../gemma-2-2b-it_FT_{PARTs_detail}_opp{opponent}_{500}ep_{option}')
        for run_idx in [1,2,3,5,6]:
            run_df = pd.read_csv(f'run{run_idx}/EVAL After FT {PARTs_detail} - independent eval {game}.csv', index_col=0)
            run_episodes = run_df[['episode', f'{r_toplot} (after)']].groupby('episode').mean()
            run_episodes_regret = moral_max - run_episodes.reset_index()[f'{r_toplot} (after)']
            #store this run's value for Moral Regret (after)
            allruns_singlepart[f'run{run_idx}'][f'{label} Moral Regret'] = run_episodes_regret.mean()
        all_runs[ylab][f'{label} Moral Regret'] = allruns_singlepart.mean(axis=1)[f'{label} Moral Regret']

        ylab = 'Game, then Utilitarian'
        PARTs_detail = '_PT3after2'
        option = 'COREUt'
        allruns_singlepart = pd.DataFrame(columns=[f'run{run_idx}' for run_idx in [1,2,3,5,6]], index=[f'{label} Moral Regret'])
        os.chdir(f'../gemma-2-2b-it_FT_{PARTs_detail}_opp{opponent}_{500}ep_{option}')
        for run_idx in [1,2,3,5,6]:
            run_df = pd.read_csv(f'run{run_idx}/EVAL After FT {PARTs_detail} - independent eval {game}.csv', index_col=0)
            run_episodes = run_df[['episode', f'{r_toplot} (after)']].groupby('episode').mean()
            run_episodes_regret = moral_max - run_episodes.reset_index()[f'{r_toplot} (after)']
            #store this run's value for Moral Regret (after)
            allruns_singlepart[f'run{run_idx}'][f'{label} Moral Regret'] = run_episodes_regret.mean()
        all_runs[ylab][f'{label} Moral Regret'] = allruns_singlepart.mean(axis=1)[f'{label} Moral Regret']

        all_runs.T[[f'{label} Moral Regret']].plot(
            kind='bar', ylabel=f'{label} Moral Regret', 
            title = f'Test time {game} performance \n (after fine-tuning on the IPD)',
            legend = False,
            xlabel = 'Fine-tuned models \n (by reward type used)',
            color=color)

    r_toplot = 'rewards_De'
    label = 'Deontological'
    moral_max = 0
    color = 'turquoise'
    plot_regret_game(r_toplot, label, moral_max, opponent, num_episodes_trained, option, 'IPD', color)
    plot_regret_game(r_toplot, label, moral_max, opponent, num_episodes_trained, option, 'ISH', color)
    plot_regret_game(r_toplot, label, moral_max, opponent, num_episodes_trained, option, 'IVD', color)

    r_toplot = 'rewards_Ut'
    label = 'Utilitarian'
    moral_max = 6
    color = 'purple'
    plot_regret_game(r_toplot, label, moral_max, opponent, num_episodes_trained, option, 'IPD', color)
    plot_regret_game(r_toplot, label, moral_max, opponent, num_episodes_trained, option, 'ISH', color)
    plot_regret_game(r_toplot, label, moral_max, opponent, num_episodes_trained, option, 'IVD', color)


#CORE plot for EVAL 
plot_regret_allgames(r_toplot='rewards_De', label='Deontological', opponent='TFT', options=['COREDe', 'COREUt'], extra='', include_NoFT=True)
plot_regret_allgames(r_toplot='rewards_Ut', label='Utilitarian', opponent='TFT', options=['COREDe', 'COREUt'], extra='', include_NoFT=True)

plot_regret_allgames(r_toplot='rewards_De', label='Deontological', opponent='LLM', options=['CORE-2learners', 'CORE-2learnersUt'], extra='_agentM', include_NoFT=True)
plot_regret_allgames(r_toplot='rewards_Ut', label='Utilitarian', opponent='LLM', options=['CORE-2learners', 'CORE-2learnersUt'], extra='_agentM', include_NoFT=True)

#after adding new baseline 
plot_regret_allgames(r_toplot='rewards_De', label='Deontological', opponent='TFT', options=['COREDe', 'COREUt'], extra='', include_NoFT=True, include_Prompted=True)
plot_regret_allgames(r_toplot='rewards_Ut', label='Utilitarian', opponent='TFT', options=['COREDe', 'COREUt'], extra='', include_NoFT=True, include_Prompted=True)





#################################################################################
#### Visualise actions on 4 unrelated prompts EVAL ####
#################################################################################
C_str='action3' 
D_str='action4'

C_str='action1' 
D_str='action2'

C_str='action2'
D_str='action1' 


os.getcwd()
visualise_unrelated_eval(opponent='TFT', extra='', options=['COREDe', 'COREUt'], C_str=C_str, D_str=D_str)
visualise_unrelated_eval(opponent='LLM', extra='_agentM', options=['CORE-2learners', 'CORE-2learnersUt'],  C_str=C_str, D_str=D_str)


visualise_othergames_eval(opponent='TFT', extra='', options=['COREDe', 'COREUt'],  C_str=C_str, D_str=D_str)
visualise_othergames_eval(opponent='LLM', extra='_agentM', options=['CORE-2learners', 'CORE-2learnersUt'],  C_str=C_str, D_str=D_str)

#NEW 
EVALS_dir = 'EVALaction34vsRandom_samestate_orderoriginal'
EVALS_dir = 'EVALaction34vsRandom_samestate_orderpermuted1'
EVALS_dir = 'EVALaction34vsRandom_samestate_orderpermuted2'
EVALS_dir = 'EVALaction34vsRandom_samestate_orderreversed'
os.chdir(f'{SAVE_FIGURES_PATH}/RESULTS/{EVALS_dir}/')
visualise_othergames_eval(opponent='TFT', extra='', options=['COREDe', 'COREUt'],  C_str=C_str, D_str=D_str, include_Prompted=False)
plot_regret_allgames(r_toplot='rewards_De', label='Deontological', opponent='TFT', options=['COREDe', 'COREUt'], extra='', include_NoFT=True, include_Prompted=False)
plot_regret_allgames(r_toplot='rewards_Ut', label='Utilitarian', opponent='TFT', options=['COREDe', 'COREUt'], extra='', include_NoFT=True, include_Prompted=False)




visualise_reciprocity_eval(opponent='TFT', extra='', options=['COREDe', 'COREUt'], C_str=C_str, D_str=D_str, stacked=True)
visualise_reciprocity_eval(opponent='LLM', extra='_agentM', options=['CORE-2learners', 'CORE-2learnersUt'], C_str=C_str, D_str=D_str, stacked=True)

visualise_reciprocity_eval(opponent='TFT', extra='', options=['COREDe', 'COREUt'], C_str=C_str, D_str=D_str, stacked=False)
visualise_reciprocity_eval(opponent='LLM', extra='_agentM', options=['CORE-2learners', 'CORE-2learnersUt'], C_str=C_str, D_str=D_str, stacked=False)


visualise_reciprocity_eval_IPDonly(opponent='TFT', extra='', options=['COREDe', 'COREUt'],  C_str=C_str, D_str=D_str, stacked=True)
visualise_reciprocity_eval_IPDonly(opponent='LLM', extra='_agentM', options=['CORE-2learners', 'CORE-2learnersUt'], C_str=C_str, D_str=D_str, stacked=True)


visualise_reciprocity_eval_IPDonly(opponent='TFT', extra='', options=['COREDe', 'COREUt'],  C_str=C_str, D_str=D_str, stacked=False)
visualise_reciprocity_eval_IPDonly(opponent='LLM', extra='_agentM', options=['CORE-2learners', 'CORE-2learnersUt'], C_str=C_str, D_str=D_str, stacked=False)


EVALS_dir = 'EVALaction12vsRandom_samestate_orderoriginal_NEW'
os.chdir(f'{SAVE_FIGURES_PATH}/RESULTS/{EVALS_dir}/')


EVALS_dir = 'EVALaction34vsRandom_samestate_orderoriginal_NEW'
os.chdir(f'{SAVE_FIGURES_PATH}/RESULTS/{EVALS_dir}/')

EVALS_dir = 'EVALaction34vsRandom_samestate_orderoriginal_NEW'


visualise_unstructured_eval(opponent='TFT', extra='', options=['COREDe', 'COREUt'], C_str=C_str, D_str=D_str)

#vis 4 IPD prompts without the two new baselines 
visualise_unstructured_eval_4ipdprompts(opponent='TFT', extra='', options=['COREDe', 'COREUt'], C_str=C_str, D_str=D_str)
os.getcwd()

#################################################################################
#### Visualise R intrinsic over time - DURING learning ####
#################################################################################



if do_PT2: 
    PART = 2 
    PARTs_detail = '_PT2'
    extra = '(Game rewards)'
    runs = [1,3,4,5]
if do_PT3: 
    PART = 3
    PARTs_detail = '_PT3'
    extra = '(De rewards)'
    runs = [1,2,3,4]
if do_PT4: 
    PART = 4
    PARTs_detail = '_PT4'
    extra = '(De&Game rewards)'
    runs = [1,2,4,5]

os.chdir(f'../gemma-2-2b-it_FT_{PARTs_detail}_opp{opponent}')

plot_reward_per_episode_playerM(runs = runs, num_episodes=1000, opponent=opponent, PART=PART, extra=extra, with_CI=True)

os.getcwd()
os.listdir('run1')









#################################################################################
#### Visualise R intrinsic - AFTER learning ####
#################################################################################


plot_bar_reward_after_playerM(runs=runs, num_episodes=1, opponent=opponent, PART=PART, extra=extra, with_CI=True)





















#################################################################################
#### MINI test ####
options = [["LW","JF"], ["JF", "LW"]]
opponent = 'Random'
opponent = 'AD'

for option in options: 
    C_symbol, D_symbol = option[0], option[1]
    os.chdir(f'../gemma-2-2b-it_FT_{option[0]}{option[1]}_100ep_opp{opponent}_largerR__PT3_newMINI') 
    plot_action_types_perepisode_aggruns(n_runs=1, num_episodes=100, PART=3, extra='(De rewards)', opponent=opponent, C_symbol=C_symbol, D_symbol=D_symbol, addition=f'C={option[0]}, D={option[1]}')



option = ["LW","JF"]
C_symbol, D_symbol = option[0], option[1]
opponent =' Random'
#os.chdir(f'../gemma-2-2b-it_FT_{option[0]}{option[1]}_100ep_opp{opponent}_largerR__PT3_newMINI2') 
os.chdir('../gemma-2-2b-it_FT_LWJF_100ep_oppRandom_largerR__PT3_newMINI2')
plot_action_types_perepisode_aggruns(n_runs=1, num_episodes=100, PART=3, extra='(De rewards)', opponent=opponent, C_symbol=C_symbol, D_symbol=D_symbol, addition=f'C={option[0]}, D={option[1]}')













#### individually for each run ####

#####################################
#plot for training vs Random opponent 
os.chdir('RESULTS/from Myriad cluster/')
os.chdir(f'gpt2_FTT_{option}_1000episodes_opponentRandom/')

run_idx = 1
during_PT1 = pd.read_csv(f'run{run_idx}/During FT PART1 (Legal rewards).csv', index_col=0)
during_PT2 = pd.read_csv(f'run{run_idx}/During FT PART2 (IPD rewards).csv', index_col=0)
during_PT3 = pd.read_csv(f'run{run_idx}/During FT PART3 (De rewards).csv', index_col=0)

after_PT1 = pd.read_csv(f'run{run_idx}/After FT PART1 (Legal rewards).csv', index_col=0)
after_PT2 = pd.read_csv(f'run{run_idx}/After FT PART1&2 (Legal&IPD pyoffs).csv', index_col=0)
after_PT3 = pd.read_csv(f'run{run_idx}/After FT PARTs1-3 (Legal, IPD & De rewards).csv', index_col=0)


#1) plot % tokens X generated per episode 
plot_C_tokens_per_episode(df_original=during_PT1, title='(During PART 1 of fine-tuning \n vs Random opponent)')
#2) plot tokens | opp prev move summed on each episode 
plot_action_types_per_episode(df_original=during_PT1, num_episodes=1000, title='(During PART 1 of fine-tuning \n vs Random opponent)', run_idx=run_idx)

plot_C_tokens_per_episode(df_original=during_PT2, title='(During PART 2 of fine-tuning \n vs Random opponent)')
plot_action_types_per_episode(df_original=during_PT2, num_episodes=1000, title='(During PART 2 of fine-tuning \n vs Random opponent)', run_idx=run_idx)

plot_C_tokens_per_episode(df_original=during_PT3, title='(During PART 3 of fine-tuning \n vs Random opponent)')
plot_action_types_per_episode(df_original=during_PT3, num_episodes=1000, title='(During PART 3 of fine-tuning \n vs Random opponent)', run_idx=run_idx)



#plot eval post-training vs AD opponent
after_PT1['response (after)'].value_counts()
after_PT2['response (after)'].value_counts()
after_PT3['response (after)'].value_counts()


##################################
# plot for training vs AD opponent 
os.chdir(f'../../gpt2_FT_{option}_1000episodes_opponentAD/run1')

during_PT1 = pd.read_csv(f'run{run_idx}/During FT PART1 (Legal rewards).csv', index_col=0)
during_PT2 = pd.read_csv(f'run{run_idx}/During FT PART2 (IPD rewards).csv', index_col=0)
during_PT3 = pd.read_csv(f'run{run_idx}/During FT PART3 (De rewards).csv', index_col=0)

after_PT1 = pd.read_csv(f'run{run_idx}/After FT PART1 (Legal rewards).csv', index_col=0)
after_PT2 = pd.read_csv(f'run{run_idx}/After FT PART1&2 (Legal&IPD pyoffs).csv', index_col=0)
after_PT3 = pd.read_csv(f'run{run_idx}/After FT PARTs1-3 (Legal, IPD & De rewards).csv', index_col=0)

during_PT1['prev_move_O'].value_counts()

plot_C_tokens_per_episode(df_original=during_PT1, title='(During PART 1 of fine-tuning \n vs AD opponent)')
plot_action_types_per_episode(df_original=during_PT1, num_episodes=1000, title='(During PART 1 of fine-tuning \n vs AD opponent)')

plot_C_tokens_per_episode(df_original=during_PT2, title='(During PART 2 of fine-tuning \n vs AD opponent)')
plot_action_types_per_episode(df_original=during_PT2, num_episodes=1000, title='(During PART 2 of fine-tuning \n vs AD opponent)')

plot_C_tokens_per_episode(df_original=during_PT3, title='(During PART 3 of fine-tuning \n vs AD opponent)')
plot_action_types_per_episode(df_original=during_PT3, num_episodes=1000, title='(During PART 3 of fine-tuning \n vs AD opponent)')

#plot eval post-training vs AD opponent
after_PT1['response (after)'].value_counts()
after_PT2['response (after)'].value_counts()
after_PT3['response (after)'].value_counts()



##################################
# plot for training vs TFT opponent 

os.chdir(f'../../gpt2_FT_{option}_1000episodes_opponentTFT') #

during_PT1 = pd.read_csv(f'run{run_idx}/During FT PART1 (Legal rewards).csv', index_col=0)
during_PT2 = pd.read_csv(f'run{run_idx}/During FT PART2 (IPD rewards).csv', index_col=0)
during_PT3 = pd.read_csv(f'run{run_idx}/During FT PART3 (De rewards).csv', index_col=0)

after_PT1 = pd.read_csv(f'run{run_idx}/After FT PART1 (Legal rewards).csv', index_col=0)
after_PT2 = pd.read_csv(f'run{run_idx}/After FT PART1&2 (Legal&IPD pyoffs).csv', index_col=0)
after_PT3 = pd.read_csv(f'run{run_idx}/After FT PARTs1-3 (Legal, IPD & De rewards).csv', index_col=0)


plot_C_tokens_per_episode(df_original=during_PT1, title='(During PART 1 of fine-tuning \n vs TFT opponent)')
plot_action_types_per_episode(df_original=during_PT1, num_episodes=1000, title='(During PART 1 of fine-tuning \n vs TFT opponent)')

plot_C_tokens_per_episode(df_original=during_PT2, title='(During PART 2 of fine-tuning \n vs TFT opponent)')
plot_action_types_per_episode(df_original=during_PT2, num_episodes=1000, title='(During PART 2 of fine-tuning \n vs TFT opponent)')

plot_C_tokens_per_episode(df_original=during_PT3, title='(During PART 3 of fine-tuning \n vs TFT opponent)')
plot_action_types_per_episode(df_original=during_PT3, num_episodes=1000, title='(During PART 3 of fine-tuning \n vs TFT opponent)')

during_PT1['prev_move_O'].value_counts()


#plot eval post-training vs AD opponent
after_PT1['response (after)'].value_counts()
after_PT2['response (after)'].value_counts()
after_PT3['response (after)'].value_counts()





##################################
# plot for training vs AC opponent 
os.chdir(f'../../gpt2_FT_{option}_1000episodes_opponentAC/run1')

during_PT1 = pd.read_csv(f'run{run_idx}/During FT PART1 (Legal rewards).csv', index_col=0)
during_PT2 = pd.read_csv(f'run{run_idx}/During FT PART2 (IPD rewards).csv', index_col=0)
during_PT3 = pd.read_csv('During FT PART3 (De rewards).csv', index_col=0)

after_PT1 = pd.read_csv(f'run{run_idx}/After FT PART1 (Legal rewards).csv', index_col=0)
after_PT2 = pd.read_csv(f'run{run_idx}/After FT PART1&2 (Legal&IPD pyoffs).csv', index_col=0)
after_PT3 = pd.read_csv(f'run{run_idx}/After FT PARTs1-3 (Legal, IPD & De rewards).csv', index_col=0)

during_PT1['prev_move_O'].value_counts()

plot_C_tokens_per_episode(df_original=during_PT1, title='(During PART 1 of fine-tuning \n vs AD opponent)')
plot_action_types_per_episode(df_original=during_PT1, num_episodes=1000, title='(During PART 1 of fine-tuning \n vs AD opponent)')

plot_C_tokens_per_episode(df_original=during_PT2, title='(During PART 2 of fine-tuning \n vs AD opponent)')
plot_action_types_per_episode(df_original=during_PT2, num_episodes=1000, title='(During PART 2 of fine-tuning \n vs AD opponent)')

plot_C_tokens_per_episode(df_original=during_PT3, title='(During PART 3 of fine-tuning \n vs AD opponent)')
plot_action_types_per_episode(df_original=during_PT3, num_episodes=1000, title='(During PART 3 of fine-tuning \n vs AD opponent)')

#plot eval post-training vs AD opponent
after_PT1['response (after)'].value_counts()
after_PT2['response (after)'].value_counts()
after_PT3['response (after)'].value_counts()
