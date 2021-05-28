import os
import json
import numpy as np
import pandas as pd
from glob import glob
from collections import defaultdict

# with open('worker_lookup.json') as json_file:
#     worker_lookup = json.load(json_file)

# CONSTANTS
INCLUDE_INHIB_TASKS = True



ABBREV = {'directed_forgetting_condition': 'DF',
          'flanker_condition':'FLANKER',
          'go_nogo_condition': 'GNG',
          'n_back_condition': 'NBACK',
          'delay_condition': 'DELAY',
          'predictable_condition': 'PREDICT',
          'shape_matching_condition': 'SHAPE',
          'stop_signal_condition': 'SS',
          'task_switch_condition': 'TSWITCH',
          'cue_condition': 'CUE',
          'cued_condition': 'CUE'}

EXPLORE_IDS = ['s264', 's419', 's010', 's320', 's142', 's025', 's066',
               's069', 's205', 's341', 's380', 's141', 's397', 's376',
               's090', 's248', 's207', 's429', 's441', 's214', 's126',
               's539', 's369', 's044', 's454', 's005', 's396', 's365',
               's135', 's360', 's295', 's490', 's350']


# used to fix weird/confusing names of conditions;
# drop unecessary columns.
CONDITION_RENAME_MAP = {
    'predictive_condition': 'predictable_condition',
    'predictive_dimension': 'predictable_dimension',
    'directed_condition': 'directed_forgetting_condition',
    'task_condition': 'task_switch_condition',
    'delay': 'delay_condition',
    'central_letter': 'center_letter',
    }
DROP_COLS = ['Unnamed: 0', 'final_accuracy',
                'final_avg_rt', 'final_credit_var',
                'final_missed_percent', 'final_responses_ok',
                'internal_node_id', 'responses',
                'stimulus', 'text',
                'trial_index', 'trial_type',
                'view_history']


# maps of eash and hard conditions for the various tasks
EASY_MAP = {
    'cued_task_switching': 'CUE:stay.*TSWITCH:stay',
    'n_back': 'DELAY:1',
    'directed_forgetting': 'DF:con',
    'flanker': 'FLANKER:congruent',
    'shape_matching': 'SHAPE:CONTROL',
    'predictable_task_switching': 'PREDICT:stay',
}

HARD_MAP = {
    'cued_task_switching': 'CUE:switch.*TSWITCH:switch',
    'n_back': 'DELAY:2',
    'directed_forgetting': 'DF:neg',
    'flanker': 'FLANKER:incongruent',
    'shape_matching': 'SHAPE:DISTRACTOR',
    'predictable_task_switching': 'PREDICT:switch',  
}

if INCLUDE_INHIB_TASKS:
    EASY_MAP['go_nogo'] = 'GNG:go'
    EASY_MAP['stop_signal'] = 'SS:go'

    HARD_MAP['go_nogo'] = 'GNG:nogo'
    HARD_MAP['stop_signal'] = 'SS:stop'


# HELPERS
# 1. for geneating clean, concatenated data
def get_exp_id(df, base_file):
    if 'exp_id' in df.columns:
        return df.exp_id.unique()[0]
    else:
        return '_'.join(base_file.split('_')[0:-1]).replace('.csv',' ')


def update_correct_trial(df):
    '''Correct simple mis-codings in the correct_trial column.'''
    if ('correct_trial' in df.columns) and\
       ('key_press'in df.columns) and\
       ('correct_response' in df.columns):
        df.rename(columns={'correct_trial': 'original_correcttrial'},
                  inplace=True)
        df['correct_trial'] = np.where(
            df['key_press'] == df['correct_response'],
            1,
            0)
    return df


def filter_to_test_trials(df, exp_id, trial_filter='all'):
    '''
    filter to test_trials (drop practice, ITIs)
    drop first trials with unfair memory-sets for nback
    drop 1st trial for task switching tasks
    '''
    if 'trial_id' in df.columns:
        df = df.query("trial_id=='test_trial'")


    if 'n_back' in exp_id:
        if any(eid in exp_id for eid in ['directed_forgetting', 'task_switching']):
            df = add_memsets(df, with_forgetting=('directed_forgetting' in exp_id))
            df = df[df.delay_condition==df.memory_set.apply(lambda x: len(x))]
        else:
            df = df[df.n_back_condition.notnull()]

    if 'predictable' in exp_id or 'predictive' in exp_id:
        if 'current_trial' in df.columns:
            df = df.query('current_trial > 1')
        else:
            assert 'predictable_condition' in df.columns
            df = df[-((pd.isna(df['predictable_condition'])) |
                    (df['predictable_condition'] == "na"))]
    if 'cue' in exp_id and 'task_switch_condition' in df.columns:
        df = df[-((pd.isna(df['task_switch_condition'])) | (df['task_switch_condition']=="na"))]
    return df.reset_index(drop=True)

def recode_df_iti_resps(data_df):
    data_df = data_df.copy()
    trial_idxs = data_df.query("trial_id=='test_trial'").index
    problem_rows = data_df.iloc[trial_idxs+1].query('rt > 0')
    for bad_row in problem_rows.iterrows():
        iti_idx = bad_row[0]
        bad_row = bad_row[1]
        prev_idx = iti_idx - 1
        if data_df.loc[prev_idx, 'rt']==-1:  # sometimes there the ITI window is catching a second response
            assert data_df.loc[prev_idx, 'key_press']==-1, print(data_df.worker_id.unique()[0], iti_idx)
            if all(
                inhib_reg not in bad_row['exp_id']
                for inhib_reg in ['stop_signal', 'go_nogo']
            ):
                assert data_df.loc[prev_idx, 'correct_trial']==False, print(data_df.worker_id.unique()[0], iti_idx)
            assert data_df.loc[prev_idx, 'rt']==-1, print(data_df.worker_id.unique()[0], iti_idx)
            data_df.loc[prev_idx, 'key_press'] = data_df.loc[iti_idx, 'key_press']
            data_df.loc[prev_idx, 'correct_trial'] = data_df.loc[prev_idx, 'key_press']==data_df.loc[prev_idx, 'correct_response']
            data_df.loc[prev_idx, 'rt'] = data_df.loc[iti_idx, 'rt'] + (data_df.loc[iti_idx, 'time_elapsed'] - data_df.loc[prev_idx, 'time_elapsed'])

    return data_df

def add_inhib_choice_acc(data_df, stim_col='stim', response_col='key_press', old_corr_resp_col='correct_response'):
    data_df = data_df.copy()
    stims = data_df[stim_col].unique()

    data_df['choice_correct_response'] = np.nan
    data_df['choice_accuracy'] = np.nan

    for stim in stims:
        old_corr_resps = data_df.loc[data_df[stim_col] == stim,
                                     old_corr_resp_col].unique()
        old_corr_resps = [i for i in old_corr_resps if int(i) != -1]
        assert len(old_corr_resps) == 1, "\t".join(str(i) for i in old_corr_resps)

        data_df.loc[data_df[stim_col] == stim, 'choice_correct_response'] = old_corr_resps[0]

    data_df['choice_accuracy'] = np.where(
        data_df['choice_correct_response'] == data_df[response_col],
        1,
        0)
    data_df['correct_trial'] = data_df['choice_accuracy']
    return data_df


def add_inhib_acc(stop_df, base):
    if base == 'go_nogo_single_task_network':
        return stop_df
    # add idiosyncratic stim columns
    if 'cued_task_switching' in base:
        stop_df['stim'] = stop_df['stim_number'].astype(str) + stop_df['task']
        stop_df['cue_task_switch'] = 'cue_' + stop_df['cue_condition'] +\
                                     '_task_' + stop_df['task_switch_condition']
    elif 'directed_forgetting' in base:
        stop_df['probe_in_bot'] = [c in l for c, l in zip(stop_df['probe'],
                                                          stop_df['bottom_stim'])]
        stop_df['probe_in_top'] = [c in l for c, l in zip(stop_df['probe'],
                                                          stop_df['top_stim'])]
        stop_df['stim'] = stop_df['probe_in_bot'].astype(str) +\
                          stop_df['probe_in_top'].astype(str) +\
                          stop_df['cue'] +\
                          stop_df['directed_forgetting_condition']
    elif 'predictable_task_switching' in base:
        stop_df['stim'] = stop_df['number'].astype(str) + stop_df['predictable_dimension']
    elif 'shape_matching' in base:
        stop_df['stim'] = stop_df['probe']==stop_df['target']

    acc_stims = {
        'flanker': 'center_letter',
        'n_back': 'n_back_condition',
        }
    return add_inhib_choice_acc(stop_df, stim_col=acc_stims.get(base.split('_with_')[-1], 'stim'))

    
def read_and_filter_df(subj_file, filter_exp='all'):

    base_file = os.path.basename(subj_file) 
    worker_id = subj_file.split('_')[-1].replace('.csv','')
    df = pd.read_csv(subj_file, index_col=0)
    if 'worker_id' not in df.columns:
        df.loc[:,'worker_id'] = worker_id    
    exp_id = get_exp_id(df, base_file)
    
    if 'directed_forgetting' in exp_id:
        df = recode_df_iti_resps(df)
    df = update_correct_trial(df)
    df = df.rename(columns=CONDITION_RENAME_MAP)
    df = filter_to_test_trials(df, exp_id)
    if any(cond in base_file for cond in ['stop', 'go_no']):
        df = add_inhib_acc(df, '_'.join(base_file.split('_')[0:-1]))
    df = standardize_conditions(df, exp_id)
    df = df.drop(DROP_COLS, axis=1, errors='ignore')
    df = df.reset_index(drop=True)
    filter_map = {
        'all': lambda x: x,
        'odd': lambda x: x.iloc[1::2].reset_index(drop=True),
        'even': lambda x: x.iloc[::2].reset_index(drop=True),
        'first_half': lambda x: x.head(int(np.floor(len(x)/2))).reset_index(drop=True),
        'last_half': lambda x: x.tail(int(np.ceil(len(x)/2))).reset_index(drop=True),
    }
    df = filter_map[filter_exp](df)
    return df,exp_id

# 2. For computing RT and ACC DVs
def add_memsets(sub_df, with_forgetting=False):
    sub_df = sub_df.copy().reset_index(drop=True)
    sub_df['memory_set'] = ''
    sub_df['memory_set'] = sub_df['memory_set'].astype('object')
    sub_df['probe_is_target'] = False
    memory_set = []
    for idx, row in sub_df.iterrows():
        # make sure this is a real row and in the same block
        if (idx > 0) and\
           (sub_df.iloc[idx-1].current_block==row.current_block):
            if not(with_forgetting) or (sub_df.iloc[idx-1].directed_forgetting_condition=='remember'):
                memory_set.append(sub_df.iloc[idx-1].probe.lower())
        else:
            memory_set = []
        if len(memory_set) > row.delay_condition:
            memory_set = memory_set[int(-row.delay_condition):]
        sub_df.at[idx, 'memory_set'] = memory_set.copy()
        if (len(memory_set) == row.delay_condition) and memory_set[0] == row.probe.lower():
            sub_df.at[idx, 'probe_is_target'] = True
            
    if with_forgetting:
        forget_presents = []
        for idx, row in sub_df.iterrows():
            forget_present = False
            for shift in range(1, int(row.delay_condition)+1):
                # same conditions above, plus enough items are in the memory set for this to be a fair trial
                if (idx-shift >= 0) and\
                (sub_df.iloc[idx-shift].current_block==row.current_block) and\
                (len(row.memory_set)==row.delay_condition): 
                    forget_present = True if (sub_df.iloc[idx-1].directed_forgetting_condition == 'forget') else forget_present
                else:
                    forget_present = np.nan
                    break
            forget_presents.append(forget_present)
        sub_df['forget_present'] = forget_presents
    
    return sub_df

def add_dv_colnames_by_condition(dv_colnames, condition_name):
    prefixes = ['RT_', 'ACC_', 'DRIFT_', 'THRESH_', 'NDT_']
    dv_colnames += [str(prefix + condition_name) for prefix in prefixes]
    return dv_colnames


def standardize_conditions(data_df, exp_id):
    ''' simplify flanker and shape matching conditions'''
    
    # collapsing across H and F in flanker_condition
    if ('flanker' in exp_id) and (len(data_df['flanker_condition'].unique())>2):
        data_df = data_df.replace(
            {'flanker_condition':[r'(^.*inc.*$)',r'(^.*_con.*$)']},\
            {'flanker_condition':['incongruent','congruent']},
            regex=True)
    # collapsing 7 levels in shape_matching_condition: 
    # noise-response-incongruent (i.e. distractor != target, or D in second letter) 
    # vs.
    # no-noise (i.e. no distractor, or N in second letter);
    # disregarding SSS & DSD
    if 'shape' in exp_id:
        data_df = data_df.replace(
            {'shape_matching_condition':['SDD','DDS','DDD','D!=T', 'mismatch']},\
            {'shape_matching_condition':['DISTRACTOR','DISTRACTOR','DISTRACTOR','DISTRACTOR', 'DISTRACTOR']})
        data_df = data_df.replace(
            {'shape_matching_condition':['SNN','DNN','NoD', 'match']},\
            {'shape_matching_condition':['CONTROL','CONTROL','CONTROL', 'CONTROL']})
        
    # for each person, tag trials by whether there was an intermediate forget trial
    if 'directed_forgetting' in exp_id and 'n_back' in exp_id:
        data_df.loc[data_df.forget_present==True, 'directed_forgetting_condition'] = 'neg'
        data_df.loc[data_df.forget_present==False, 'directed_forgetting_condition'] = 'con'
    
    return data_df


# 4. umbrella function
def make_clean_concat_data(filter_exp='all', stop_subset=False, dataset='discovery', data_paths_file='./raw_data_path.txt'):
    # path setup
    with open(data_paths_file, 'r') as f:
        paths = f.readlines()
    raw_dir = paths[0]
    
    task_dfs = defaultdict(pd.DataFrame)
    all_files = [i for i in glob(raw_dir + 's*/*') if ('demographics' not in i)]
    if dataset == 'discovery':
        files = [i for i in all_files if (i.split('_')[-1].replace('.csv','') in EXPLORE_IDS)]
    elif dataset == 'validation':
        files = [i for i in all_files if (i.split('_')[-1].replace('.csv','') not in EXPLORE_IDS)]
    elif dataset == 'all':
        files = all_files
    if stop_subset:
        files = [i for i in files if ('stop' in i)]
    else:
        files = [i for i in files if ('stop' not in i)]
    for subj_file in files:
        df, exp_id = read_and_filter_df(subj_file, filter_exp=filter_exp)
        task_dfs[exp_id] = pd.concat([task_dfs[exp_id], df], axis=0, sort=True)
    if stop_subset:
        assert len(task_dfs.keys())==8
    else:
        assert len(task_dfs.keys())==36
    return task_dfs
