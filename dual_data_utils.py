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

# setup paths
project_dir = '/Users/henrymj/Documents/r01network/mturk/'
raw_dir = project_dir + 'all_data/raw_data/'

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

def ezdiff(rt,correct,s=.1):
    logit = lambda p:np.log(p/(1-p))

    if len(rt)==0:
        return(None, None, None)
    assert len(rt)==len(correct)

    
    assert np.max(correct)<=1
    assert np.min(correct)>=0
    
    pc=np.mean(correct)
    if pc==0:
        return(None, None, None)
    assert pc > 0
    
    # subtract or add 1/2 an error to prevent division by zero
    if pc==1.0:
        pc=1 - 1/(2*len(correct))
    if pc==0.5:
        pc=0.5 + 1/(2*len(correct))
    MRT=np.mean(rt[correct==1])
    VRT=np.var(rt[correct==1])
    if VRT==0:
        return(None,None,None)
    assert VRT > 0
    
    r=(logit(pc)*(((pc**2) * logit(pc)) - pc*logit(pc) + pc - 0.5))/VRT
    v=np.sign(pc-0.5)*s*(r)**0.25
    a=(s**2 * logit(pc))/v
    y=(-1*v*a)/(s**2)
    MDT=(a/(2*v))*((1-np.exp(y))/(1+np.exp(y)))
    t=MRT-MDT
    
    return(v,a,t)

def calc_ezdiff(dv_df, subj_df, worker_id, condition_name):
    # NOTE - exdiff params are being computed on trials w responses
    rts = subj_df.loc[subj_df['rt'] > -1, 'rt'].values
    corrects = subj_df.loc[subj_df['rt'] > -1, 'correct_trial'].values
    v,a,t = ezdiff(rts,corrects)
    dv_df.loc[worker_id, str('DRIFT_'+condition_name)] = v
    dv_df.loc[worker_id, str('THRESH_'+condition_name)] = a
    dv_df.loc[worker_id, str('NDT_'+condition_name)] = t
    return dv_df

def calc_rt(dv_df, subj_df, worker_id, condition_name):
    '''
    if the sub has no trials, return nan.
    if the condition is a stop/GNG inhibit condition, 
        ignore successful inhibits (rt=-1) and choice accuracy.
    Else, get mean RT or correct trials
    '''
    if len(subj_df[subj_df['correct_trial']==1])==0:
        mean_rt = np.nan
    else:
        if any(cond in condition_name for cond in ['nogo', 'stop']):
            mean_rt = subj_df.loc[subj_df['rt'] > -1, 'rt'].mean()
        else:
            mean_rt = subj_df.loc[subj_df['correct_trial']==1, 'rt'].mean()
    dv_df.loc[worker_id, 'RT_'+str(condition_name)] = mean_rt
    return dv_df

def calc_acc(dv_df, subj_df, worker_id, condition_name):
    '''
    if the sub has no trials, return nan.
    else, get the mean choice accuracy for trials with a response.
    Note: this will ignore successful inhibits and focus on inhibit-failure responses for
        stop and GNG. This is good and what we want.
    '''
    accuracy = np.nan
    if len(subj_df) != 0:
            accuracy = subj_df.loc[subj_df['rt'] > -1, 'correct_trial'].mean()
    dv_df.loc[worker_id, str('ACC_'+condition_name)] = accuracy
    
    return dv_df


def get_subj_dvs(dv_df, subj_df, worker_id, exp_id, condition_name, conditions):
    dv_df = calc_rt(dv_df, subj_df, worker_id, condition_name)
    dv_df = calc_acc(dv_df, subj_df, worker_id, condition_name)
    dv_df = calc_ezdiff(dv_df, subj_df, worker_id, condition_name)
    return dv_df

def drop_ignored_dvs(dv_df):
    dv_df.drop([i for i in dv_df.columns if (('TSWITCH:switch' in i) & ('CUE:stay' in i))| (('TSWITCH:stay' in i) & ('CUE:switch' in i))], axis=1, inplace=True, errors='ignore')
    dv_df.drop([i for i in dv_df.columns if 'DF:pos' in i], axis=1, inplace=True, errors='ignore')
    dv_df.drop([i for i in dv_df.columns if 'DELAY:3.0' in i], axis=1, inplace=True, errors='ignore')
    dv_df.drop([i for i in dv_df.columns if ('SSS' in i) | ('DSD' in i)], axis=1, inplace=True, errors='ignore')
    dv_df.drop([i for i in dv_df.columns if ('countGoTrials' in i) & ('SS:stop' in i)], axis=1, inplace=True, errors='ignore')
    
    return dv_df

def get_DVs(concat_df, exp_id, manipulations):
    dv_colnames = []
    if len(manipulations)==1:
        manipulation0_conditions = concat_df[manipulations[0]].unique()
        manipulation0_conditions.sort()

        for condition in manipulation0_conditions:
            condition_name = str(ABBREV[manipulations[0]] + ':' + str(condition))
            dv_colnames = add_dv_colnames_by_condition(dv_colnames, condition_name)
            
        dv_df = pd.DataFrame(index=concat_df.worker_id.unique(),columns=dv_colnames)    
        for condition in manipulation0_conditions:
            condition_name = str(ABBREV[manipulations[0]] + ':' + str(condition))
            for worker_id in dv_df.index:        
                subj_df = concat_df[(concat_df['worker_id']==worker_id) & (concat_df[manipulations[0]]==condition)]
                dv_df = get_subj_dvs(dv_df, subj_df, worker_id, exp_id, condition_name, [condition])


    elif len(manipulations) == 2:
        manipulation0_conditions = concat_df[manipulations[0]].unique()
        manipulation0_conditions.sort()
        manipulation1_conditions = concat_df[manipulations[1]].unique()
        manipulation1_conditions.sort()
        
        for condition0 in manipulation0_conditions:
            for condition1 in manipulation1_conditions:
                # e.g. TASK:stay_&_GNG:go
                condition_name = str(ABBREV[manipulations[0]] + ':' + str(condition0) + '_&_' + ABBREV[manipulations[1]] + ':' + str(condition1))
                dv_colnames = add_dv_colnames_by_condition(dv_colnames, condition_name)
                
        dv_df = pd.DataFrame(index=concat_df.worker_id.unique(),columns=dv_colnames)
        for condition0 in manipulation0_conditions:
            for condition1 in manipulation1_conditions:
                condition_name = str(ABBREV[manipulations[0]] + ':' + str(condition0) + '_&_' + ABBREV[manipulations[1]] + ':' + str(condition1))
                for worker_id in dv_df.index:        
                    subj_df = concat_df[(concat_df['worker_id']==worker_id) & (concat_df[manipulations[0]]==condition0) & (concat_df[manipulations[1]]==condition1)]
                    dv_df = get_subj_dvs(dv_df, subj_df, worker_id, exp_id, condition_name, [condition0, condition1])
                    
    elif len(manipulations) == 3:
        manipulation0_conditions = concat_df[manipulations[0]].unique()
        manipulation0_conditions.sort()
        manipulation1_conditions = concat_df[manipulations[1]].unique()
        manipulation1_conditions.sort()
        manipulation2_conditions = concat_df[manipulations[2]].unique()
        manipulation2_conditions.sort()

        for condition0 in manipulation0_conditions:
            for condition1 in manipulation1_conditions:
                for condition2 in manipulation2_conditions:
                    condition_name = str(ABBREV[manipulations[0]] + ':' + str(condition0) + '_&_' + ABBREV[manipulations[1]] + ':' + str(condition1) + '_&_' + ABBREV[manipulations[2]] + ':' + str(condition2))
                    dv_colnames = add_dv_colnames_by_condition(dv_colnames, condition_name)

        dv_df = pd.DataFrame(index=concat_df.worker_id.unique(),columns=dv_colnames)
        for condition0 in manipulation0_conditions:
            for condition1 in manipulation1_conditions:
                for condition2 in manipulation2_conditions:
                    condition_name = str(ABBREV[manipulations[0]] + ':' + str(condition0) + '_&_' + ABBREV[manipulations[1]] + ':' + str(condition1) + '_&_' + ABBREV[manipulations[2]] + ':' + str(condition2))
                    for worker_id in dv_df.index:        
                        subj_df = concat_df[(concat_df['worker_id']==worker_id) & (concat_df[manipulations[0]]==condition0) & (concat_df[manipulations[1]]==condition1) & (concat_df[manipulations[2]]==condition2)]
                        dv_df = get_subj_dvs(dv_df, subj_df, worker_id, exp_id, condition_name, [condition0, condition1, condition2])

    elif len(manipulations) == 4:
        manipulation0_conditions = concat_df[manipulations[0]].unique()
        manipulation0_conditions.sort()
        manipulation1_conditions = concat_df[manipulations[1]].unique()
        manipulation1_conditions.sort()
        manipulation2_conditions = concat_df[manipulations[2]].unique()
        manipulation2_conditions.sort()
        manipulation3_conditions = concat_df[manipulations[3]].unique()
        manipulation3_conditions.sort()

        for condition0 in manipulation0_conditions:
            for condition1 in manipulation1_conditions:
                for condition2 in manipulation2_conditions:
                    for condition3 in manipulation3_conditions:
                        condition_name = str(ABBREV[manipulations[0]] + ':' + str(condition0) + '_&_' + ABBREV[manipulations[1]] + ':' + str(condition1) + '_&_' + ABBREV[manipulations[2]] + ':' + str(condition2) + '_&_' + ABBREV[manipulations[3]] + ':' + str(condition3))
                        dv_colnames = add_dv_colnames_by_condition(dv_colnames, condition_name)
            
        dv_df = pd.DataFrame(index=concat_df.worker_id.unique(),columns=dv_colnames)
        for condition0 in manipulation0_conditions:
            for condition1 in manipulation1_conditions:
                for condition2 in manipulation2_conditions:
                    for condition3 in manipulation3_conditions:
                        condition_name = str(ABBREV[manipulations[0]] + ':' + str(condition0) + '_&_' + ABBREV[manipulations[1]] + ':' + str(condition1) + '_&_' + ABBREV[manipulations[2]] + ':' + str(condition2) + '_&_' + ABBREV[manipulations[3]] + ':' + str(condition3))
                        for worker_id in dv_df.index:        
                            subj_df = concat_df[(concat_df['worker_id']==worker_id) & (concat_df[manipulations[0]]==condition0) & (concat_df[manipulations[1]]==condition1) & (concat_df[manipulations[2]]==condition2)]
                            dv_df = get_subj_dvs(dv_df, subj_df, worker_id, exp_id, condition_name, [condition0, condition1, condition2, condition3])
    return dv_df

# 3. Helpers for Umbrella Functions
def get_col_regex(col, key):
    if 'cued' in col:
        return EASY_MAP[key] if 'stay' in col else HARD_MAP[key]
    else:
        return col.split('_')[-1]


def make_single_v_meanedDual_score_df(df, key):
    df = df.copy()
    score_df = pd.DataFrame()
    for col in df.filter(regex='single').columns:
        regex = get_col_regex(col, key)
        score_df[regex+'_single'] = df[col]
        score_df[regex+'_collapsed_dual'] = df.filter(regex='with.*'+regex).mean(1)
        for difficulty, cond_map in [('easy', EASY_MAP), ('hard', HARD_MAP)]:
            dual_conditions_regex = '|'.join(
                val for _, val in cond_map.items() if val!=regex
                )
            score_df[regex+'_%s_dual' % difficulty] = df.filter(
                regex='with.*'+regex
                ).filter(
                    regex=dual_conditions_regex
                    ).mean(1)
    return score_df

def make_unfurled_tmp_df(df, key, separate_conditions_dict):
    tmp_df = pd.DataFrame()
    #build up temp df for 
    for col in df.filter(regex='single').columns:
        regex = get_col_regex(col, key)
        
        tmp_df[regex+'_single'] = df[col]
            
        for dual_task in [k for k in separate_conditions_dict.keys() if k!= key]:
            dual_df = df.filter(regex='with.*'+regex).filter(regex=dual_task)
            assert dual_df.shape[1] == 2, print(col, regex, dual_task, dual_df.columns) # hard and easy condition
            tmp_df[regex+'_dual-'+dual_task+'_collapsed'] = dual_df.mean(1)
            tmp_df[regex+'_dual-'+dual_task+'_easy'] = dual_df.filter(regex=EASY_MAP[dual_task])
            tmp_df[regex+'_dual-'+dual_task+'_hard'] = dual_df.filter(regex=HARD_MAP[dual_task])
    return tmp_df

# 4. umbrella functions
def make_clean_concat_data(filter_exp='all', stop_subset=False, dataset='explore'):
    sub_list = EXPLORE_IDS if dataset=='explore' else []
    task_dfs = defaultdict(pd.DataFrame)
    explore_files = [i for i in glob(raw_dir + 's*/*') if (i.split('_')[-1].replace('.csv','') in sub_list) and ('demographics' not in i)]
    if stop_subset:
        explore_files = [i for i in explore_files if ('stop' in i)]
    else:
        explore_files = [i for i in explore_files if ('stop' not in i)]
    for subj_file in explore_files:
        df, exp_id = read_and_filter_df(subj_file, filter_exp=filter_exp)
        task_dfs[exp_id] = pd.concat([task_dfs[exp_id], df], axis=0, sort=True)
    if stop_subset:
        assert len(task_dfs.keys())==8
    else:
        assert len(task_dfs.keys())==36
    return task_dfs

def make_dv_dict(task_dfs):
    dv_dict = {}
    for exp_id, concat_df in task_dfs.items():
        manipulations = [i for i in concat_df.columns if ('condition' in i) and ('n_back_condition' not in i)]  # filtering out match/mismatch for n-back
        manipulations.sort()
        dv_df = get_DVs(concat_df, exp_id, manipulations)
        dv_df = drop_ignored_dvs(dv_df)

        if ('flanker_with_cued_task_switching' in exp_id) or ('shape_matching_with_cued_task_switching' in exp_id):
            dv_df.columns = dv_df.columns.str.replace(r'switch_new','switch')
        if 'no_go' in exp_id: exp_id = exp_id.replace('no_go','nogo')

        dv_df.columns = dv_df.columns.str.replace('\.0','', regex=True)
        dv_dict[exp_id] = dv_df
    return dv_dict


def make_jeanette_df(separate_conditions_dict):
    full_jeanette_DV_df = pd.DataFrame()

    for key, df in separate_conditions_dict.items():
        df = df.copy()    
        easy_reg = EASY_MAP[key]
        hard_reg = HARD_MAP[key]
        tmp_df = make_unfurled_tmp_df(df, key, separate_conditions_dict)
        
        task_jeanette_df = pd.DataFrame()
        task_jeanette_df['easy'] = tmp_df[easy_reg+'_single'].astype(float)
        task_jeanette_df['hard'] = tmp_df[hard_reg+'_single'].astype(float)
        task_jeanette_df['dual_condition'] = 'single'
        task_jeanette_df['dual_difficulty'] = None
        task_jeanette_df.index.name = 'subject'
        task_jeanette_df = task_jeanette_df.reset_index()

        for dual_task in [k for k in separate_conditions_dict.keys() if k!= key]:
            for difficulty in ['easy', 'hard', 'collapsed']:

                dualtask_jeanette_df = pd.DataFrame()
                dualtask_jeanette_df['easy'] = tmp_df[easy_reg+'_dual-'+dual_task+'_%s' % difficulty].astype(float)
                dualtask_jeanette_df['hard'] = tmp_df[hard_reg+'_dual-'+dual_task+'_%s' % difficulty].astype(float)
                dualtask_jeanette_df['dual_condition'] = dual_task
                dualtask_jeanette_df['dual_difficulty'] = difficulty
                dualtask_jeanette_df.index.name = 'subject'
                dualtask_jeanette_df = dualtask_jeanette_df.reset_index()
                task_jeanette_df = pd.concat([task_jeanette_df, dualtask_jeanette_df], 0)

        task_jeanette_df.insert(0, 'primary_task', key)      
        full_jeanette_DV_df = pd.concat([full_jeanette_DV_df, task_jeanette_df], 0)
    return full_jeanette_DV_df


def make_separate_conditions_dict(dv_dict, dv_regex):
    single_tasks = [k for k in dv_dict.keys() if 'single' in k]
    dual_tasks = [k for k in dv_dict.keys() if 'with' in k]
    # put single tasks first
    ordered_tasks = [item for sublist in [single_tasks, dual_tasks] for item in sublist]
    separate_conditions_dict = defaultdict(pd.DataFrame)
    for task in ordered_tasks:
        if INCLUDE_INHIB_TASKS | (('stop' not in task) & ('go_nogo' not in task)):
            df_df = dv_dict[task].filter(regex=dv_regex).add_prefix(task+'_')
            task_keys = task.split('_single_')[:1] if (task in single_tasks) else task.split('_with_')
            for task_key in task_keys:
                separate_conditions_dict[task_key] = pd.concat([separate_conditions_dict[task_key], df_df], axis=1, sort=True)
    # Even if they've been included as dual conditions, we don't need the inhibition as main effects
    if INCLUDE_INHIB_TASKS:
        _ = separate_conditions_dict.pop('stop_signal')
        _ = separate_conditions_dict.pop('go_nogo')
    assert 'stop_signal' not in separate_conditions_dict.keys()
    assert 'go_nogo' not in separate_conditions_dict.keys()
    return separate_conditions_dict

def make_each_combination_dv_df(separate_conditions_dict):
    full_combination_DV_df = pd.DataFrame()

    for key, df in separate_conditions_dict.items():
        df = df.copy()    
        easy_reg = EASY_MAP[key]
        hard_reg = HARD_MAP[key]
        tmp_df = make_unfurled_tmp_df(df, key, separate_conditions_dict)
        # build up the output df
        for dual_task in [k for k in separate_conditions_dict.keys() if k!= key]:
            for difficulty in ['easy', 'hard', 'collapsed']:
                combination_DV_df = pd.DataFrame()
                # slowing: easy in dual - easy in single
                combination_DV_df['slowing'] = (tmp_df[easy_reg+'_dual-'+dual_task+'_%s' % difficulty] - tmp_df[easy_reg+'_single']).astype(float)
                
                # effect: hard - easy in dual
                combination_DV_df['effect'] = (
                    tmp_df[hard_reg+'_dual-'+dual_task+'_%s' % difficulty] - tmp_df[easy_reg+'_dual-'+dual_task+'_%s' % difficulty]
                ).astype(float)

                # efffect delta: effect in dual (hard - easy) - effect in single (hard - easy)
                combination_DV_df['effect_delta'] = (
                    (tmp_df[hard_reg+'_dual-'+dual_task+'_%s' % difficulty] - tmp_df[easy_reg+'_dual-'+dual_task+'_%s' % difficulty])  - \
                    (tmp_df[hard_reg+'_single'] - tmp_df[easy_reg+'_single'])
                ).astype(float)
                
                combination_DV_df['slowing_rank'] = combination_DV_df['slowing'].rank()
                combination_DV_df['effect_rank'] = combination_DV_df['effect'].rank()
                combination_DV_df['effect_delta_rank'] = combination_DV_df['effect_delta'].rank()
                for col in ['slowing', 'effect', 'effect_delta']:
                    col_zscore = col + '_zscore'
                    combination_DV_df[col_zscore] = (combination_DV_df[col] - combination_DV_df[col].mean())/combination_DV_df[col].std(ddof=0)
                combination_DV_df.index.name = 'subject'
                combination_DV_df = combination_DV_df.reset_index()
                combination_DV_df.insert(0, 'dual_task_difficulty', difficulty)
                combination_DV_df.insert(0, 'dual_task', dual_task)
                combination_DV_df.insert(0, 'primary_task', key)
                
                full_combination_DV_df = pd.concat([full_combination_DV_df, combination_DV_df], 0)
    return full_combination_DV_df

def make_single_effect_df(separate_conditions_dict):
    single_effect_df = pd.DataFrame()

    for key, df in separate_conditions_dict.items():
        df = df.copy()    
        easy_reg = EASY_MAP[key]
        hard_reg = HARD_MAP[key]
        tmp_df = make_unfurled_tmp_df(df, key, separate_conditions_dict)
        single_DV_df = pd.DataFrame()
        single_DV_df[key] = (
            tmp_df[hard_reg+'_single'] - tmp_df[easy_reg+'_single']
        ).astype(float)

        single_effect_df = pd.concat([single_effect_df, single_DV_df], 1, sort=True)
    
    return single_effect_df

def get_slowing_and_effect_dfs(separate_conditions_dict):
    concat_separate_conditions = pd.DataFrame()
    slowing_vs_effectDelta = pd.DataFrame()
    for key,df in separate_conditions_dict.items():
        easy_reg = EASY_MAP[key]
        hard_reg = HARD_MAP[key]
        score_df = make_single_v_meanedDual_score_df(df, key)
        if key not in ['stop_signal', 'go_nogo']:
            slow_effect_df = pd.DataFrame()
            for difficulty in ['easy', 'hard', 'collapsed']:
                slow_effect_df['%s_%s_slowing' % (key, difficulty)] = score_df[easy_reg+'_%s_dual' % difficulty] - score_df[easy_reg+'_single']
                slow_effect_df['%s_%s_effect_delta' % (key, difficulty)] = (score_df[hard_reg+'_%s_dual' % difficulty] - score_df[easy_reg+'_%s_dual' % difficulty])  - \
                                                      (score_df[hard_reg+'_single'] - score_df[easy_reg+'_single'])

            concat_separate_conditions = pd.concat([concat_separate_conditions, score_df], axis=1, sort=True)
            slowing_vs_effectDelta = pd.concat([slowing_vs_effectDelta, slow_effect_df], axis=1, sort=True)
            for col in slowing_vs_effectDelta.columns:
                slowing_vs_effectDelta[col] = slowing_vs_effectDelta[col].astype(float)
    return(slowing_vs_effectDelta, concat_separate_conditions)

def dual_data_pipeline(split='full', filter_exp='all'):
    out_dict = {}
    task_dfs = make_clean_concat_data(filter_exp=filter_exp)
    dv_dict = make_dv_dict(task_dfs)
    for dv_regex in ['RT', 'ACC', 'DRIFT', 'THRESH', 'NDT']:
        out_dict[dv_regex] = {
            'separate_conditions_dict': make_separate_conditions_dict(dv_dict, dv_regex)
        }
        out_dict[dv_regex]['jeanette_df'] = make_jeanette_df(
            out_dict[dv_regex]['separate_conditions_dict']
        )
        # out_dict[dv_regex]['full_combination_DV_df'] = make_each_combination_dv_df(
        #     out_dict[dv_regex]['separate_conditions_dict']
        # )
        # out_dict[dv_regex]['single_effect_df'] = make_single_effect_df(
        #     out_dict[dv_regex]['separate_conditions_dict']
        # )
        # slowing_vs_effectDelta, concat_separate_conditions = get_slowing_and_effect_dfs(
        #     out_dict[dv_regex]['separate_conditions_dict']
        # )
        # out_dict[dv_regex]['slowing_vs_effectDelta'] = slowing_vs_effectDelta
        # out_dict[dv_regex]['concat_separate_conditions'] = concat_separate_conditions

    return out_dict
