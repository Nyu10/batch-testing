import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score, recall_score, f1_score
from scipy.stats import ttest_ind
from scipy.optimize import fsolve


def classification(subject, typeII_error, typeI_error):
    """
    a function give the test result given the subject, false positive rate and false negative rate
    
    Input:
        subject(int): 1 stands for postive and 0 stands for negative
        typeII_error(float): the prob to predict negative given the positive tester (false negative)
        typeI_error(float): the prob to predict positive given the negative tester (false positive)
    Output:
        (int): the test result
    """
    if subject == 1:
        # tester is postive
        return int(np.random.uniform(0,1,1) > typeII_error)
    elif subject == 0:
        # tester is negative
        return int(np.random.uniform(0,1,1) < typeI_error)
    else:
        raise('check your data type')

def repeat_classification(subject,typeII_error, typeI_error, repeat):
        """
        A function gives the simultanous test results for a given subject.
        If the simultanous results are all negative, then report the subject as negative.
        Otherwise, report the subject as postive.

        Input:
            subject(int): 1 stands for postive and 0 stands for negative
            typeII_error(float): the prob to predict negative given the positive tester (false negative)
            typeI_error(float): the prob to predict positive given the negative tester (false positive)
            repeat(int): the number of simultaneously pooled sample tests


        Output:
            (int): test result

        """
        subject_repeat = pd.Series(np.repeat(subject, repeat))
        pred_list = subject_repeat.apply(classification, args = (typeII_error, typeI_error))
        if pred_list.sum() >=repeat/2:
            return 1
        else:
            return 0


def batch_test(subject_list, typeII_error = 0, typeI_error = 0, repeat = 1):
    """
    A function gives the test result for a batch of people. There are steps:
    Step 1: collect all people's sample and pool them together
    Step 2: do the test over the pooled sample. If the (repeated) test result is(are) negative, 
            report all testers as negative and quit the function. If the test result is positive
    Step 3: do the test over all the individuals and report the result.
    
    Input:
        subject_list(pandas.Series): a pandas.Series contains all subjects' conditions
        typeII_error(float): the prob to predict negative given the positive tester
        typeI_error(float): the prob to predict positive given the negative tester
        repeat(int): the number of simultaneously pooled sample tests
        
    Output:
        (pandas.Series): a pandas.Series contains the test result for all individuals
        (int): the comsuption of test-kits
    
    """
    batch_size = len(subject_list)

    def repeat_classification(subject,typeII_error, typeI_error, repeat):
        """
        A function gives the simultanous test results for a given subject.
        If the simultanous results are all negative, then report the subject as negative.
        Otherwise, report the subject as postive.

        Input:
            subject(int): 1 stands for postive and 0 stands for negative
            typeII_error(float): the prob to predict negative given the positive tester (false negative)
            typeI_error(float): the prob to predict positive given the negative tester (false positive)
            repeat(int): the number of simultaneously pooled sample tests


        Output:
            (int): test result

        """
        subject_repeat = pd.Series(np.repeat(subject, repeat))
        pred_list = subject_repeat.apply(classification, args = (typeII_error, typeI_error))
        if pred_list.sum() >=repeat/2:
            return 1
        else:
            return 0

        


    if 1 in set(subject_list):
        if repeat_classification(1, typeII_error, typeI_error, repeat)>= 1:
            # Batch_test result is positive. Then do individual test
            return [subject_list.apply(classification, args = (typeII_error, 
                                                            typeI_error)),
                   batch_size + repeat]
        else:
            # Batch_test result is negative. Report all cases as negative
            return [pd.Series(np.zeros(len(subject_list), dtype = int)), repeat]
    else:
        if repeat_classification(0, typeII_error, typeI_error, repeat) >= 1:
            # Batch_test result is positive. Then do individual test
            return [subject_list.apply(classification, args = (typeII_error, 
                                                            typeI_error)),
                   batch_size + repeat]
        else:
            # Batch_test result is negative. Report all cases as negative
            return [pd.Series(np.zeros(len(subject_list), dtype = int)), repeat]



def chunkbatch(subject_list, batch_size):
    """
    a helper function to split a pd.Series object into nearly equal size batchs
    """
    return (subject_list[0+i:batch_size + i] for i in range(0, len(subject_list), batch_size))


def test(subject_list, batch_size, typeII_error = 0, typeI_error = 0, repeat = 1):
    """
    A function gives the test result for people. There are 2 steps:
    Step 1: split all subjects into nearly equal size batchs
    Step 2: do the batch test for each batch and combine all results (test results + 
    consumption of test-kits)
    
    Input:
        subject_list(pandas.Series): a pandas.Series contains all subjects' conditions
        batch_size(int): the predetermined batch size
        typeII_error(float): the prob to predict negative given the positive tester
        typeI_error(float): the prob to predict positive given the negative tester
        repeat(int): the number of simultaneously pooled sample tests

        
    Output:
        (pandas.Series): a pandas.Series contains the test result for all individuals
        (int): the comsuption of test-kits
    
    """
    test_result = pd.Series([], dtype = int)
    test_kit_num = 0
    for i in list(chunkbatch(subject_list, batch_size)):
        temp_result, temp_kit_no = batch_test(i, typeII_error, typeI_error, repeat)
        test_kit_num = test_kit_num + temp_kit_no
        test_result = test_result.append(temp_result)
        
    return[test_result, test_kit_num]



def conventional_test(subject_list, typeII_error = 0, typeI_error = 0, repeat = 1):
    """
    A function gives the conventional test result for people, which means that
    test every subject on the list.
    
    Input:
        subject_list(pandas.Series): a pandas.Series contains all subjects' conditions
        typeII_error(float): the prob to predict negative given the positive tester
        typeI_error(float): the prob to predict positive given the negative tester
        
    Output:
        (pandas.Series): a pandas.Series contains the test result for all individuals
        (int): the comsuption of test-kits
    
    """
    return [subject_list.apply(repeat_classification, args = (typeII_error, 
                                                            typeI_error, repeat)),
                   len(subject_list)*repeat]


def num_test_kit_fixed_batch_size(prevalence_rate, 
                                 batch_size_list = np.arange(2, 21),
                                 typeII_error = 0, typeI_error = 0):
    """
    A function gives the comsumption of testkits given prevalence rate, batch size, type II error and type I error.
    Inputs:
        prevalence_rate(float): infection rate
        batch_size_list(list(int)): batch size list
        typeII_error(float): the prob to predict negative given the positive tester
        typeI_error(float): the prob to predict positive given the negative tester

    Outputs:
        (pandas.DataFrame): a pandas.DataFrame contains the batch size and the corresponding simulation comsumption
        of test-kits


    """

    list_length = len(batch_size_list)
    test_num_list = np.zeros(list_length)
    subject_list = pd.Series(np.random.binomial(size = 2000, n = 1, p = prevalence_rate))
    for i in range(list_length):
        _, test_num_list[i] = test(subject_list, batch_size_list[i])
    df = pd.DataFrame({'batch_size': batch_size_list,
                      'test_num': test_num_list})
    return df


def comparsion(subject_list, batch_size, typeII_error = 0.15, typeI_error = 0, repeat = 1):
    """
    Given the subject list, batch size, type II error, and type I error, this function 
    gives the comparison between batch test and conventional test.

    Inputs:
        subject_list(list(int)): a list of testers
        batch_size(int): the size of batch size
        typeII_error(float): the prob to predict negative given the positive tester
        typeI_error(float): the prob to predict positive given the negative tester
        repeat(int):

    Outputs:
        (pd.Dataframe) a dataframe contains comparison between batch test and conventional
        test:
            Accuracy_Batch: accuracy for batch test
            Precision_Batch: precision for batch test
            Recall_Batch: recall for batch test
            f1_Batch: f1 scores for batch test
            Accuracy_Con: accuracy for conventional test
            Precision_Con: precision for conventional test
            Recall_Con: recall for conventional test
            f1_Con: f1 scores for conventional test
            Test_Num_Difference: the testkit consumption difference between batch test 
                                and conventional test

    """
    batch_pred, batch_num = test(subject_list, batch_size, typeII_error, typeI_error, repeat)
    # Batch_test
    accuracy_batch = np.mean(batch_pred.values == subject_list.values)
    precision_batch = precision_score(y_true = subject_list, y_pred  = batch_pred)
    recall_batch = recall_score(y_true = subject_list, y_pred = batch_pred)
    f1_batch = f1_score(y_true = subject_list, y_pred = batch_pred)
    # Conventional test
    conven_pred, conven_num = conventional_test(subject_list,typeII_error, typeI_error)
    accuracy_con = np.mean(conven_pred.values == subject_list.values)
    precision_con = precision_score(y_true = subject_list, y_pred  = conven_pred)
    recall_con = recall_score(y_true = subject_list, y_pred = conven_pred)
    f1_con = f1_score(y_true = subject_list, y_pred = conven_pred)
    
    df = pd.DataFrame({'Accuracy_Batch': [accuracy_batch],
                       'Precision_Batch': [precision_batch],
                       'Recall_Batch': [recall_batch],
                       'f1_Batch': [f1_batch],
                       'Accuracy_Con': [accuracy_con],
                       'Precision_Con': [precision_con],
                       'Recall_Con': [recall_con],
                       'f1_Con': [f1_con],
                       'Test_Num_Difference': [(batch_num - conven_num)]
                        })
    return df


def repeat_comparison(repeat_time, batch_size, prevalence_rate,n_population = 2000, typeII_error = 0.15,typeI_error = 0, repeat = 1):
    """
    TO DO
    """
    subject_ln = [pd.Series(np.random.binomial(size = n_population, n = 1, p = prevalence_rate)) for _ in range(repeat_time)]
    df_list = [comparsion(subject_list, batch_size, typeII_error, typeI_error, repeat) for subject_list in subject_ln]
    df = pd.concat(df_list)
    return df

def one_batch_test_solver(prevalence_rate,typeII_error, typeI_error,n_initial_guess = 2):
    
    """
    A function gives (float) the best batch size for one batch test given the infection rate
    
    Inputs:
        prevalence_rate(float): infection rate
        typeII_error(float): TO DO
        typeI_error(float):  TO DO
        n_initial_guess(float): the initial guess 

    Output:
        (float): the optimal batch size

    """
    q = 1- prevalence_rate # To consistent with the notation of our document
    func = lambda n : n*q**(n/2) - (-(1-typeII_error - typeI_error)*np.log(q))**(-1/2)
    n_solution = fsolve(func, n_initial_guess)
    
    return float(n_solution)

def one_batch_test_int_solver(prevalence_rate,typeII_error, typeI_error, n_initial_guess = 2):
    """
    A function gives (int) the best batch size for one batch test given the infection rate
    
    Inputs:
        prevalence_rate(float): infection rate
        n_initial_guess(float): the initial guess 
        typeII_error(float): TO DO
        typeI_error(float):  TO DO

    Output:
        (int): the optimal batch size
    """

    
    sol_float = one_batch_test_solver(prevalence_rate,typeII_error, typeI_error, n_initial_guess)
    floor, ceil = np.floor(sol_float), np.ceil(sol_float)
    func = lambda batch_size: 1/batch_size + 1 - typeII_error -(1 - typeII_error - typeI_error)*(1-prevalence_rate)**batch_size
    if func(floor) < func(ceil):
        return int(floor)
    else:
        return int(ceil)
    
def one_batch_test_num(batch_size, prevalence_rate, population = 2000):
    """
    A function gives (float) the expectation of total test kits consumptions for one batch test
    given the batch size, infection rate, and  population

    Inputs:
        batch_size(float) : the batch size
        prevalence_rate(float): the infection rate
        population(float): the population

    Outputs:
        (float) the total test kits consumptions

    """
    q = 1- prevalence_rate # To consistent with the notation of our document

    return float(population * (1/batch_size + 1 - q**batch_size))

def sequential_batch_test_num(batch_size, prevalence_rate , population = 2000):
    """
    A function gives (float) the expectation of total test kits consumptions for sequential batch test
    given the batch size, infection rate, and  population

    Inputs:
        batch_size(float) : the batch size
        prevalence_rate(float): the infection rate
        population(float): the population

    Outputs:
        (float) the total test kits consumptions
    """
    q = 1- prevalence_rate # To consistent with the notation of our document

    return float(population * ((2-q**batch_size)/batch_size + (1 - q**batch_size) ** 2))

def sequential_batch_test_solver(prevalence_rate, n_initial_guess = 2):
    """
    A function gives (float) the best batch size for sequential batch test given the infection rate
    
    Inputs:
        prevalence_rate(float): infection rate
        n_initial_guess(float): the initial guess 

    Output:
        (float): the optimal batch size
    """
    q = 1- prevalence_rate # To consistent with the notation of our document
    func = lambda n: -2/n**2 + 1/n**2 * q ** n - 1/n* np.log(q) * q**n + 2*np.log(q)* q ** (2*n) - 2*np.log(q)*q**(n)
    n_solution = fsolve(func, n_initial_guess)
    return float(n_solution)
    

def infection_rate_on_negative_batch(p,batch_size,typeII_error, typeI_error):
    """
    To DO
    """
    q = 1-p
    r = typeII_error * (1 - q ** batch_size)/((1 - typeI_error) * q ** batch_size + typeII_error *(1 - q**batch_size))
    return p*r/(1-q**batch_size)


def infection_rate_on_positive_batch(p, batch_size, typeII_error, typeI_error):
    
    """
    To DO
    """  

    q = 1-p
    r = (1 - typeII_error) * (1 - q ** batch_size)/(typeI_error * q ** batch_size + (1 - typeII_error) * (1 - q **batch_size))
    return p*r/(1 - q** batch_size)


def neg_pos_batch_split(subject_df, batch_size, typeII_error, typeI_error):
    """
    To Do
    """
    neg_batch = pd.DataFrame([], columns = ['subject'], dtype = int)
    pos_batch = pd.DataFrame([], columns = ['subject'], dtype = int)
    test_consum = len(list(chunkbatch(subject_df, batch_size)))
    for temp_batch in list(chunkbatch(subject_df, batch_size)):
        if 1 in set(temp_batch['subject']):
            if classification(1, typeII_error, typeI_error) == 1:
                pos_batch = pos_batch.append(temp_batch)
            else:
                neg_batch = neg_batch.append(temp_batch)
        else:
            if classification(0, typeII_error, typeI_error) == 0:
                neg_batch = neg_batch.append(temp_batch)
            else:
                pos_batch = pos_batch.append(temp_batch)
    return (neg_batch, pos_batch, test_consum)    


def helpfunction(subject_df, p, batch_size ,typeII_error, typeI_error):
    
    """
    To Do
    """
    temp0, temp1, temp_con = neg_pos_batch_split(subject_df,batch_size,typeII_error, typeI_error)
    p0 = infection_rate_on_negative_batch(p, batch_size, typeII_error, typeI_error)
    p1 = infection_rate_on_positive_batch(p, batch_size, typeII_error, typeI_error)
    n0= one_batch_test_int_solver(p0, typeII_error, typeI_error)
    n1 = one_batch_test_int_solver(p1, typeII_error, typeI_error)
    return(temp0, temp1, temp_con, p0, p1, n0, n1)


def seq_test(subject_df, p, batch_size, typeII_error, typeI_error, repeat = 1, prob_clip = 1):
    """
    """
    temp_list = []
    neg_list = []
    pos_list = []
    consum = 0
    temp = {'data': subject_df,
           'NB_Num': 0,
           'PB_Num': 0,
           'p': p,
           'batch_size': batch_size}
    temp_list.append(temp)
    new_list = []
    neg_df = pd.DataFrame(columns = ['subject'])
    pos_df = pd.DataFrame(columns = ['subject'])
    while len(temp_list) > 0:
        for i in temp_list:
            temp0, temp1, temp_con, p0, p1, n0, n1 = helpfunction(i['data'], i['p'], i['batch_size'],
                                                                            typeII_error = 0.15, typeI_error = 0.01)
            temp0 = {'data': temp0,
                    'NB_Num': i['NB_Num'] + 1,
                    'PB_Num': i['PB_Num'],
                    'p': p0,
                    'batch_size': n0}
            temp1 = {'data': temp1,
                    'NB_Num': i['NB_Num'],
                    'PB_Num': i['PB_Num'] + 1,
                    'p': p1,
                    'batch_size': n1}
            if temp0['NB_Num'] >= 3:
                neg_list.append(temp0)
            else:
                new_list.append(temp0)
            if temp1['PB_Num'] >= 3 or temp1['p'] >= prob_clip:
                pos_list.append(temp1)
            else:
                new_list.append(temp1)
            consum += temp_con 
        temp_list = new_list
        new_list = []
    for i in neg_list:
        neg_df = neg_df.append(i['data'])
    for i in pos_list:
        pos_df = pos_df.append(i['data'])
        
    neg_df['subject'] = 0
    individual_test, individual_con = conventional_test(pos_df['subject'], typeII_error, typeI_error, repeat)
    pos_df['subject'] = individual_test
    consum += individual_con
    result = pd.concat([neg_df, pos_df])
    result.sort_index(inplace = True)
    return (result, consum, individual_con)


def repeat_result(num_repeat,n_pop, p, batch_size, typeII_error, typeI_error, repeat,  prob_clip = 1):
    """
    """
    accuracy_tab = np.zeros(num_repeat)
    precision_tab = np.zeros(num_repeat)
    recall_tab = np.zeros(num_repeat)
    f1_tab = np.zeros(num_repeat)
    total_consump = np.zeros(num_repeat)
    individual_consump = np.zeros(num_repeat)
    for i in range(num_repeat):
        subject_df = pd.DataFrame(np.random.binomial(size = n_pop, n = 1,p = p), columns = ['subject'])
        result, con, temp_con = seq_test(subject_df, p, batch_size, typeII_error, typeI_error, repeat, prob_clip)
        accuracy_tab[i] = np.mean(result == subject_df)
        precision_tab[i] = precision_score(y_true = subject_df, y_pred = result)
        recall_tab[i] = recall_score(y_true = subject_df, y_pred = result)
        f1_tab[i] = f1_score(y_true = subject_df, y_pred = result)
        total_consump[i] = con
        individual_consump[i] = temp_con
    df = pd.DataFrame({'accuracy': accuracy_tab, 
                   'precision': precision_tab, 
                   'recall': recall_tab, 
                   'f1': f1_tab, 
                   'total_consump': total_consump, 
                   'individual_consump': individual_consump})
    return df