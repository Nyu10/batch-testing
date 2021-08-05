import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score, recall_score, f1_score
from scipy.optimize import fsolve
import numba
from numba import njit,jit
#
@jit(parallel = True)
def conventional_test(subject_array, typeII_error, typeI_error, repeat = 1,
seq = True):


    """
    A function gives the test results to a subject array given the probability of
    type II error, the probability of Type I error, and the number of repeatition,
    and setting of sequence testing or not.
    
    Input:
        subject_array(Numpy Array): an array contains subject id and subject's
        condition (1 stands for infection and 0 stands for uninfection)
        typeII_error (float): probability of type II error 
        typeI_error (float): probability of type I error
        repeat (int): the number of repetition 
        seq (boolean): True stands for sequential testing. The test will end
        when the test result is positive or run up the number of repetition.
        False stands for simutanlous testing with majority voting.

    Output:
        test_result (Numpy Array): an array contains subjects' id and test results
        consum (int): the total test consumption
    """


    # Sequential Testing
    if seq == True:
        consum = 0
        
        test_result = np.zeros(subject_array.shape, dtype = int)
        
        random_table = np.random.uniform(0, 1, (subject_array.shape[0], repeat))
        for i in range(len(subject_array)):
            temp = 0
            j = 0
            subject = subject_array[i,1]
            while j < repeat and temp == 0:
                random_num = random_table[i, j]
                consum += 1
                if subject == 1:
                    temp = 1 if random_num > typeII_error else 0
                else:
                    temp = 1 if random_num < typeI_error else 0
                j += 1
                

            test_result[i,0] = subject_array[i,0]
            test_result[i,1] = temp
        
        return test_result, consum
    
    # Simultanous Testing    
    else:       
        test_result = np.zeros(subject_array.shape, dtype = int)
        

        random_table = np.random.uniform(0, 1, (subject_array.shape[0], repeat))
        for i in range(len(subject_array)):
            temp = 0
            for j in range(repeat):
                temp_random = random_table[i, j]
                if subject_array[i, 1] == 1:
                    temp_1 = 1 if temp_random > typeII_error else 0
                elif subject_array[i, 1] == 0:
                    temp_1 = 1 if temp_random < typeI_error else 0
                temp += temp_1
            temp = 1 if temp >= repeat/2 else 0
            test_result[i,0] = subject_array[i,0]
            test_result[i,1] = temp
      
        return test_result, len(subject_array)*repeat


@njit(parallel = True)
def parallel_test(subject_array, typeII_error, typeI_error, num):
    test_result = np.zeros(subject_array.shape, dtype = int)
    random_table = np.random.uniform(0, 1, (subject_array.shape[0], num))
    for i in range(len(subject_array)):
        subject = subject_array[i, 1]
        if subject == 1:
            temp = 1 if max(random_table[i,:]) > typeII_error else 0
        elif subject == 0:
            temp = 1 if min(random_table[i,:]) < typeI_error else 0

        test_result[i,0] = subject_array[i,0]
        test_result[i,1] = temp

    return test_result,len(subject_array)*num,len(subject_array)*num


def infection_rate_on_negative_batch(p,batch_size,typeII_error, typeI_error):
    """
    
    Given infection rate, batch size, prob of type II error and prob of type I error, this
    function gives the infection rate on the negative batch.
    
    Input:
        p (float): the infection rate
        batch_size (int): the batch size
        typeII_error (float): the prob of type II error
        typeI_error (float): the prob of type I error

    Output:
        (float): the infection rate on the negative batch



    """
    q = 1-p
    r = typeII_error * (1 - q ** batch_size)/((1 - typeI_error) * q ** batch_size + typeII_error *(1 - q**batch_size))
    return p*r/(1-q**batch_size)


def infection_rate_on_positive_batch(p, batch_size, typeII_error, typeI_error):
    
    """
    Given infection rate, batch size, prob of type II error and prob of type I error, this
    function gives the infection rate on the positive batch.
    
    Input:
        p (float): the infection rate
        batch_size (int): the batch size
        typeII_error (float): the prob of type II error
        typeI_error (float): the prob of type I error

    Output:
        (float): the infection rate on the positive batch
    """  

    q = 1-p
    r = (1 - typeII_error) * (1 - q ** batch_size)/(typeI_error * q ** batch_size + (1 - typeII_error) * (1 - q **batch_size))
    return p*r/(1 - q** batch_size)


def one_batch_test_solver(prevalence_rate,typeII_error, typeI_error,n_initial_guess = 2):
    
    """
    A function gives (float) the best batch size for one batch test given the infection rate
    
    Inputs:
        prevalence_rate(float): infection rate
        typeII_error(float): the prob of type II error
        typeI_error(float):  the prob of type I error
        n_initial_guess(float): the initial guess 

    Output:
        (float): the optimal batch size

    """
    q = 1- prevalence_rate # To consistent with the notation of our document
    func = lambda n : n*q**(n/2) - (-(1-typeII_error - typeI_error)*np.log(q))**(-1/2)
    # print(func(n_initial_guess))
    n_solution = fsolve(func, n_initial_guess)
    
    return float(n_solution)

def one_batch_test_int_solver(prevalence_rate,typeII_error, typeI_error,batch_limit,n_initial_guess = 2):
    """
    A function gives (int) the best batch size for one batch test given the infection rate
    
    Inputs:
        prevalence_rate(float): infection rate
        n_initial_guess(float): the initial guess 
        typeII_error(float): the prob of type II error
        typeI_error(float):  the prob of type I error
        n_initial_guess:
        batch_limit (int): the upper limit of batch size

    Output:
        (int): the optimal batch size
    """

    
    sol_float = one_batch_test_solver(prevalence_rate,typeII_error, typeI_error, n_initial_guess)
    floor, ceil = np.floor(sol_float), np.ceil(sol_float)
    func = lambda batch_size: 1/batch_size + 1 - typeII_error -(1 - typeII_error - typeI_error)*(1-prevalence_rate)**batch_size
    if func(floor) < func(ceil):
        temp = int(floor)
    else:
        temp = int(ceil)
    if temp <= batch_limit:
        return temp
    else:
        return int(batch_limit)


def neg_pos_batch_split(subject_array, batch_size, typeII_error, typeI_error):
    """
    A function gives a list of sujects on the negative batch(es),
    a list of subjects on the postive batch(es) and the test-kit 
    consumption given the probability of type II error, the 
    probability of Type I error.
    
    Input:
        subject_array (Numpy Array): an array contains subject id and subject's
        condition (1 stands for infection and 0 stands for uninfection)
        batch_size (int): batch size
        typeII_error (float): probability of type II error 
        typeI_error (float): probability of type I error
        

    Output:
        neg_batch (Numpy Array): an array of subjects on the negative batch(es)
        pos_batch (Numpy Array): an array of subjects on the postive batch(es)
        test_consum (int): the number of test-kit consumptions
        
    """
    neg_batch = []
    pos_batch = []
    test_consum = np.ceil(len(subject_array)/batch_size)
    random_table = np.random.uniform(0, 1, int(test_consum))
    i = 0
    for temp_batch in np.array_split(subject_array, test_consum):
        if 1 in (temp_batch[:,1]):
            if random_table[i] > typeII_error:
                pos_batch.append(temp_batch)
            else:
                neg_batch.append(temp_batch)
        else:
            if random_table[i] > typeI_error:
                neg_batch.append(temp_batch)
            else:
                pos_batch.append(temp_batch)
        i += 1
    neg_batch = np.concatenate(neg_batch) if len(neg_batch) > 0 else np.array([])
    pos_batch = np.concatenate(pos_batch) if len(pos_batch) > 0 else np.array([])
    return (neg_batch, pos_batch, test_consum)

def helpfunction(subject_array, p, batch_size ,typeII_error, typeI_error, batch_limit):
    
    """
    The helpfunction is a handy function to give the list of subjects on the
    negative batch(es), the list of subjects on the postive batch(es), the 
    test-kit consumption, the infection rate on the negative batches, the 
    infection rate on the positive batches, the optimal batch size for
    negative batches and the optimal batch size for positive batches.

    Input: 
        subject_array (Numpy Array): an array contains subject id and subject's
        condition (1 stands for infection and 0 stands for uninfection)
        p (float): Infection rate
        batch_size (int): batch size
        typeII_error (float): probability of type II error 
        typeI_error (float): probability of type I error
        batch_limit (int): batch size upper limit

    Output:
        temp0 (Numpy Array): an array of subjects on the negative batch(es)
        temp1 (Numpy Array): an array of subjects on the postive batch(es)
        temp_con (int): the number of test-kit consumptions
        p0 (float): the infection rate on the negative batches
        p1 (float): the infection rate on the positive batches
        n0 (float): the optimal batch size for the negative batches
        n1 (float): the optimal batch size for the positive batches
    """
    batch_size = min(batch_size, batch_limit)

    p0 = infection_rate_on_negative_batch(p, batch_size, typeII_error, typeI_error)
    p1 = infection_rate_on_positive_batch(p, batch_size, typeII_error, typeI_error)
    n0= one_batch_test_int_solver(p0, typeII_error, typeI_error, batch_limit)
    n1 = one_batch_test_int_solver(p1, typeII_error, typeI_error, batch_limit)
    if subject_array == np.array([]):
        return (np.array([]), np.array([]), p0, p1, n0, n1)
    temp0, temp1, temp_con = neg_pos_batch_split(subject_array,batch_size,typeII_error, typeI_error)
    return(temp0, temp1, temp_con, p0, p1, n0, n1)

def seq_test(subject_array,stop_rule,p, batch_size, typeII_error, typeI_error, repeat = 1, 
prob_threshold = 1, seq = True, batch_limit = 32):
    """
    A function gives the test results to a subject array and the total number of 
    test-kit consumption and the individual testing number given the subject array,
    the stop rule, the batch size, the probability of type II error, the probability of 
    Type I error, and the number of repeatition, the probability threshold, and 
    setting of sequence testing or not.
    
    Input:
        subject_array(Numpy Array): an array contains subject id and subject's
        condition (1 stands for infection and 0 stands for uninfection)
        stop_rule (int): the number of postive batches to enter individual testing
        p (float): infection rate
        batch_size (int): batch size
        typeII_error (float): probability of type II error 
        typeI_error (float): probability of type I error
        repeat (int): the number of repetition 
        prob_threshold (float): if the infection rate of a batch is beyond prob_threshold, 
        the subjects on that batch will enter individual testing phase
        seq (boolean): True stands for sequential testing. The test will end
        when the test result is positive or run up the number of repetition.
        False stands for simutanlous testing with majority voting.
        batch_limit (int):

    Output:
        result (Numpy Array): an array contains subjects' id and test results
        consum (int): the total test consumption
        individual_con (int): the test consumption for individual testings

    """
    temp_list = []
    neg_list = [] #renamed to negativeInfoList
    pos_list = [] #renamed to positiveInfoList
    consum = 0
    temp = {'data': subject_array,
           'NB_Num': 0,
           'PB_Num': 0,
           'p': p,
           'batch_size': batch_size}
    temp_list.append(temp)
    new_list = []
    neg_array = [] #renamed to negativeBatches
    pos_array = [] #renamed to positiveBatches
    while len(temp_list) > 0:
        for i in temp_list:
            temp0, temp1, temp_con, p0, p1, n0, n1 = helpfunction(i['data'], i['p'], i['batch_size'],
                                                                            typeII_error, typeI_error, 
                                                                            batch_limit = batch_limit)
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
            if len(temp0['data']) > 0:
                if temp0['NB_Num'] >= stop_rule:
                    neg_list.append(temp0)
                else:
                    new_list.append(temp0)
            
            if len(temp1['data'])>0:
                if temp1['PB_Num'] >= stop_rule or temp1['p']>=prob_threshold:
                    pos_list.append(temp1)
                else:
                    new_list.append(temp1)
            consum += temp_con 
        temp_list = new_list
        new_list = []
    for j in neg_list:
        neg_array.append(j['data'])
    neg_array = np.concatenate(neg_array)
    for k in pos_list:
        pos_array.append(k['data'])
    pos_array = np.concatenate(pos_array)
        
    neg_array[:,1] = 0
    individual_test, individual_con = conventional_test(pos_array, typeII_error, typeI_error, repeat, seq)
    pos_array = individual_test
    consum += individual_con
    result = np.concatenate((pos_array, neg_array))
    result = result[result[:,0].argsort()]
    result = result.astype('int64')
    return (result, consum, individual_con)

def npv_score(y_true, y_pred):
    """
    A function provides npv given the prediction and the truth 
    """
    tn, _, fn, _ = confusion_matrix(y_true = y_true,
    y_pred = y_pred).ravel()
    return tn/(tn + fn)

def specificity_score(y_true, y_pred):
    """
    A function provides specificty given the prediction and the truth 
    """
    tn, fp, _, _ = confusion_matrix(y_true = y_true,
    y_pred = y_pred).ravel()
    return tn/(tn + fp)

@jit(parallel = True)
def data_gen(size, p):
    """
    data_gen provides a faster way to generate a random population with
    infection rate p.
    Input:
        size (int): the size of population
        p (float): the infection rate
    Output:
        test_array (array): the first column is for id and the second column
        is the condition, where 1 stands for infection and 0 stands for uninfection

    """
    #print(np.random.get_state()[1][0])
    random_table = np.random.binomial(size = size, p = p, n = 1)
    test_array = np.zeros((size, 2), dtype = int)
    for i in range(size):
        test_array[i,0] = i
        test_array[i,1] = random_table[i]
    return test_array


def test_result(data, seq_test, **kwargs):
    """
    a helper function provides convenient results for a given test method with its **kwargs

    Input:
        data (array or list of arrays)
        seq_test (test_method object): could be seq_test, matrix_test and other test_method objects
    Output:
        result (DataFrame): a dataframe contains important evaluation metrics for the test method 
    """
    if isinstance(data, list) == False:
          
        pred,consum, ind_con = seq_test(data, **kwargs)
        result = {'acc': np.mean(pred[:,1] == data[:,1]),
        'sens': recall_score(data[:,1], pred[:,1]),
        'spec': specificity_score(data[:,1], pred[:,1]),
        'PPV': precision_score(data[:, 1], pred[:,1]),
        'NPV': npv_score(data[:, 1], pred[:,1]),
        'test_consum': consum,
        'ind_consum': ind_con,
        'batch_consum': consum - ind_con}
        return result
    else:
        length = len(data)
        acc = np.zeros(length)
        sens = np.zeros(length)
        spec = np.zeros(length)
        ppv = np.zeros(length)
        npv = np.zeros(length)
        test_consum = np.zeros(length)
        ind_consum = np.zeros(length)
        batch_consum = np.zeros(length)
        for i in range(length):
             
            pred,consum, ind_con = seq_test(data[i], **kwargs)
            
            acc[i] = np.mean(pred[:,1] == data[i][:,1])
            sens[i] = recall_score(data[i][:,1], pred[:,1])
            spec[i] = specificity_score(data[i][:,1], pred[:,1])
            ppv[i] = precision_score(data[i][:,1], pred[:,1])
            npv[i] = npv_score(data[i][:,1], pred[:,1])
            test_consum[i] = consum
            ind_consum[i] = ind_con
            batch_consum[i] = consum-ind_con

        result = {'acc': acc,
        'sens': sens,
        'spec': spec,
        'PPV': ppv,
        'NPV': npv,
        'test_consum': test_consum,
        'ind_consum': ind_consum,
        'batch_consum': batch_consum}
        return pd.DataFrame(result)



def matrix_test(subject_array, side_length, typeII_error, typeI_error, sq_repeat = 1 ,ind_repeat = 1, seq = True):

    """
    This function provides the matrix testing results for a given subject array.

   Input:
        subject_array(Numpy Array): an array contains subject id and subject's
        condition (1 stands for infection and 0 stands for uninfection)
        side_length (int): the side length of the matrix testing
        typeII_error (float): probability of type II error 
        typeI_error (float): probability of type I error
        sq_repeat (int): the number of parallel testing for the column/row batch testing
        ind_repeat (int): the number of potential individual testing for the positive crossings
        seq (boolean): True stands for sequential testing. The test will end
        when the test result is positive or run up the number of repetition.
        False stands for simutanlous testing with majority voting.

    Output:
        result (Numpy Array): an array contains subjects' id and test results
        consum (int): the total test consumption
        individual_con (int): the test consumption for individual testings
    """



    matrix_test_num = len(subject_array)//(side_length**2)
    matrix_test_array = subject_array[0:matrix_test_num*side_length**2, :]
    ind_test_array = subject_array[matrix_test_num*side_length**2:, :]
    
    ind_idx = []
    
    for temp_batch in np.array_split(matrix_test_array, matrix_test_num):
        temp_batch = temp_batch.reshape(side_length, side_length, 2)
        temp_row = []
        temp_col = []
        random_num_row = np.random.uniform(0, 1, sq_repeat)
        random_num_col = np.random.uniform(0, 1, sq_repeat)
        for i in range(side_length):
            if 1 in (temp_batch[i,:,1]):
                if max(random_num_row) > typeII_error:
                    temp_row.append(temp_batch[i,:,0])
            else:
                if min(random_num_row) < typeI_error:
                    temp_row.append(temp_batch[i, :, 0])
            if 1 in (temp_batch[:,i,1]):
                if max(random_num_col) > typeII_error:
                    temp_col.append(temp_batch[:,i,0])
            else:
                if min(random_num_col) < typeI_error:
                    temp_col.append(temp_batch[:, i, 0])
        ind_idx.append(np.intersect1d(temp_row, temp_col))

    ind_idx = np.concatenate(ind_idx)
    ind_idx = ind_idx.astype('int')
    
    if len(ind_idx) == 0:
        neg_array = matrix_test_array
    else:
        mask = np.zeros(subject_array.shape[0], dtype = bool)
        mask[ind_idx] = True
        mask[matrix_test_num*side_length**2:] = True
        ind_test_array = subject_array[mask,:]
        
        
        neg_array = subject_array[~mask, :]
        

    
    
    neg_array[:, 1] = 0
    
    ind_test, ind_con = conventional_test(ind_test_array,
    typeII_error, typeI_error, repeat = ind_repeat, seq = seq)
   
    
   
    batch_test_num = matrix_test_num * 2 * side_length * sq_repeat
    result = np.concatenate((neg_array, ind_test))
    result = result[result[:, 0].argsort()]
    
    return (result, batch_test_num + ind_con, ind_con)


def parallel_batch_testing(subject_array, batch_size, typeII_error, typeI_error, parallel_num, ind_repeat, seq):

    """
        This function provides the parallel batch testing results for a given subject array.

        Input:
            subject_array(Numpy Array): an array contains subject id and subject's
            condition (1 stands for infection and 0 stands for uninfection)
            batch_size (int): batch size
            typeII_error (float): probability of type II error 
            typeI_error (float): probability of type I error
            parallel_num (int): the number of parallel testing for the batch testing
            ind_repeat (int): the number of potential individual testing for the positive batches
            seq (boolean): True stands for sequential testing. The test will end
            when the test result is positive or run up the number of repetition.
            False stands for simutanlous testing with majority voting.

        Output:
            result (Numpy Array): an array contains subjects' id and test results
            consum (int): the total test consumption
            individual_con (int): the test consumption for individual testings
    """



    neg_batch = []
    pos_batch = []
    batch_consum = np.ceil(len(subject_array)/batch_size)* parallel_num
    for temp_batch in np.array_split(subject_array, np.ceil(len(subject_array)/batch_size)):
        random_table = np.random.uniform(0, 1, (1, parallel_num))
        if 1 in (temp_batch[:, 1]):
            if random_table.max() > typeII_error:
                pos_batch.append(temp_batch)
            else:
                neg_batch.append(temp_batch)
        else:
            if random_table.min() < typeI_error:
                pos_batch.append(temp_batch)
            else:
                neg_batch.append(temp_batch)
    neg_batch = np.concatenate(neg_batch) if len(neg_batch) > 0 else np.array([])
    pos_batch = np.concatenate(pos_batch) if len(pos_batch) > 0 else np.array([])

    neg_batch[:, 1] = 0
    individual_test, individual_con = conventional_test(pos_batch, typeII_error, typeI_error,
    repeat = ind_repeat, seq = seq)
    result = np.concatenate((individual_test, neg_batch))
    result = result[result[:,0].argsort()]
    result = result.astype('int64')
    return (result, batch_consum+individual_con, individual_con)
            

def fixed_batch_seq_test(subject_array,stop_rule, p, batch_size, typeII_error, typeI_error, repeat, prob_threshold = 0.3, seq = True):
    """
         This function provides the parallel batch testing results for a given subject array.

        Input:
            subject_array(Numpy Array): an array contains subject id and subject's
            condition (1 stands for infection and 0 stands for uninfection)
            stop_rule (int): the number of positive batches to enter the individual testing phase
            batch_size (int): batch size
            typeII_error (float): probability of type II error 
            typeI_error (float): probability of type I error
            repeat (int): the number of potential individual testing for the positive crossings
            prob_threshold (float): if the infection rate of a batch is beyond prob_threshold, 
            the subjects on that batch will enter individual testing phase
            seq (boolean): True stands for sequential testing. The test will end
            when the test result is positive or run up the number of repetition.
            False stands for simutanlous testing with majority voting.

        Output:
        result (Numpy Array): an array contains subjects' id and test results
        consum (int): the total test consumption
        individual_con (int): the test consumption for individual testings
    """
    
    temp_list = []
    neg_list = []
    pos_list = []
    consum = 0
    temp = {'data': subject_array,
           'NB_Num': 0,
           'PB_Num': 0,
           'p': p,
           'batch_size': batch_size}
    temp_list.append(temp)
    new_list = []
    neg_array = []
    pos_array = []
    while len(temp_list) > 0:
        for i in temp_list:
            temp0, temp1, temp_con, p0, p1, n0, n1 = helpfunction(i['data'], i['p'], i['batch_size'],
                                                                            typeII_error, typeI_error)
            temp0 = {'data': np.random.permutation(temp0),
                    'NB_Num': i['NB_Num'] + 1,
                    'PB_Num': i['PB_Num'],
                    'p': p0,
                    'batch_size': batch_size}
            temp1 = {'data': np.random.permutation(temp1),
                    'NB_Num': i['NB_Num'],
                    'PB_Num': i['PB_Num'] + 1,
                    'p': p1,
                    'batch_size': batch_size}
            if len(temp0['data']) > 0:
                if temp0['NB_Num'] >= stop_rule:
                    neg_list.append(temp0)
                else:
                    new_list.append(temp0)
            
            if len(temp1['data'])>0:
                if temp1['PB_Num'] >= stop_rule or temp1['p']>=prob_threshold:
                    pos_list.append(temp1)
                else:
                    new_list.append(temp1)
            consum += temp_con 
        temp_list = new_list
        new_list = []
    for j in neg_list:
        neg_array.append(j['data'])
    neg_array = np.concatenate(neg_array)
    for k in pos_list:
        pos_array.append(k['data'])
    pos_array = np.concatenate(pos_array)
        
    neg_array[:,1] = 0
    individual_test, individual_con = conventional_test(pos_array, typeII_error, typeI_error, repeat, seq)
    pos_array = individual_test
    consum += individual_con
    result = np.concatenate((pos_array, neg_array))
    result = result[result[:,0].argsort()]
    result = result.astype('int64')
    return (result, consum, individual_con)


    
def name_fun(n):
    """
    input: stopping rule
    output: finish nodes
    """
    output = []
    temp = ['']
    for i in range(2*n-1):
        temp_cur = []
        for j in temp:
            candidate_pos = j + '+'
            candidate_neg = j + '-'
            if str.count(candidate_pos, '+') >= n:
                output.append(candidate_pos)
            else:
                temp_cur.append(candidate_pos)

            if str.count(candidate_neg, '-') >= n:
                output.append(candidate_neg)
            else:
                temp_cur.append(candidate_neg)

        temp = temp_cur

        neg_symbol = [x for x in output if str.count(x, '-') == n]
        pos_symbol = [x for x in output if str.count(x, '+') == n]

    return output, neg_symbol, pos_symbol



def seq_test_with_node(subject_array,stop_rule,p, batch_size, typeII_error, typeI_error, repeat = 1, 
prob_threshold = 1, seq = True, batch_limit = 32):
    """
    A function gives the test results to a subject array and the total number of 
    test-kit consumption and the individual testing number given the subject array,
    the stop rule, the batch size, the probability of type II error, the probability of 
    Type I error, and the number of repeatition, the probability threshold, and 
    setting of sequence testing or not.
    
    Input:
        subject_array(Numpy Array): an array contains subject id and subject's
        condition (1 stands for infection and 0 stands for uninfection)
        stop_rule (int): the number of postive batches to enter individual testing
        p (float): infection rate
        batch_size (int): batch size
        typeII_error (float): probability of type II error 
        typeI_error (float): probability of type I error
        repeat (int): the number of repetition 
        prob_threshold (float): if the infection rate of a batch is beyond prob_threshold, 
        the subjects on that batch will enter individual testing phase
        seq (boolean): True stands for sequential testing. The test will end
        when the test result is positive or run up the number of repetition.
        False stands for simutanlous testing with majority voting.
        batch_limit (int):

    Output:
        result (Numpy Array): an array contains subjects' id and test results
        consum (int): the total test consumption
        individual_con (int): the test consumption for individual testings

    """
    temp_list = []
    neg_list = []
    pos_list = []
    batch_num_list = []
    consum = 0
    temp = {'data': subject_array,
           'NB_Num': 0,
           'PB_Num': 0,
           'p': p,
           'batch_size': batch_size,
           'node': ''}
    temp_list.append(temp)
    new_list = []
    neg_array = []
    neg_node = []
    pos_node = []
    pos_array = []
    while len(temp_list) > 0:
        for i in temp_list:
            temp0, temp1, temp_con, p0, p1, n0, n1 = helpfunction(i['data'], i['p'], i['batch_size'],
                                                                            typeII_error, typeI_error, 
                                                                            batch_limit = batch_limit)
            temp0 = {'data': temp0,
                    'NB_Num': i['NB_Num'] + 1,
                    'PB_Num': i['PB_Num'],
                    'p': p0,
                    'batch_size': n0,
                    'node': i['node'] + '-'}
            temp1 = {'data': temp1,
                    'NB_Num': i['NB_Num'],
                    'PB_Num': i['PB_Num'] + 1,
                    'p': p1,
                    'batch_size': n1,
                    'node': i['node'] + '+'}
            if len(temp0['data']) > 0:
                if temp0['NB_Num'] >= stop_rule:
                    neg_list.append(temp0)
                else:
                    new_list.append(temp0)
            
            if len(temp1['data'])>0:
                if temp1['PB_Num'] >= stop_rule or temp1['p']>=prob_threshold:
                    pos_list.append(temp1)
                else:
                    new_list.append(temp1)
            consum += temp_con
        batch_num_list.append(consum) 
        temp_list = new_list
        new_list = []
    for j in neg_list:
        neg_array.append(j['data'])
        temp = [[x, j['node']] for x in j['data'][:,0]]
        neg_node.append(temp)
    neg_array = np.concatenate(neg_array)
    #print(neg_array)
    #print(neg_node)
    #neg_node = np.concatenate(neg_node)

    for k in pos_list:
        pos_array.append(k['data'])
        #pos_node.append(k['node'])
        #pos_node.append(np.column_stack((k['data'][:,0],np.repeat(k['node'], len(k['data'])))))
        temp = [[x, k['node']] for x in k['data'][:,0]]
        pos_node.append(temp)
    pos_array = np.concatenate(pos_array)
    #pos_node = np.concatenate(pos_node)

        
    neg_array[:,1] = 0
    individual_test, individual_con = conventional_test(pos_array, typeII_error, typeI_error, repeat, seq)
    pos_array = individual_test
    consum += individual_con
    result = np.concatenate((pos_array, neg_array))
    #node = np.concatenate((pos_node, neg_node))
    pos_node.extend(neg_node)
    node = pos_node
    node = sum(node, [])
    node.sort()
    node = [x[1] for x in node]
    #node = node[node[:,0].argsort()]
    result = result[result[:,0].argsort()]
    result = result.astype('int64')
    return (result, consum, individual_con, node, batch_num_list)






