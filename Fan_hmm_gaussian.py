#!/usr/bin/env python3
import numpy as np
if not __file__.endswith('_hmm_gaussian.py'):
    print('ERROR: This file is not named correctly! Please name it as Lastname_hmm_gaussian.py (replacing Lastname with your last name)!')
    exit(1)

DATA_PATH = "/u/cs446/data/em/" 
#DATA_PATH = "/Users/fanyangxin/Desktop/Machine_Learning/"

def parse_data(args):
    num = float
    dtype = np.float32
    data = []
    with open(args.data_file, 'r') as f:
        for line in f:
            data.append([num(t) for t in line.split()])
    dev_cutoff = int(.9*len(data))
    train_xs = np.asarray(data[:dev_cutoff],dtype=dtype)
    dev_xs = np.asarray(data[dev_cutoff:],dtype=dtype) if not args.nodev else None
    return train_xs, dev_xs

def init_model(args):
    np.random.seed(42)
    if args.cluster_num:
        mus = np.zeros((args.cluster_num,2))
        if not args.tied:
            sigmas = np.zeros((args.cluster_num,2,2))
        else:
            sigmas = np.zeros((2,2))
        transitions = np.zeros((args.cluster_num,args.cluster_num)) #transitions[i][j] = probability of moving from cluster i to cluster j
        initials = np.zeros(args.cluster_num) #probability for starting in each state
        #TODO: randomly initialize clusters (mus, sigmas, initials, and transitions)

        # initialize initials
        initials = np.random.random(args.cluster_num)
        initials /= initials.sum()
        
        # initialize mus
        mus = np.random.random((args.cluster_num,2))
        
        # initialize sigmas to be random diagonal matrices
        if not args.tied:
            sigmas = np.random.random((args.cluster_num,2,2))
            for i in range(args.cluster_num):
                sigmas[i][0][1] = 0
                sigmas[i][1][0] = 0

        else:
            sigmas = np.random.random((2,2))
            sigmas[0][1] = 0
            sigmas[1][0] = 0

        # initialize transitions
        transitions = np.random.random((args.cluster_num,args.cluster_num))
        for i in range(args.cluster_num):
            weight = transitions[i].sum()
            for j in range(args.cluster_num):
                transitions[i][j] /= weight
    
    else:
        mus = []
        sigmas = []
        transitions = []
        initials = []
        with open(args.clusters_file,'r') as f:
            for line in f:
                #each line is a cluster, and looks like this:
                #initial mu_1 mu_2 sigma_0_0 sigma_0_1 sigma_1_0 sigma_1_1 transition_this_to_0 transition_this_to_1 ... transition_this_to_K-1
                vals = list(map(float,line.split()))
                initials.append(vals[0])
                mus.append(vals[1:3])
                sigmas.append([vals[3:5],vals[5:7]])
                transitions.append(vals[7:])
        initials = np.asarray(initials)
        transitions = np.asarray(transitions)
        mus = np.asarray(mus)
        sigmas = np.asarray(sigmas)
        args.cluster_num = len(initials)

    #TODO: Do whatever you want to pack mus, sigmas, initials, and transitions into the model variable (just a tuple, or a class, etc.)
    model = (mus, sigmas, initials, transitions)
    return model

def forward(model, data, args):
    from scipy.stats import multivariate_normal
    from math import log
    alphas = np.zeros((args.cluster_num,len(data)))
    log_likelihood = 0.0
    #TODO: Calculate and return forward probabilities (normalized at each timestep; see next line) and log_likelihood
    #NOTE: To avoid numerical problems, calculate the sum of alpha[t] at each step, normalize alpha[t] by that value, and increment log_likelihood by the log of the value you normalized by. This will prevent the probabilities from going to 0, and the scaling will be cancelled out in train_model when you normalize (you don't need to do anything different than what's in the notes). This was discussed in class on April 3rd.
    mus = model[0]
    sigmas = model[1]
    initials = model[2]
    transitions = model[3]
    alphas[:, 0] = np.ones(args.cluster_num)

    A = np.zeros((args.cluster_num,args.cluster_num))
    B = np.zeros((args.cluster_num, len(data)))

    for n in range(len(data)):
        for i in range(args.cluster_num):
            if not args.tied:
                B[i][n] = multivariate_normal(mean=mus[i], cov=sigmas[i]).pdf(data[n])
            else:
                B[i][n] = multivariate_normal(mean=mus[i], cov=sigmas).pdf(data[n]) 
        
    for i in range(args.cluster_num):
        for j in range(args.cluster_num):
            A[i][j] = transitions[i][j]
            
    for t in range(len(data)):
        
        if t == 0:
            alphas[:, t] = B[:, t] * initials 
            
        else:
            alphas[:, t] = B[:, t] * np.dot(A.T, alphas[:, t-1])
            
        alpha_sum = np.sum(alphas[:, t])
        alphas[:, t] /= alpha_sum
        log_likelihood = log_likelihood + np.log(alpha_sum)
        
    return alphas, log_likelihood

def backward(model, data, args):
    from scipy.stats import multivariate_normal
    betas = np.zeros((args.cluster_num, len(data)))
    #TODO: Calculate and return backward probabilities (normalized like in forward before)

    betas[:, -1] = np.ones(args.cluster_num)
    mus = model[0]
    sigmas = model[1]
    initials = model[2]
    transitions = model[3]

    A = np.zeros((args.cluster_num,args.cluster_num))
    B = np.zeros((args.cluster_num, len(data)))

    for n in range(len(data)):
        for i in range(args.cluster_num):
            if not args.tied:
                B[i][n] = multivariate_normal(mean=mus[i], cov=sigmas[i]).pdf(data[n])
            else:
                B[i][n] = multivariate_normal(mean=mus[i], cov=sigmas).pdf(data[n]) 
       
        
    for i in range(args.cluster_num):
        for j in range(args.cluster_num):
            A[i][j] = transitions[i][j]
            
    for t in range(len(data) - 1)[::-1]:
        betas[:, t] = np.dot(A, (betas[:, t + 1])*B[:, t + 1])
        betas[:, t] /= np.sum(betas[:, t])

    return betas


def train_model(model, train_xs, dev_xs, args):
    from scipy.stats import multivariate_normal
    #TODO: train the model, respecting args (note that dev_xs is None if args.nodev is True)

    data = train_xs
         
    for itr in range(args.iterations):

        mus = model[0]
        sigmas = model[1]
        initials = model[2]
        transitions = model[3]
        
        # Expectation step
  
        # compute forward probs
        alphas, log_likelihood = forward(model, data, args)

        # compute backward probs
        betas = backward(model, data, args)

        ksi = np.zeros((len(data)-1,args.cluster_num, args.cluster_num))
        gamma = np.zeros((args.cluster_num, len(data)))

        A = np.zeros((args.cluster_num,args.cluster_num))
        B = np.zeros((args.cluster_num, len(data)))

        for n in range(len(data)):
            for i in range(args.cluster_num): 
                if not args.tied:
                    B[i][n] = multivariate_normal(mean=mus[i], cov=sigmas[i]).pdf(data[n])
                else:
                    B[i][n] = multivariate_normal(mean=mus[i], cov=sigmas).pdf(data[n]) 
                
                
        
        for i in range(args.cluster_num):
            for j in range(args.cluster_num):
                A[i][j] = transitions[i][j]

        # compute ksi and gamma  
        for n in range(len(data) - 1):
            sum_2 = 0
            for i in range(args.cluster_num):
                for j in range(args.cluster_num):
                    ksi[n][i][j] = alphas[i][n] * betas[j][n+1] * A[i][j] * B[j][n+1]
                    sum_2 += ksi[n][i][j]
                   
            for i in range(args.cluster_num):
                for j in range(args.cluster_num):
                    ksi[n][i][j]/=sum_2
                    
        for n in range(len(data)):
            sum_1=0
            for i in range(args.cluster_num):
                gamma[i][n] = alphas[i][n]*betas[i][n]
                sum_1 += gamma[i][n]
            
            for i in range(args.cluster_num):
                gamma[i][n] /= sum_1
   
        # Maximization step
        
        # Update A
        sum_ij = np.zeros((args.cluster_num,args.cluster_num))
        sum_i = np.zeros((args.cluster_num))
        for i in range(args.cluster_num):
            sum_11=0
            for j in range(args.cluster_num):
                for n in range(len(data) - 1):
                    sum_11+=ksi[n][i][j]
            sum_i[i]=sum_11

        for i in range(args.cluster_num):
            for j in range(args.cluster_num):
                sum_10=0
                for n in range(len(data) - 1):
                    sum_10+=ksi[n][i][j]
                sum_ij[i][j]=sum_10

        for i in range(args.cluster_num):
            for j in range(args.cluster_num):
                A[i][j]=sum_ij[i][j]/sum_i[i] 
        
        # update mus
        for i in range(args.cluster_num):
            sum_6 = 0
            sum_7 = 0
            for n in range(len(data)):
                sum_6 += gamma[i][n] * train_xs[n]
                sum_7 += gamma[i][n]
            mus[i] = sum_6/sum_7
                      
        # update sigmas
        if not args.tied:
            for i in range(args.cluster_num):
                sum_8 = 0
                sum_9 = 0
                for n in range(len(data)):
                    sum_8 += gamma[i][n] * np.outer(data[n]-mus[i],data[n]-mus[i])
                    sum_9 += gamma[i][n]
                sigmas[i] = sum_8/sum_9

        if args.tied:
            sum = 0
            for i in range(args.cluster_num):
                for n in range(len(data)):
                    sum += gamma[i][n] * np.outer(data[n]-mus[i],data[n]-mus[i])
            sigmas = sum/len(data)
                

        # update initials
        for i in range(args.cluster_num):
            initials[i]=gamma[i][0]

        # update transitions
        for i in range(args.cluster_num):
            sum=0
            for j in range(args.cluster_num):
                transitions[i][j] = A[i][j]
                sum+=transitions[i][j]
                
        model = (mus, sigmas, initials, transitions)
        
    return model

def average_log_likelihood(model, data, args):
    #TODO: implement average LL calculation (log likelihood of the data, divided by the length of the data)
    #NOTE: yes, this is very simple, because you did most of the work in the forward function above
    ll = 0.0
    alphas, log_likelihood = forward(model, data, args)
    ll = log_likelihood/len(data)
    return ll

def extract_parameters(model):
    mus = model[0]
    sigmas = model[1]
    initials = model[2]
    transitions = model[3]
    return initials, transitions, mus, sigmas

def main():
    import argparse
    import os
    print('Gaussian') #Do not change, and do not print anything before this.
    parser = argparse.ArgumentParser(description='Use EM to fit a set of points')
    init_group = parser.add_mutually_exclusive_group(required=True)
    init_group.add_argument('--cluster_num', type=int, help='Randomly initialize this many clusters.')
    init_group.add_argument('--clusters_file', type=str, help='Initialize clusters from this file.')
    parser.add_argument('--nodev', action='store_true', help='If provided, no dev data will be used.')
    parser.add_argument('--data_file', type=str, default=os.path.join(DATA_PATH, 'points.dat'), help='Data file.')
    parser.add_argument('--print_params', action='store_true', help='If provided, learned parameters will also be printed.')
    parser.add_argument('--iterations', type=int, default=1, help='Number of EM iterations to perform')
    parser.add_argument('--tied',action='store_true',help='If provided, use a single covariance matrix for all clusters.')
    args = parser.parse_args()
    if args.tied and args.clusters_file:
        print('You don\'t have to (and should not) implement tied covariances when initializing from a file. Don\'t provide --tied and --clusters_file together.')
        exit(1)

    train_xs, dev_xs = parse_data(args)
    model = init_model(args)
    model = train_model(model, train_xs, dev_xs, args)
    nll_train = average_log_likelihood(model, train_xs, args)
    print('Train LL: {}'.format(nll_train))
    if not args.nodev:
        nll_dev = average_log_likelihood(model, dev_xs, args)
        print('Dev LL: {}'.format(nll_dev))
    initials, transitions, mus, sigmas = extract_parameters(model)
    if args.print_params:
        def intersperse(s):
            return lambda a: s.join(map(str,a))
        print('Initials: {}'.format(intersperse(' | ')(np.nditer(initials))))
        print('Transitions: {}'.format(intersperse(' | ')(map(intersperse(' '),transitions))))
        print('Mus: {}'.format(intersperse(' | ')(map(intersperse(' '),mus))))
        if args.tied:
            print('Sigma: {}'.format(intersperse(' ')(np.nditer(sigmas))))
        else:
            print('Sigmas: {}'.format(intersperse(' | ')(map(intersperse(' '),map(lambda s: np.nditer(s),sigmas)))))

if __name__ == '__main__':
    main()
