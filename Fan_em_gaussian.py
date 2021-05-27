#!/usr/bin/env python3
import numpy as np
if not __file__.endswith('_em_gaussian.py'):
    print('ERROR: This file is not named correctly! Please name it as LastName_em_gaussian.py (replacing LastName with your last name)!')
    exit(1)


DATA_PATH = "/u/cs246/data/em/" 
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
    if args.cluster_num:
            
        # initialize lambdas
        lambdas = np.random.random(args.cluster_num)
        lambdas /= lambdas.sum()
        
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
        
    else:
        lambdas = []
        mus = []
        sigmas = []
        with open(args.clusters_file,'r') as f:
            for line in f:
                #each line is a cluster, and looks like this:
                #lambda mu_1 mu_2 sigma_0_0 sigma_0_1 sigma_1_0 sigma_1_1
                lambda_k, mu_k_1, mu_k_2, sigma_k_0_0, sigma_k_0_1, sigma_k_1_0, sigma_k_1_1 = map(float,line.split())
                lambdas.append(lambda_k)
                mus.append([mu_k_1, mu_k_2])
                sigmas.append([[sigma_k_0_0, sigma_k_0_1], [sigma_k_1_0, sigma_k_1_1]])
        lambdas = np.asarray(lambdas)
        mus = np.asarray(mus)
        sigmas = np.asarray(sigmas)
        args.cluster_num = len(lambdas)

    model = (lambdas, mus, sigmas)

    return model

def train_model(model, train_xs, dev_xs, args):
    from scipy.stats import multivariate_normal
    #NOTE: you can use multivariate_normal like this:
    #probability_of_xn_given_mu_and_sigma = multivariate_normal(mean=mu, cov=sigma).pdf(xn)
    
    Z = np.zeros((len(train_xs),args.cluster_num))
    lambdas = model[0]
    mus = model[1]
    sigmas = model[2]
    
    for itr in range(args.iterations):
    	# Expectation step
        for i in range(len(train_xs)):
            sum = 0
            for j in range(args.cluster_num):
                if not args.tied:
                    Z[i][j] = lambdas[j]*multivariate_normal(mean=mus[j], cov=sigmas[j]).pdf(train_xs[i])
                else:
                    Z[i][j] = lambdas[j]*multivariate_normal(mean=mus[j], cov=sigmas).pdf(train_xs[i])
                sum += Z[i][j]
            for j in range(args.cluster_num):
                Z[i][j] = Z[i][j]/sum

        # Maximization step
        for j in range(args.cluster_num):
            sum_0 = 0
            sum_1 = 0
            sum_2 = 0
            sum_3 = 0
            
            for i in range(len(train_xs)):
                sum_0 += Z[i][j]
                sum_1 += Z[i][j]*train_xs[i]
        
            lambdas[j] = sum_0/len(train_xs)
            mus[j] = sum_1/sum_0

            if not args.tied:
                for i in range(len(train_xs)):
                    sum_2 += Z[i][j]*np.outer(train_xs[i]-mus[j],train_xs[i]-mus[j])
                sigmas[j] = sum_2 / sum_0

        if args.tied:
            for i in range(len(train_xs)):
                for j in range(args.cluster_num):
                    sum_3 += Z[i][j]*np.outer(train_xs[i]-mus[j],train_xs[i]-mus[j])
            sigmas = sum_3/len(train_xs)
           
    model = (lambdas, mus, sigmas)
    return model
    

def average_log_likelihood(model, data, args):
    from math import log
    from scipy.stats import multivariate_normal
    
    lambdas = model[0]
    mus = model[1]
    sigmas = model[2]

    ll = 0
    for i in range(len(data)):
        sum = 0
        
        for j in range(args.cluster_num):
            if not args.tied:
                sum += lambdas[j] * multivariate_normal(mean=mus[j], cov=sigmas[j]).pdf(data[i])
            else:
                sum += lambdas[j] * multivariate_normal(mean=mus[j], cov=sigmas).pdf(data[i])
        ll += np.log(sum)
        
    ll = ll/len(data)
        
    return ll

def extract_parameters(model):
    lambdas = model[0]
    mus = model[1]
    sigmas = model[2]
    return lambdas, mus, sigmas

def main():
    import argparse
    import os
    print('Gaussian') #Do not change, and do not print anything before this.
    parser = argparse.ArgumentParser(description='Use EM to fit a set of points.')
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
    ll_train = average_log_likelihood(model, train_xs, args)
    print('Train LL: {}'.format(ll_train))
    if not args.nodev:
        ll_dev = average_log_likelihood(model, dev_xs, args)
        print('Dev LL: {}'.format(ll_dev))
    lambdas, mus, sigmas = extract_parameters(model)
    if args.print_params:
        def intersperse(s):
            return lambda a: s.join(map(str,a))
        print('Lambdas: {}'.format(intersperse(' | ')(np.nditer(lambdas))))
        print('Mus: {}'.format(intersperse(' | ')(map(intersperse(' '),mus))))
        if args.tied:
            print('Sigma: {}'.format(intersperse(' ')(np.nditer(sigmas))))
        else:
            print('Sigmas: {}'.format(intersperse(' | ')(map(intersperse(' '),map(lambda s: np.nditer(s),sigmas)))))

if __name__ == '__main__':
    main()
