import numpy as np
import scipy 
from causal_utils import infer_nonsta_dir
import math

def simulation (timepoints=1000, sig2noise_ratio=2, samplesize=1000):
    
    # generate time points
    T = np.random.uniform(0, 1, timepoints)

    ## Step 2: define P(T) abnd E; generate the distribution of peaks
    # Define the time points function to peaks: P = f(T, E) 
    # P = f(T, E) = T/2 + 0.2
    P_T = T / 2 + 0.2
    E = np.random.normal(0, 0.001, timepoints)
    P = P_T + E 
    ## generate peaks distribution
    # Define the time points function to peaks: P = f(T, E) 
    # P = f(T, E) = T/2 + 0.2
    P = P_T + E 
    trials = 10
    ## make sure the sample size and time points
    Peaks = np.random.binomial(trials, P, samplesize)


    ### estimate the variance of the noise_signal ratio
    peak_var = trials * P *(np.ones(P.shape) - P)
    noise_var = peak_var / sig2noise_ratio

    ##defile Negative Binomial Distribution and generate Gene expression values
    ## We have the coupled case
    ## Define the parameters of the gene expression 
    ## which are the function of the peaks
    
    r = T + np.abs(np.random.normal(0, np.sqrt(noise_var)) + np.abs(np.cos(Peaks))) 
    PR =  P+ np.abs(np.cos(Peaks))
    
    for i  in range(len(PR)):
        p = PR[i]
        if  p >1:
            PR[i] = p-1
    
    genes = np.random.negative_binomial(r, PR, samplesize)


def causal_delta_score(genes, Peaks, T):
    score_ar, _, _ = infer_nonsta_dir(Peaks, genes, T, width=0.1)
    score_ra, _, _ =infer_nonsta_dir(genes, Peaks, T, width=0.1)
    decouple_score = -(score_ar - score_ra)
    couple_score =  score_ar - score_ra 
    return decouple_score, couple_score , score_ar, score_ra




def generate_couple_decouple_pairs_noise(samples=10, size=100, sig2noise_ratio=10):
    P_P = []

    couple_P_G = []
    couple_R_G = []

    decouple_P_G = []
    decouple_R_G = []

    decouple_d =[]
    decouple_c = []

    couple_d = []
    couple_c = []
    PT = []
    DT = []
    CT = []
    TT = []

    dars = []
    dras = []

    cars = []
    cras = []

    low = 0  # lower bound
    high = 1  # upper bound
    trials = 10
    for s in range(samples): 
        print("iteration {} ...".format(s))
        ## Peaks
        T = np.random.uniform(low, high, size)
        TT = np.concatenate([TT, T])
        P = T
        Peaks = np.random.binomial(trials, P, size)
        P_P.append(P)
        PT = np.concatenate([PT, Peaks])

        
        
        ### get the noise variance
        peak_var = trials * P *(np.ones(P.shape) - P)
        
        if sig2noise_ratio == 0:
            noise_var = np.zeros(peak_var.shape)
        else:    
            noise_var = peak_var / sig2noise_ratio


        ## normal sigma
        # sigma = np.sqrt(np.log(noise_var+1))    
        sigma = np.sqrt(noise_var)     
 

        # print("noise_var {}".format(noise_var))        
        ## couple gene
        # r = T / 10 + np.random.lognormal(0, np.sqrt(noise_var)) 
        r = T *10
        # PR =  P + Peaks/trials + np.random.normal(0, np.sqrt(noise_var))/100
        # PR =  P + Peaks/trials + np.random.lognormal(0, np.sqrt(noise_var))/1000
        # PR =  P + Peaks/trials + np.random.lognormal(0, np.sqrt(noise_var))/1000
        # Peaks =  Peaks + np.random.lognormal(1, sigma)

        PR =  P + Peaks/100
        for i  in range(len(PR)):
            p = PR[i]
            if  p >1:
                PR[i] = p-1
        
        couple_genes = np.random.negative_binomial(r, PR, size) 
        
        couple_genes = np.random.negative_binomial(r, PR, size) + np.random.lognormal(1, sigma) 
        couple_R_G.append(r)
        couple_P_G.append(PR)
        CT = np.concatenate([CT, couple_genes])


        # for i  in range(len(PR)):
        #     p = PR[i]
            
        #     if  p >=1:
        #         PR[i] =math.modf(PR[i])[0]
        
        # couple_genes = np.random.negative_binomial(r, PR, size)
        # couple_R_G.append(r)
        # couple_P_G.append(PR)
        CT = np.concatenate([CT, couple_genes])
        # r_d = np.exp(10*T)+np.random.lognormal(0, np.sqrt(noise_var)) 
        r_d = np.exp(10*T) 


        # PR_d =  np.abs(np.cos(np.square(T))+100*Peaks) +np.random.lognormal(0, np.sqrt(noise_var)) / 1000
        # PR_d =  np.abs(np.cos(np.exp(10*T))+Peaks/trials) +np.random.lognormal(0, np.sqrt(noise_var)) / 1000
        PR_d =  np.abs(np.cos(1000*T+100*Peaks))


        decouple_P_G.append(PR_d)
        decouple_R_G.append(r_d)

        for i  in range(len(PR_d)):
            p = PR_d[i]
            if p<= 0:
                PR_d[i] = np.abs(p)
            elif p> 1:
                PR_d[i] =math.modf(PR_d[i])[0]


        # decouple_genes = np.random.negative_binomial(r_d, PR_d, size) / 1e3
        # decouple_genes = np.random.negative_binomial(r_d, PR_d, size) 
        decouple_genes = np.random.negative_binomial(r_d, PR_d, size) + np.random.lognormal(1, sigma) 

        DT = np.concatenate([DT, decouple_genes])

        couple_genes = np.expand_dims(couple_genes,1)
        decouple_genes = np.expand_dims(decouple_genes,1)
        Peaks = np.expand_dims(Peaks, 1)
        Peaks2 = Peaks.copy() 
        T = np.expand_dims(T,1)

        # print(decouple_genes.shape, Peaks.shape, T.shape)
        dds, dcs , dar, dra = causal_delta_score(decouple_genes, Peaks, T)
        decouple_d.append(dds)
        decouple_c.append(dcs)
        dars.append(dar)
        dras.append(dra)


        # print(couple_genes.shape, Peaks.shape, T.shape)

        cds, ccs, car, cra = causal_delta_score(couple_genes, Peaks2, T)
        couple_d.append(cds)
        couple_c.append(ccs)
        cars.append(car)
        cras.append(cra)
        

    return  couple_d, decouple_d, cars, dars, dras, cras, couple_P_G, couple_R_G, decouple_P_G, decouple_R_G, P_P , CT,  DT, TT, PT





