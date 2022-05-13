import numpy as np
import pandas as pd
from itertools import product

#np.random.seed(1802) # for the first run
#np.random.seed(1803) # for the second run
np.random.seed(1804) # for the third run

def mating_event(gr21_pref = 0.5, or42_pref=0.5,n=20):
    gr21a_side = np.random.choice(['light', 'dark'], int(n/2), p= (gr21_pref, 1-gr21_pref))
    or42b_side = np.random.choice(['light', 'dark'], int(n/2), p= (or42_pref, 1-or42_pref))

    side = [*gr21a_side, *or42b_side]

    # compute the probability that a random male in each side is gr21a
    gr21_males_light =  gr21_pref/(gr21_pref + or42_pref)
    gr21_males_dark = (1-gr21_pref)/(2-gr21_pref-or42_pref)

    p_gr21a = [gr21_males_light if j == 'light' else gr21_males_dark for j in side]
    chosen_male = [np.random.choice(['gr21a', 'or42b'], p=(k, 1-k)) for k in p_gr21a]

    return(chosen_male)



def replicate (gr21a=1.0, or42b=1.0, male_tb='or42b', atr=True, id=0, remating=0.5, precedence=0.8):

    mother_genotype = [*['gr21a']*10, *['or42b']*10]

    # create the theoretical strength columns
    str_or42b = [or42b] * 20
    str_gr21a = [gr21a] * 20

    # correct the actual strength by ATR
    gr21 = gr21a if atr else 0
    or42 = or42b if atr else 0

    # proportion of flies than go to the light side 
    gr21_pref = 0.5 - gr21/2 
    or42_pref = 0.5 + or42/2

    # which male is selected in each event (first 10 gr21a females, then 10 or42b)
    first_mating = mating_event(gr21_pref, or42_pref)
    second_mating = mating_event(gr21_pref, or42_pref)

    tb_first = [x== male_tb for x in first_mating]
    tb_second = [x== male_tb for x in second_mating]

    # total number of pupae produced by each female
    total = np.clip(np.random.normal(40, 17.7, 20).astype(int), 0, 100)
    
    # number of tb produced in each mating
    remated = np.random.choice([1,0], 20, p=(remating, 1-remating))  # did the female remate?
    

    tb_1 =  np.multiply([np.random.binomial(total, p=0.5)], tb_first)[0]
    tb_2 =  np.multiply([np.random.binomial(total, p=0.5)], tb_second)[0]


    tb_first = [j*(1-precedence) for j in tb_1]  
    tb_second = [tb_2[i]*precedence if remated[i] else tb_1[i]*precedence for i in range(20)]
    
    tb = [int(tb_first[k] + tb_second[k]) for k in range(20)]
    wt = total - tb

    remating = [remating] * 20
    precedence = [precedence]  *20
    male_tb = [male_tb] * 20
    id = [id] * 20
    atr = [atr] * 20

    result = pd.DataFrame(list(zip(id, male_tb, mother_genotype, first_mating, second_mating, atr, tb, wt, total, str_gr21a, str_or42b, remating, remated, precedence)),
    columns =['id', 'male_tb', 'mother_genotype', 'first_male', 'second_male', 'atr', 'tb', 'wt', 'total', 'str_gr21a', 'str_or42b', 'remating_prob', 'remated', 'precedence'])
    
    return result


strengths = [0, 0.25, 0.5, 0.75, 1.0]

#rematings = [0, 0.2, 0.4, 0.6, 0.8, 1.0] # for the first run
#rematings = [0.6, 0.7, 0.8, 0.9, 1.0] # for the second run
rematings = [0.8, 0.85, 0.9, 0.95, 1] # for the third run

#precedences = [0.5, 0.6, 0.7, 0.8, 0.9] # for the first run
#precedences = [0.6, 0.65, 0.7, 0.75, 0.8] # for the second run
precedences = [0.65, 0.675 ,0.7, 0.725 ,0.75] # for the third run

param = list(product(strengths, strengths,[True,False], ['or42b', 'gr21a'], rematings, precedences, repeat=1 ))*100

empty = pd.DataFrame( columns =['id', 'male_tb', 'mother_genotype', 'first_male', 'second_male', 'atr', 'tb', 'wt', 'total', 'str_gr21a', 'str_or42b', 'remating_prob', 'remated', 'precedence'])
empty.to_csv('assortative_mating_simulation_3.csv', header=True, index=False)


for i, item in enumerate(param):
    a = replicate(item[0], item[1], item[3], item[2], id=i, remating=item[4], precedence=item[5])
    with open('assortative_mating_simulation_3.csv', 'a') as file:
                    a.to_csv(file, header=False, index=False)
    print('finished %f percent '% (float(i/300000)*100))


