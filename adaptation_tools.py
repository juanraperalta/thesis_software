import numpy as np
import pandas as pd
from scipy import optimize as opt
from scipy.stats import norm as normal
import matplotlib.pyplot as plt
from statistics import mean
import scipy.stats as stats

# This module contains the functions that create the environment 



# living population is a pd dataframe with 15 defining variables and 1 observation per individual
# individuals last for one generation and then die after reproduction
# when their cycle is completed, population is recorded in a log

# CONSTANTS
adj_scale = 0.2
plas_scale = 0.2
disp_scale = 0.1

column_names = [
        'patch', 'adjusted' ,'genotype', 'phenotype',                  #3  
        'mismatch', 'fitness', 'id', 'generation', 'mother', 'father', #9
        'disp_p', 'plas_p', 'adj_p',                                   #12
        'disp_e', 'plas_e', 'adj_e', 'lineage', 'multiplier','disperser']          #18

class Population:
    def __init__(self, n, n_loci=50, mismatch_cost=0.2, uniform_patches=False, uniform_alleles=True,
    block_plas = False, block_disp = False, block_adj = False):
        self.amount = n
        self.n_loci = n_loci
        self.mismatch_cost = mismatch_cost
        self.uniform_patches = uniform_patches
        self.uniform_alleles = uniform_alleles
    
        self.generation = 0
        self.individuals = pd.DataFrame(columns=column_names)

        self.individuals['patch'] = 5 * np.random.random_sample(size=self.amount) - 2.5 if self.uniform_patches else np.random.normal(size=self.amount) 
        self.individuals['genotype'] = np.random.randint(2*self.n_loci+1, size=self.amount)

        self.individuals['adjusted'] = self.individuals['patch']
        self.individuals['phenotype'] = [(5/(2*self.n_loci))*g - 2.5 for g in self.individuals['genotype']] # convert genotype into a phenotype in the (-2.5, 2.5) range

        self.block_plas = block_plas
        self.block_adj = block_adj
        self.block_disp = block_disp


        for i in ['disp_p', 'plas_p', 'adj_p']:  
            #self.individuals[i] = np.zeros(self.amount) # This sets things to 0 
            self.individuals[i] = np.random.randint(2*self.n_loci+1, size=self.amount) # uniform distribution
        
        if block_adj: 
            self.individuals['adj_p'] = np.zeros(self.amount) 
        
        if block_disp: 
            self.individuals['disp_p'] = np.zeros(self.amount) 
        
        if block_plas: 
            self.individuals['plas_p'] = np.zeros(self.amount) 

        for i in ['disp_e', 'plas_e', 'adj_e', 'generation', 'mother', 'father', 'disperser']:
            self.individuals[i] = np.zeros(self.amount)  

        self.individuals['id'] = list(range(1,self.amount+1)) # Assign an ordered id value to every row
        self.individuals['lineage'] = self.individuals['id']
        self.individuals['multiplier'] = self.amount*[1]

        compute_mismatch(self.individuals)
        compute_fitness(self.individuals, mismatch_cost=self.mismatch_cost, n_loci = self.n_loci)

    def advance_generation(self, generations=1, save=False, mutation_rate=0, decay_rate=1, pop_limit=10, habitat_change=0,
     dispersal_cost=0.1, plasticity_cost=0.1, adjustment_cost=0.1, 
     sex = False, spatial_autocorrelation=0.5, filename='population.csv'):

        if save:
            self.individuals.to_csv(filename, header=True, index=False)
        
        for g in range(generations):
            pop = self.individuals.values 
            self.generation += 1
            n_loci = self.n_loci
            children_p = np.clip(pop[:,5] * pop[:,17], 0, 1)
            np.nan_to_num(children_p, copy=False, nan=0.0, posinf=1.0, neginf=0.0)
            id_amount = list(np.random.binomial(2, children_p)) # toss 2 coins with a p = fitness * multiplier to determine the number of children
            child_values = np.repeat(pop, id_amount, axis=0)  # repeat pop rows for every child created
            row_number = child_values.shape[0]

            if row_number > 0:
                if sex: 
                    child_values[:,9] = np.random.choice(self.individuals.id.values, row_number)  # select a random father
                    
                    father_data = list(self.individuals[['genotype', 'disp_p', 'plas_p', 'adj_p']].itertuples(index=False, name=None))
                    father_data = dict(zip(self.individuals.id, father_data))   # dictionaire containing info from the fathers

                    child_values[:,2]  = np.clip([mating_result(child_values[r,2], father_data[int(child_values[r,9])][0], n_loci=self.n_loci)  for r in range(row_number)], 0, 2*n_loci) # discrete genotype
                    child_values[:,10] = np.clip([mating_result(child_values[r,10], father_data[int(child_values[r,9])][1], n_loci=self.n_loci)  for r in range(row_number)], 0, 2*n_loci) #dispersal
                    child_values[:,11] = np.clip([mating_result(child_values[r,11], father_data[int(child_values[r,9])][2], n_loci=self.n_loci)  for r in range(row_number)], 0, 2*n_loci) #  plasticity
                    child_values[:,12] = np.clip([mating_result(child_values[r,12], father_data[int(child_values[r,9])][3], n_loci=self.n_loci)  for r in range(row_number)], 0, 2*n_loci) #  adjustment
                
                if mutation_rate !=0:
                    genes = child_values[:, [2, 10, 11, 12]]
                    genes = genes/(2*n_loci)
                    flip = np.random.choice([1,0],size=(row_number, 4), p=(mutation_rate, 1-mutation_rate)) 

                    with np.nditer(genes, op_flags=['readwrite']) as it:
                        for x in it:
                            x[...] = np.random.choice([-1, 1], p=(x, 1-x))

                    genes = genes * flip
                    child_values[:, 10:13] += genes[:,1:4]
                    child_values[:, 10:13] = np.clip(child_values[:, 10:13], 0, 2*n_loci)
                    child_values[:, 2] += genes[:,0]
                    child_values[:,2] = np.clip(child_values[:,2], 0, 2*n_loci)
                    child_values[:,18] = np.zeros(row_number) 
                    

                    if self.block_adj: 
                        child_values[:,12] = np.zeros(row_number) 
        
                    if self.block_disp: 
                        child_values[:,10] = np.zeros(row_number) 
        
                    if self.block_plas: 
                        child_values[:,11] = np.zeros(row_number) 
                    

                if habitat_change != 0:
                    change = np.random.normal(size=row_number)
                    child_values[:,0] = np.average([change,child_values[:,0]], axis=0, weights =[habitat_change, 1-habitat_change]) 

                if decay_rate != 0:
                    child_values[:,1] = np.average([child_values[:,0], child_values[:,1]], axis=0, weights =[decay_rate, 1-decay_rate])


                child_values[:,3] = [(5/(2*n_loci))*g - 2.5 for g in child_values[:,2]]
                child_values[:,8] = child_values[:,6]  # move id to mother id 
                child_values[:,7] += 1 # advance generation number
                child_values[:,13:16] = np.zeros((row_number, 3))  # reset effort to 0

                available_patches = normal.pdf(child_values[:,0], loc=0, scale=1) * pop_limit
                try:
                    density = stats.kde.gaussian_kde(child_values[:,0])
                    occupied_patches = density(child_values[:,0]) * row_number
                except:
                    print('there was an error seting the kde, using a generic')
                    occupied_patches = [0.2*row_number] * row_number

                
                child_values[:,17] = available_patches/occupied_patches # calculate the density-dependent fitness multiplier

            child = pd.DataFrame(child_values, columns= column_names)  # Turn into a data frame
            if child.empty:
                self.individuals = child
                print( 'Population went extinct')
                break

            try:
                if (self.block_plas ==False and self.block_adj==False): 
                    phenotype_adjustment(child, mismatch_cost=self.mismatch_cost, n_loci=self.n_loci)
                    
                elif self.block_adj == False: 
                    adjust_habitat(child, mismatch_cost=self.mismatch_cost, n_loci=self.n_loci)
                    
                elif self.block_plas == False:
                    adapt_phenotype(child, mismatch_cost=self.mismatch_cost, n_loci=self.n_loci)
                    
                else:
                    pass

                if self.block_disp == False:
                    choose_habitat(child, spatial_autocorrelation = spatial_autocorrelation, mismatch_cost=self.mismatch_cost, n_loci=self.n_loci) 
                
            except:
                print('there was an error when calculating strategies. Population likely close to extinction')
                break
            
            compute_mismatch(child)
            compute_fitness(child, plasticity_cost=plasticity_cost, 
            dispersal_cost=dispersal_cost, adjustment_cost=adjustment_cost, 
            mismatch_cost=self.mismatch_cost, n_loci=self.n_loci)

            row_number = child.values.shape[0]
            child = child.sort_values(by='mother')
            child.id = list(range(1,row_number+1))  # add a unique id to each row

            
            self.individuals = child 

            if save:
                with open(filename, 'a') as f:
                    self.individuals.to_csv(f, header=False, index=False)

            print(filename, 'Generation %d' % (g+1))


# LINK FUNCTIONS
# Return the cost of mismatch reduction given a potential

def exponential_link(effort, potential):
    """ 
    [LINK FUNCTION]
    Make cost grow exponentially with reduced mismatch. The exponent is the mismatch reduction 
    multiplied by the potential value. 1 is substracted from the result to adjust for mismatch_reduction = 0 

    Arguments:
    ----------
        mismatch_reduction(float)

        potential(float): Either plasticity or adjustment
    
    Returns:
    --------
        float: The cost of reducing that amount of mismatch given the potential value, expressed in fitness units

    Notes:
    ------
        Mismatch is assumed to relate linearly to fitness: m = 4f
    """
    #effort =np.nan_to_num(effort, copy=False, nan=0.0, posinf=1, neginf=0.0)
    p = 1 - potential
    cost =  np.expm1(effort*p)  # this returns a 0 if potential or mismatch = 0  
    if cost < 0:
        return 0 
    if cost > 1:
        return 1
    return cost

def linear_link(effort, potential):
    """ 
    [LINK FUNCTION]
    Make cost grow linearly with reduced mismatch. The slope is equal to 1 - the potential value

    Arguments:
    ----------
        mismatch_reduction(float)

        potential(float): Either plasticity or adjustment
    
    Returns:
    --------
        float: The cost of reducing that amount of mismatch given the potential value, expressed in fitness units

    Notes:
    ------
        Mismatch is assumed to relate linearly to fitness: m = 4f
    """
    p = 1 - potential
    cost = (effort*p)  # this returns a 0 if potential or mismatch = 0  
    return cost

# MAIN FUNCTIONS
# complex functions that perform the 3 fitness strategies 

def phenotype_adjustment(population, adj_link=exponential_link, plas_link=exponential_link, mismatch_cost=0.2,  n_loci=10):
    """
    Maximize fitness by choosing the optimal amount of plasticity and adjustment given the potential values

    Arguments:
    ----------
        population(DataFrame)

        adj and plas links(function): functions to link strategy cost to reduced mismatch
            Default: exponential_link()

        adj and plas costs(float): the fitness cost of maintaning a potential value of 1
    
    Returns:
    --------
        DataFrame: Same as the input, with adjusted_patch and phenotype values modified.

    Notes:
    ------
        This represents habitat adjustment and plasticity during development.
    """
    
    compute_mismatch(population)
    mismatch = population.mismatch.values
    adj_potential = (0.5/n_loci) * population.adj_p.values
    plas_potential =(0.5/n_loci) * population.plas_p.values

    results = [optimize_fitness(mismatch[i], adj_potential[i], plas_potential[i], adj_link=adj_link, plas_link=plas_link, mismatch_cost=mismatch_cost) for i in range(len(mismatch))]
    adjustment, plasticity, adj_effort, plas_effort = zip(*results)
    
    population.adj_e += adj_effort
    population.plas_e += plas_effort

    adj_direction = np.sign(population.phenotype.values-population.adjusted.values)
    plas_direction = np.sign(population.adjusted.values-population.phenotype.values)

    population.adjusted = population.adjusted.values + adj_direction * adjustment
    population.phenotype = population.phenotype.values + plas_direction * plasticity

    compute_mismatch(population)

def choose_habitat(population, spatial_autocorrelation = 0.5, mismatch_cost=0.2, n_loci=50):

    compute_mismatch(population)
    phenotype = population.phenotype.values
    mismatch = population.mismatch.values
    patch = population.patch.values
    disp_potential = (0.5/n_loci) * population.disp_p.values
    n_row = len(mismatch)
    

    results = [optimize_habitat(mismatch[i], patch[i], phenotype[i], disp_potential[i], 
    sp_acor=spatial_autocorrelation, mismatch_cost=mismatch_cost) if disp_potential[i] != 0 else (patch[i],0) for i in range(n_row)]

    new_habitat = [results[j][0] for j in range(n_row)]
    effort = [results[j][1] for j in range(n_row)]


    population.disp_e += effort
    population.adjusted = [population.adjusted[k] if population.patch[k] == new_habitat[k] else new_habitat[k] for k in range(n_row)]
    population.disperser = [1 if population.patch[k] != new_habitat[k] else 0 for k in range(n_row)]
    population.patch = new_habitat
    compute_mismatch(population)

def adapt_phenotype(population, mismatch_cost=0.2,  n_loci=50):

    compute_mismatch(population)
    mismatch = population.mismatch.values
    plas_potential = (0.5/n_loci) * population.plas_p.values
    n_row = len(mismatch)

    results = [optimize_phenotype(mismatch[i], plas_potential[i], 
    mismatch_cost=mismatch_cost) for i in range(n_row)]
    plasticity, plas_effort = zip(*results)
    
    population.plas_e += plas_effort
    plas_direction = np.sign(population.adjusted.values-population.phenotype.values)
    population.phenotype = population.phenotype.values + plas_direction * plasticity

    compute_mismatch(population)

def adjust_habitat(population, mismatch_cost=0.2,  n_loci=50):

    compute_mismatch(population)
    mismatch = population.mismatch.values
    adj_potential = (0.5/n_loci) * population.adj_p.values
    n_row = len(mismatch)

    results = [optimize_adjustment(mismatch[i], adj_potential[i], 
    mismatch_cost=mismatch_cost) for i in range(n_row)]
    adjustment, adj_effort = zip(*results)
    
    population.adj_e += adj_effort
    adj_direction = np.sign(population.phenotype.values-population.adjusted.values)
    population.phenotype = population.phenotype.values + adj_direction * adjustment

    compute_mismatch(population)

# POPULATION FUNCTIONS 
# Perform key calculations on the whole population

def compute_mismatch(population):
    """
    Compute mismatch of the individuals in a population as the difference between 
    the adjusted patch and the phenotype.
    
    Arguments:
    ----------
        population(DataFrame)
    
    Returns:
    --------
        DataFrame: Same as the input, except with the mismatch values modified.
    """
    population.mismatch = abs(population.adjusted.values - population.phenotype.values)
    
def compute_fitness(population, plasticity_cost=0.1, adjustment_cost=0.1, dispersal_cost=0.1, mismatch_cost=0.2, n_loci=50 ):
    """ 
    Compute fitness of the individuals in a population based on mismatch, potential 
    and effort. 

    Arguments:
    ----------
        population(DataFrame)

        strategy_cost(float): A factor that converts potential values into fitness units. 
    
    Returns:
    --------
        DataFrame: Same as the input, except with the fitness values modified.

    Notes:
    ------
        This functions assumes that fitness units relate linearly to mismatch units
        fitness = 0.25 * mismatch
    """

    population.fitness = 1-(population.mismatch.values * mismatch_cost)  

    population.fitness = (population.fitness.values - 
    (plasticity_cost * (0.5/n_loci) * population.plas_p.values + 
    adjustment_cost * (0.5/n_loci) * population.adj_p.values +
    dispersal_cost * (0.5/n_loci) * population.disp_p.values +
    population.disp_e.values +
    population.plas_e.values + 
    population.adj_e.values)) 

    population.fitness = np.clip(population.fitness, 0, 1)  # keeps fitness in the adequate range

# HELPER FUNCTION
# Perform specific individual calculations 

def mating_result(mother, father, n_loci):
    'compute a plausible offspring genotypic value from two parents, this is generalized to any number of loci'

    mother = float(mother)
    father = float(father)
    m = mean([mother,father])

    sd = 0.5 * ((mother + father) - (mother**2 + father**2)/(2*n_loci))**(.5)

    result = round(np.random.normal(m,sd)+0.5)
    return result

def optimize_fitness(mismatch, adj_potential, plas_potential, mismatch_cost=0.2, adj_link = exponential_link, plas_link = exponential_link):
    """ 
    Maximize fitness given mismatch, potential plasticity and adjustment values and their link function. 

    Arguments:
    ----------
        mismatch(float)

        plas_potential(float)

        asj_potential(float)

        plas_link(function, optional): 
            Default: exponential_link
        
        adj_link(function, optional): 
            Default: exponential_link
    
    Returns:
    --------
        float: The optimal amount of reduced mismatch in mismatch units. 

        float: The cost of that reduction in fitness units.

    Notes:
    ------
        This function is used in the compute phenotype and compute adjustment functions.
        Available link functions are defined above. 
    """

    def fitness_function(params):
        """
        Calculate fitness difference as 0.25 * mismatch reduction minus the cost for both adjustment and plasticity. 

        Arguments:
        ----------
            params(list): two scalars a and p that function as the "x" for adjustment and plasticity respectively 
        
        Returns:
        --------
            float
        
        Notes:
        ------
            This function is taylored to work with the opt.minimize() function
            It returns and inverse value so opt.minimize() will return the maximum fitness increase
        """
        a, p = params
        fitness_recovery = mismatch_cost * (a+p)
        production_cost = (plas_link(p*plas_scale, plas_potential) + adj_link(a*adj_scale, adj_potential))
        y = fitness_recovery - production_cost
        return -1*y 

    b = mean([0, mismatch])
    initial_guess = [b, b]
    bound = opt.Bounds([0,0], [mismatch, mismatch])
    constraint = opt.LinearConstraint([[1,1]], 0, mismatch)
    fitness_result = opt.minimize(fitness_function, initial_guess, constraints=constraint, bounds=bound)

    mismatch_reduction = fitness_result.x
    adj_cost = adj_link(adj_scale*mismatch_reduction[0], adj_potential)
    plas_cost = plas_link(plas_scale*mismatch_reduction[1], plas_potential)

    return (mismatch_reduction[0], mismatch_reduction[1], adj_cost, plas_cost)

def habitat_availability(target_patch, local_patch, sp_acor = 0.5):

    if local_patch == target_patch:
        return 1
    
    if sp_acor == 1: 
        return 0

    sd = (1-sp_acor + sp_acor*(1-sp_acor) * (local_patch)**2)**0.5
    availability= normal(sp_acor*local_patch, sd)
    p = availability.pdf(target_patch)
    if p < 0:
        p = 0 
    if p > 1:
        p = 1
    return p 

def optimize_habitat(mismatch, patch_value, phenotype, disp_potential, disp_link=exponential_link, sp_acor=0.5, mismatch_cost=0.2):

    def habitat_fitness(x):
        fitness_recovery = mismatch_cost * (mismatch - abs(phenotype-x))
        production_cost = disp_link(disp_scale*(1- habitat_availability(x, patch_value, sp_acor)),disp_potential)
        y = fitness_recovery - production_cost   
        return -1*y #negative because it has to be maximized instead of minimized
    
    
    fitness_result = opt.minimize_scalar(habitat_fitness, bounds=[phenotype-mismatch, phenotype+mismatch])

    best_habitat = fitness_result.x
    disp_cost = disp_link(disp_scale*(1- habitat_availability(best_habitat, patch_value, sp_acor)),disp_potential)
    delta_fitness = (mismatch - abs(phenotype-best_habitat))* mismatch_cost - exponential_link(4*(1- habitat_availability(best_habitat, patch_value, sp_acor)),disp_potential)

    if delta_fitness < 0:
        return (patch_value, 0)

    return (best_habitat, disp_cost)

def optimize_phenotype(mismatch, plas_potential, plas_link=exponential_link, mismatch_cost=0.2):
    
    def phenotype_fitness(x):
        fitness_recovery = mismatch_cost * x
        production_cost = plas_link(plas_scale*x, plas_potential)
        y = fitness_recovery - production_cost   
        return -1*y #negative because it has to be maximized instead of minimized
    
    fitness_result = opt.minimize_scalar(phenotype_fitness, bounds=[0, mismatch], method='bounded')

    plasticity_amount = fitness_result.x
    
    plas_cost = plas_link(plas_scale*plasticity_amount, plas_potential)

    return (plasticity_amount, plas_cost)

def optimize_adjustment(mismatch, adj_potential, adj_link=exponential_link, mismatch_cost=0.2):
    
    def adjustment_fitness(x):
        fitness_recovery = mismatch_cost * x
        production_cost = adj_link(adj_scale*x, adj_potential)
        y = fitness_recovery - production_cost   
        return -1*y #negative because it has to be maximized instead of minimized
    
    fitness_result = opt.minimize_scalar(adjustment_fitness, bounds=[0, mismatch], method='bounded')

    adjustment_amount = fitness_result.x
    

    adj_cost = adj_link(adj_scale*adjustment_amount, adj_potential)

    return (adjustment_amount, adj_cost)

# DATA VISUALIZATION

def plot_fitness_tree(filename, every=1, branches=True, color='fitness', show=True, n_loci=50):

    """ 
    NOTE: EVERY RAISES AN ERROR SOMETIMES
    Plot a phylogenetic tree mapping fitness to node color.

    Arguments:
    ----------
        filename(String): Must contain a valid population history data frame

        every(int): generation jump when doing the branches 

        branches(bool): If true, draw the links between nodes
    
    Returns:
    --------
        print a plt figure 
    """

    data=pd.read_csv(filename, usecols=['id','generation', 'genotype', 'patch', 'fitness', 'mother', 'father', 'plas_p', 'adj_p', 'disp_p', 'lineage'])

    if every > 1:
        for idx, row in data.iterrows():
            if row['generation'] in range(every, data.shape[0]+1, every):
                gen = row['generation']
                if row['mother'] != 0:
                    mother_id = row['id']
                    for e in range(every):
                        current_gen = gen-e
                        mother_gen = data[data['generation']==current_gen]
                        new_mother = mother_gen[mother_gen['id']== mother_id].head(1).values[0,5]
                        mother_id = new_mother
                    data.iloc[idx,5] = new_mother
                
    data = data[data.generation % every == 0]               

    l = len(data.fitness)
    col =  list(zip(1-data.fitness, data.fitness, l*[0])) 
    #col =  np.rint((5+data.patch)*10)
    #col = list(zip(data.fitness, data.fitness, data.fitness))


    if color == 'mismatch':
        mismatch = pd.Series([min(1,x) for x in abs(data.genotype - data.patch)/4])
        print(mismatch)
        col = list(zip(1-mismatch, mismatch, l*[0]))
    

    plt.scatter(x=data.generation, y=data.id, c=col)
    if branches:
        for x,item in data.iterrows():
            if item['mother'] > 0:
                plt.plot((item['generation'], item['generation']-every), (item['id'], item['mother']), c=(item['plas_p']*(0.5/n_loci), item['disp_p']*(0.5/n_loci), item['adj_p']*(0.5/n_loci),1), lw=1)
                # red:   phenotypic plasticity
                # green: habitat choice
                # blue:  habitat adjustment
    if show:
        plt.show()


