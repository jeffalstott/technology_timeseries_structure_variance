
# coding: utf-8

# In[135]:

import pandas as pd
import seaborn as sns
get_ipython().magic('pylab inline')


# In[3]:

x = [-0.50,0.00,0.24,0.19,-0.05,0.00,0.02,0.15,-0.02,0.00,0.02,0.04,-0.45,0.00,0.38,0.14,-0.58,0.00,0.32,-0.15,-0.08,0.00,0.05,1.00,-0.10,0.00,0.06,0.46,-0.07,0.00,0.06,0.32,-0.06,0.00,0.04,0.36,-0.07,0.00,0.07,0.91,-0.10,0.00,0.06,0.02,-0.07,0.00,0.05,0.74,-0.10,0.00,0.10,0.61,-0.08,0.00,0.05,-0.22,-0.10,0.00,0.15,0.05,-0.09,0.00,0.08,0.12,-0.08,0.00,0.06,0.33,-0.05,0.00,0.05,0.38,-0.06,0.00,0.05,-0.03,-0.07,0.00,0.08,0.02,-0.08,0.00,0.08,0.88,-0.36,0.00,0.29,0.37,-0.12,0.00,0.10,-0.16,-0.10,0.00,0.08,0.40,-0.04,0.00,0.02,-0.24,-0.06,0.00,0.09,-0.04,-0.10,0.00,0.07,0.26,-0.05,0.00,0.07,0.30,-0.06,0.00,0.06,-0.26,-0.04,0.00,0.05,0.75,-0.10,0.00,0.09,-1.00,-0.84,0.00,0.83,0.26,-0.02,0.00,0.02,0.83,-0.07,0.00,0.06,0.36,-0.03,0.00,0.04,0.85,-0.08,0.00,0.09,-1.00,-0.08,0.01,0.11,1.00,-0.03,0.01,0.05,-1.00,-0.04,0.01,0.09,0.24,-0.07,0.02,0.10,1.00,-0.07,0.02,0.10,0.75,-0.05,0.02,0.09,-0.10,-0.01,0.02,0.02,0.42,-0.08,0.02,0.14,0.29,-0.07,0.03,0.11,0.73,-0.06,0.03,0.09,0.04,-0.02,0.03,0.04,-0.14,-0.08,0.03,0.15,0.31,-0.01,0.03,0.02,-1.00,-0.04,0.04,0.05,-0.41,-0.02,0.06,0.08,0.39,-0.03,0.06,0.05,-1.00,-0.02,0.09,0.04,0.73]
df = pd.DataFrame(array(x).reshape((round(len(x)/4), 4)), columns=['mu', 'p', 'K', 'theta'])
df['-mu'] = df['mu']*-1
logdf = log(df)


# In[4]:

from scipy.stats import linregress

print(linregress(df['mu'], df['K']))
print(linregress(logdf['-mu'], logdf['K']))


# In[5]:

sns.lmplot(x="-mu", y="K", data=df)#, label="Linear Regression and Error")
xlabel(r"$-\mu$")
ylabel(r"$\widetilde{K}$")
plot(xlim(),xlim(), label="Line of Equivalent Scaling", c='g')
legend(loc=4)
xlim(xmin=.00)
ylim(ymin=.00)


# In[6]:

sns.lmplot(x="-mu", y="K", data=df)#, label="Linear Regression and Error")
xlabel(r"$-\mu$")
ylabel(r"$\widetilde{K}$")
plot(sort(df['-mu']),sort(df['-mu']), label="Line of Equivalent Scaling", c='g')
legend(loc=4)
xlim(xmin=.007)
ylim(ymin=.007, ymax=1)
yscale('log')
xscale('log')
yticks([.01, .1, 1],['.01', '.1', '1'])
xticks([.01, .1, 1],['.01', '.1', '1'])


# In[7]:

sns.lmplot(x="-mu", y="K", data=logdf)#, label="Linear Regression and Error")
xlabel(r"$-\mu$")
ylabel(r"$\widetilde{K}$")
plot(xlim(),xlim(), label="Line of Equivalent Scaling", c='g')
legend(loc=4)
# xlim(xmin=.001)
# ylim(ymin=.001)


# In[136]:

### Deterministic model 
p_combination = .25
n_oi = 10
n_steps = 40

sim_df = pd.DataFrame(columns=['n_oi_c'],index=arange(n_steps),data=0.0)
sim_df.ix[0] = n_oi
for t in arange(1,n_steps):
    new_oi = p_combination*n_oi/2
    n_oi += new_oi
    sim_df.ix[t] = n_oi

sim_df.plot()
xlim(xmin=0)
ylabel("# of Operating Ideas")
xlabel("Time Step")
figure()
sim_df.plot()
yscale('log')
xlim(xmin=0)
ylabel("# of Operating Ideas")
xlabel("Time Step")
print(linregress(sim_df.index, log(sim_df['n_oi_c'].values)))


# In[137]:

#### Probabilistic model
p_oi_combination = .25
n_oi_basic = 10
n_steps = 45

oi_df = [array(i) for i in eye(n_oi_basic).tolist()]
n_oi_df = pd.DataFrame(columns=['n_oi_c'],
                       index=arange(n_steps),
                       data=n_oi_basic)

for t in arange(1,n_steps):
    n_to_compare = len(oi_df)
    ois = permutation(n_to_compare)
    for oi_i, oi_j in zip(ois[:round(n_to_compare/2)],ois[round(n_to_compare/2):]): #Pair up operating ideas for possible combination
        combined_oi = oi_df[oi_i]+oi_df[oi_j]
        if (rand()<p_oi_combination 
            and not any([(combined_oi == x).all() for x in oi_df]) # Check that the created combination is new
#             and not any(combined_oi>1) #Check that the two operating ideas don't use the same basic operating ideas
            ):
            oi_df += [combined_oi]
    n_oi_df.ix[t] = len(oi_df) 

n_oi_df.plot()
xlim(xmin=0)
ylabel("# of Operating Ideas")
xlabel("Time Step")
figure()
n_oi_df.plot()
yscale('log')
xlim(xmin=0)
ylabel("# of Operating Ideas")
xlabel("Time Step")
print(linregress(n_oi_df.index, log(n_oi_df['n_oi_c'].values)))


# In[93]:

(n_oi_df.diff()/n_oi_df).plot()


# In[146]:

#### Probabilistic model, repeated use of same basic operating idea not allowed
p_oi_combination = .25
n_oi_basic = 10
n_steps = 50

oi_df = [array(i) for i in eye(n_oi_basic).tolist()]
n_oi_df = pd.DataFrame(columns=['n_oi_c'],
                       index=arange(n_steps),
                       data=n_oi_basic)

for t in arange(1,n_steps):
    n_to_compare = len(oi_df)
    ois = permutation(n_to_compare)
    for oi_i, oi_j in zip(ois[:round(n_to_compare/2)],ois[round(n_to_compare/2):]): #Pair up operating ideas for possible combination
        combined_oi = oi_df[oi_i]+oi_df[oi_j]
        if (rand()<p_oi_combination 
            and not any([(combined_oi == x).all() for x in oi_df]) # Check that the created combination is new
            and not any(combined_oi>1) #Check that the two operating ideas don't use the same basic operating ideas
            ):
            oi_df += [combined_oi]
    n_oi_df.ix[t] = len(oi_df) 

n_oi_df.plot()
xlim(xmin=0)
ylabel("# of Operating Ideas")
xlabel("Time Step")
figure()
n_oi_df.plot()
yscale('log')
xlim(xmin=0)
ylabel("# of Operating Ideas")
xlabel("Time Step")
print(linregress(n_oi_df.index, log(n_oi_df['n_oi_c'].values)))


# In[154]:

#### Probabilistic model, repeated use of same basic operating idea not allowed
p_oi_combination = .25
n_oi_basic = 10
n_oi = n_oi_basic
n_steps = 40

oi_ls = [array(i) for i in eye(n_oi_basic).tolist()]
n_oi_ls = [n_oi]

for t in arange(1,n_steps):
    ois = permutation(n_oi)
    for oi_i, oi_j in zip(ois[:round(n_oi/2)],ois[round(n_oi/2):]): #Pair up operating ideas for possible combination
        combined_oi = oi_ls[oi_i]+oi_ls[oi_j]
        if (rand()<p_oi_combination 
            and not any([(combined_oi == x).all() for x in oi_ls]) # Check that the created combination is new
            and not any(combined_oi>1) #Check that the two operating ideas don't use the same basic operating ideas
            ):
            oi_ls += [combined_oi]
            n_oi +=1
    n_oi_ls.append(n_oi)

plot(n_oi_ls)
xlim(xmin=0)
ylabel("# of Operating Ideas")
xlabel("Time Step")
figure()
plot(n_oi_ls)
yscale('log')
xlim(xmin=0)
ylabel("# of Operating Ideas")
xlabel("Time Step")
print(linregress(arange(len(n_oi_ls)), log(array(n_oi_ls))))


# In[271]:

#### Probabilistic model, repeated use of same basic operating idea not allowed, do N/2 valid combiations
p_oi_combination = .25 
n_oi_basic = 10
n_fields = 10
oi_creation_threshold = 1.5
n_steps = 70 


n_oi = n_oi_basic
field_fitness = rand(n_fields)
FU_previous = sum(field_fitness)

n_possible_combinations = 2**n_oi_basic-1
oi_df = pd.DataFrame(columns=arange(n_oi_basic), index=arange(n_possible_combinations),data=0)
oi_df.iloc[:n_oi_basic] = eye(n_oi_basic)

possibly_valid_combinations = zeros((n_possible_combinations,n_possible_combinations))
possibly_valid_combinations[triu_indices(n_oi_basic,1)] = 1


n_oi_ls = [n_oi]

for t in arange(1,n_steps):
    print(t)
    #Operating Ideas Combining
    n_oi_at_start = n_oi
    n_pairs_to_attempt = round(n_oi_at_start/2)
    n_pairs_attempted = 0
    oi_i = 0
    oi_j = 1
    
    print(n_pairs_to_attempt)
    possible_pairs = list(zip(*where(possibly_valid_combinations)))
    shuffle(possible_pairs)
    print(len(possible_pairs))
    while n_pairs_attempted<n_pairs_to_attempt and possible_pairs!=[]:
        oi_i, oi_j = possible_pairs.pop()
        combined_oi = oi_df.iloc[oi_i]+oi_df.iloc[oi_j]
        if (any(combined_oi>1)
            or  (combined_oi == oi_df).T.all().any()
            ):
            possibly_valid_combinations[oi_i,oi_j] = 0
        else:
            n_pairs_attempted += 1
            if rand()<p_oi_combination:
                    oi_df.iloc[n_oi+1] = combined_oi
                    possibly_valid_combinations[:n_oi,n_oi] = 1
                    n_oi +=1
                    
#     while n_pairs_attempted<n_pairs_to_attempt and oi_i<n_oi_at_start:
#         combined_oi = oi_df.iloc[oi_i]+oi_df.iloc[oi_j]
#         if (not any(combined_oi>1)
#             and not (oi_df == combined_oi).T.all().any()
#             ):
#             n_pairs_attempted += 1
#             if rand()<p_oi_combination:
#                     oi_df.iloc[n_oi+1] = combined_oi
#                     n_oi +=1
#         oi_j+=1
#         if oi_j==n_oi_at_start:
#             oi_j=1
#             oi_i+=1
    n_oi_ls.append(n_oi)
    if n_oi==oi_df.shape[0]:
        break
                    

#     for oi_i in arange(n_oi):
#         for oi_j in arange(oi_i,n_oi):
#             combined_oi = oi_df.iloc[oi_i]+oi_df.iloc[oi_j]
#             if (not any(combined_oi>1)
#                 and not (oi_df == combined_oi).T.all().any()
#                 ):
#                 if rand()<p_oi_combination:
#                     oi_df.iloc[n_oi+1] = combined_oi
#                     n_oi +=1
#                 n_pairs_attempted += 1
#                 if n_pairs_attempted==n_pairs_to_attempt:
#                     break

#     ois = permutation(n_oi)
#     for oi_i, oi_j in zip(ois[:round(n_oi/2)],ois[round(n_oi/2):]): #Pair up operating ideas for possible combination
#         combined_oi = oi_df.iloc[oi_i]+oi_df.iloc[oi_j]
#         if (rand()<p_oi_combination 
#             and not any(combined_oi>1) #Check that the two operating ideas don't use the same basic operating ideas
#             and not (oi_df == combined_oi).T.all().any() # Check that the created combination is new 
#             ):
#             oi_df.iloc[n_oi+1] = combined_oi
#             n_oi +=1
#     n_oi_ls.append(n_oi)



plot(n_oi_ls)
xlim(xmin=0)
ylabel("# of Operating Ideas")
xlabel("Time Step")
figure()
plot(n_oi_ls)
yscale('log')
xlim(xmin=0)
ylabel("# of Operating Ideas")
xlabel("Time Step")
print(linregress(arange(len(n_oi_ls)), log(array(n_oi_ls))))


# In[133]:

#### Probabilistic model, repeated use of same basic operating idea not allowed, with science
p_oi_combination = .25 
n_oi_basic = 5
n_fields = 10
oi_creation_threshold = 1.5
n_steps = 50 


n_oi = n_oi_basic
field_fitness = rand(n_fields)
FU_previous = sum(field_fitness)

oi_df = pd.DataFrame(columns=arange(n_oi_basic), index=arange(2**n_oi_basic-1),data=0)
oi_df.iloc[:n_oi_basic] = eye(n_oi_basic)

n_oi_df = pd.DataFrame(columns=['n_oi_c'],
                       index=arange(n_steps),
                       data=n_oi_basic)

for t in arange(1,n_steps):
    #Operating Ideas Combining
    ois = permutation(n_oi)
    for oi_i, oi_j in zip(ois[:round(n_oi/2)],ois[round(n_oi/2):]): #Pair up operating ideas for possible combination
        combined_oi = oi_df.iloc[oi_i]+oi_df.iloc[oi_j]
        if (rand()<p_oi_combination 
            and not any(combined_oi>1) #Check that the two operating ideas don't use the same basic operating ideas
            and not (oi_df == combined_oi).T.all().any() # Check that the created combination is new 
            ):
            oi_df.iloc[n_oi+1] = combined_oi
            n_oi +=1
    n_oi_df.ix[t] = n_oi
    
    #Scientific Fields combining
    fields = permutation(n_fields)
    for field_i, field_j in zip(fields[:round(n_fields/2)],fields[round(n_fields/2):]):
        fitness_i = field_fitness[field_i]
        fitness_j = field_fitness[field_j]
        new_f = triangular(0,mean([fitness_i,fitness_j]),sum([fitness_i,fitness_j]))
        lowest_f = argmin([new_f,fitness_i,fitness_j])
        if lowest_f==1:
            field_fitness[field_i] = new_f
        elif lowest_f==2:
            field_fitness[field_j] = new_f
    FU = sum(field_fitness)
    
    #Science adds basic operating ideas
    if FU/FU_previous>oi_creation_threshold:
#         print("New basic OI!")
        FU_previous = FU
        oi_df_previous = oi_df
        n_oi_basic+=1
        oi_df = pd.DataFrame(columns=arange(n_oi_basic), index=arange(2**n_oi_basic-1),data=0)
        oi_df.iloc[:n_oi,:n_oi_basic-1] = oi_df_previous
        oi_df.iloc[n_oi,n_oi_basic-1] = 1
        n_oi+=1



n_oi_df.plot()
xlim(xmin=0)
ylabel("# of Operating Ideas")
xlabel("Time Step")
figure()
n_oi_df.plot()
yscale('log')
xlim(xmin=0)
ylabel("# of Operating Ideas")
xlabel("Time Step")
print(linregress(n_oi_df.index, log(n_oi_df['n_oi_c'].values)))


# In[134]:

(n_oi_df.diff()/n_oi_df).plot()


# In[128]:

(n_oi_df.diff()/n_oi_df).plot()

