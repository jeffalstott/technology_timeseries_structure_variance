
# coding: utf-8

# In[2]:

import pandas as pd
import seaborn as sns
import pystan
get_ipython().magic('pylab inline')


# In[3]:

df = pd.read_csv('HPITPdata.csv', index_col=0)

df = df.iloc[2:]


# In[74]:

arma_model_code = """
data { 
int<lower=1> T; 
real y[T];
}

parameters { 
real mu; 
//real phi;
real<lower=-1,upper=1> theta; 
real<lower=0> sigma; 
} 

transformed parameters{
real K;
K <- sqrt( (1+theta^2) * sigma^2 );
}

model {
real err; 
mu ~ normal(0,10); 
theta ~ uniform(-1,1); //normal(0,2); 
sigma ~ cauchy(0,5); 
err <- y[1] - 2*mu; 
err ~ normal(0,sigma); 
for (t in 2:T){ 
    err <- y[t] - (mu + y[t-1] + theta * err); 
    err ~ normal(0,sigma);
    }
}
"""

# model {
# vector[T] nu; // prediction for time t
# vector[T] err; // error for time t
 
# nu[1] <- mu + phi * mu; // assume err[0] == 0 
# err[1] <- y[1] - nu[1]; 
# for (t in 2:T) { 
#     nu[t] <- mu + phi * y[t-1] + theta * err[t-1]; 
#     err[t] <- y[t] - nu[t];
#     } 

# mu ~ normal(0,10); // priors
# phi ~ normal(0,2); 
# theta ~ normal(0,2); 
# sigma ~ cauchy(0,5); 
# err ~ normal(0,sigma); // likelihood
# }


# In[75]:

arma_model = pystan.StanModel(model_code=arma_model_code)


# In[76]:

col = 'DNA Sequencing'
d = df[col].dropna().values.astype('float')
data = {'y':log(d),
   'T':len(d)}
fit = arma_model.sampling(data=data)


# In[77]:

random_walk_parameters = pd.DataFrame(columns=['mu', 'K', 'theta', 'sigma'], dtype='float')

for col in df.columns:
    print(col)
    d = df[col].dropna().values.astype('float')
    data = {'y':log(d)/3,
       'T':len(d)}
    fit = arma_model.sampling(data=data)
    random_walk_parameters.ix[col, 'mu'] = fit['mu'].mean()
    random_walk_parameters.ix[col, 'K'] = fit['K'].mean()
    random_walk_parameters.ix[col, 'theta'] = fit['theta'].mean()
    random_walk_parameters.ix[col, 'sigma'] = fit['sigma'].mean()
# random_walk_parameters = random_walk_parameters.astype('float')


# In[78]:

# K = 0.24
# theta = 0.19
# sqrt(K**2/(1+theta**2))

# sqrt((1+theta**2)*.24**2)


# In[83]:

random_walk_parameters['mu'].hist(bins=50)
plot((0,0), ylim(), 'k--')
title("Mu")
figure()
random_walk_parameters['K'].hist(bins=50)
title("K")
figure()
random_walk_parameters['sigma'].hist(bins=50)
title("sigma")
figure()
random_walk_parameters['theta'].hist(bins=50)
title("theta")


# In[80]:

random_walk_log_parameters = log10(random_walk_parameters)


# In[84]:

random_walk_parameters.plot('mu', 'sigma', kind='scatter')
figure()
random_walk_log_parameters.plot('mu', 'sigma', kind='scatter')


# In[81]:

random_walk_parameters.plot('mu', 'K', kind='scatter')
figure()
random_walk_log_parameters.plot('mu', 'K', kind='scatter')


# In[65]:

from scipy.stats import linregress
linregress(random_walk_parameters['mu'], random_walk_parameters['K'])


# In[31]:

files[0].split('_v1')[0]


# In[16]:

from os import listdir
files = listdir('Rates_11.18.13_PRM/')
for f in files:
    f = pd.read_excel('Rates_11.18.13_PRM/'+f,sheetname='rawdata',index_col=0)
    print(f)

