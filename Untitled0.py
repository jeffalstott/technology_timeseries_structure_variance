
# coding: utf-8

# In[1]:

from statsmodels import discrete


# In[2]:

from discrete.dcm_clogit import CLogit, CLogitResults


# In[1]:

import pandas as pd
import seaborn as sns
get_ipython().magic('pylab inline')


# In[2]:

x = [0.50,0.00,0.24,0.19,-0.05,0.00,0.02,0.15,-0.02,0.00,0.02,0.04,-0.45,0.00,0.38,0.14,-0.58,0.00,0.32,-0.15,-0.08,0.00,0.05,1.00,-0.10,0.00,0.06,0.46,-0.07,0.00,0.06,0.32,-0.06,0.00,0.04,0.36,-0.07,0.00,0.07,0.91,-0.10,0.00,0.06,0.02,-0.07,0.00,0.05,0.74,-0.10,0.00,0.10,0.61,-0.08,0.00,0.05,-0.22,-0.10,0.00,0.15,0.05,-0.09,0.00,0.08,0.12,-0.08,0.00,0.06,0.33,-0.05,0.00,0.05,0.38,-0.06,0.00,0.05,-0.03,-0.07,0.00,0.08,0.02,-0.08,0.00,0.08,0.88,-0.36,0.00,0.29,0.37,-0.12,0.00,0.10,-0.16,-0.10,0.00,0.08,0.40,-0.04,0.00,0.02,-0.24,-0.06,0.00,0.09,-0.04,-0.10,0.00,0.07,0.26,-0.05,0.00,0.07,0.30,-0.06,0.00,0.06,-0.26,-0.04,0.00,0.05,0.75,-0.10,0.00,0.09,-1.00,-0.84,0.00,0.83,0.26,-0.02,0.00,0.02,0.83,-0.07,0.00,0.06,0.36,-0.03,0.00,0.04,0.85,-0.08,0.00,0.09,-1.00,-0.08,0.01,0.11,1.00,-0.03,0.01,0.05,-1.00,-0.04,0.01,0.09,0.24,-0.07,0.02,0.10,1.00,-0.07,0.02,0.10,0.75,-0.05,0.02,0.09,-0.10,-0.01,0.02,0.02,0.42,-0.08,0.02,0.14,0.29,-0.07,0.03,0.11,0.73,-0.06,0.03,0.09,0.04,-0.02,0.03,0.04,-0.14,-0.08,0.03,0.15,0.31,-0.01,0.03,0.02,-1.00,-0.04,0.04,0.05,-0.41,-0.02,0.06,0.08,0.39,-0.03,0.06,0.05,-1.00,-0.02,0.09,0.04,0.73]


# In[14]:

df = pd.DataFrame(array(x).reshape((round(len(x)/4), 4)), columns=['mu', 'p', 'K', 'theta'])


# In[65]:

logdf = log(df)


# In[71]:

#df.plot('mu', 'K', kind='scatter')
sns.lmplot("mu", "K", data=df, label="Linear Regression and Error")

# xscale('log')
# yscale('log')

xlabel(r"$-\mu$")
ylabel(r"$\widetilde{K}$")
plot(xlim(),xlim(), label="Line of Equivalent Scaling", c='g')
legend(loc=4)
# xlim(xmin=.007)
# ylim(ymin=.007)


# In[72]:

#df.plot('mu', 'K', kind='scatter')
sns.lmplot("mu", "K", data=logdf, label="Linear Regression and Error")

# xscale('log')
# yscale('log')

xlabel(r"$log(-\mu$)")
ylabel(r"$log(\widetilde{K})$")
plot(xlim(),xlim(), label="Line of Equivalent Scaling", c='g')
legend(loc=4)
# xlim(xmin=.007)
# ylim(ymin=.007)


# In[78]:

df.iloc[where(logdf.mu<-4)[0]]


# In[55]:

sns.lmplot("mu", "K", data=df)
#xscale('log')
#yscale('log')
#plot(xlim(),ylim())


# In[12]:

tech = pd.DataFrame()
n_rows = 10000
tech['A'] = rand(n_rows)
tech['d'] = rand(n_rows)
tech['k'] = rand(n_rows)
tech['K'] = tech['k']*tech['A']/tech['d']
tech.corr()


# In[8]:

properties.plot('k', 'K', kind='scatter')


# In[2]:

get_ipython().magic('load_ext rmagic')


# In[2]:

import numpy as np
import pylab
X = np.array([0,1,2,3,4])
Y = np.array([3,5,4,6,7])
pylab.scatter(X, Y)


# In[4]:

get_ipython().run_cell_magic('R', '-i X,Y -o XYcoef', 'XYlm = lm(Y~X)\nXYcoef = coef(XYlm)\nprint(summary(XYlm))\npar(mfrow=c(2,2))\nplot(XYlm)')


# In[2]:

from pylab import *
get_ipython().magic('matplotlib inline')


# In[5]:

data = zeros(100000).astype('bool')
baseline_rate = .1
data[:round(len(data)*(baseline_rate))] = True

TP = []
FP = []
for prediction_threshold in arange(.01, 1, .01):
    predictions = rand(len(data)) < prediction_threshold

    true_positives = sum(predictions*data)
    false_positives = sum(predictions*~data)
    true_negatives = sum(~predictions*~data)
    false_negatives = sum(~predictions*data)

    TP.append(true_positives/sum(data))
    FP.append(false_positives/sum(~data))

scatter(TP, FP)
plot((0,1), (0, 1), 'k')
xlim(0,1)
ylim(0,1)
ylabel("True Positive Rate\nWhat Portion of True Data are Labeled True")
xlabel("False Positive Rate\nWhat Portion of False Data are Labeled True")


# In[70]:

import networkx as nx
def sample_bottlenecks_and_variance(n_nodes=100,
                                    p=.1,
                                    n_iterations=100):
    max_d_mins = empty(n_iterations)
    var_d_mins = empty(n_iterations)

    for i in range(n_iterations):
        g = nx.erdos_renyi_graph(n_nodes, p, directed=True)
        d_mins = []
        for n in g.nodes():
            try:
                d_min = min(map(lambda x: g.out_degree(x), g.predecessors(n)))
                d_mins.append(d_min)
            except ValueError:
                continue
        max_d_mins[i] = max(d_mins)
        var_d_mins[i] = var(d_mins)
    return max_d_mins, var_d_mins


# In[4]:

import statsmodels.api as st


# In[80]:

import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
plots = PdfPages('Random_Network_Bottlenecking_Figures.pdf')

for n_nodes in [100, 1000, 10000]:
    for p in [.01, .05, .1]:
        print(n_nodes, p)
        max_d_mins, var_d_mins = sample_bottlenecks_and_variance(n_nodes=n_nodes, p=p, n_iterations=1000)
        figure()
        sns.kdeplot(max_d_mins, var_d_mins)
        xlabel(r'$d^*$ (interpreting as $-\mu$)')
        ylabel(r'variance of $d_{i}^{min}$ (interpreting as $K$)')
        scatter(max_d_mins, var_d_mins, color='k', s=.5, label='Samples')
        title('Random networks with %i nodes and %.0f%% connection probability, %i samples\n'
              'Contours: Kernely density estimate'%(n_nodes, p*100, n_iterations))
        plots.savefig(gcf())

plot.close()


# In[81]:

plots.close()


# In[15]:

df = pd.DataFrame(columns=['A', 'B', 'C'])
df['A'] = ['a', 'a', 'b', 'b']
df['B'] = [1,2,3,4]
df['C'] = [1,2,3,4]

the_indices_we_want = df.ix[[0,3],['B','C']]
df = df.set_index(['B', 'C'])


# In[25]:

get_ipython().run_cell_magic('time', '', "df.ix[pd.Index(the_indices_we_want.astype('object'))]")


# In[18]:

df.loc[pd.Index(the_indices_we_want)]


# In[3]:

df = pd.DataFrame(columns=['A', 'B', 'C'])
df['A'] = ['a', 'a', 'b', 'b']
df['B'] = [1,2,3,4]
df['C'] = [1,2,3,4]

the_indices_we_want = df.ix[[0,3],['A','B']]
df = df.set_index(['A', 'B']) #Create a multiindex

df.ix[the_indices_we_want] #ValueError: Cannot index with multidimensional key

df.ix[[tuple(x) for x in the_indices_we_want.values]]


# In[7]:

pd.Index(the_indices_we_want)


# In[57]:

the_index_we_want


# In[58]:

df


# In[72]:

get_ipython().magic('pinfo df.loc')


# In[73]:

df.ix[[('b',4), ('b',4)]]


# In[68]:

df.loc[[tuple(x) for x in the_index_we_want.values]]


# In[54]:

df.ix[the_index_we_want]


# In[33]:

the_index_we_want['A']


# In[34]:

df.ix[the_index_we_want['A']]


# In[52]:

df.loc[['a', 'b'], [1,3]]


# In[50]:

df.loc[the_index_we_want.values.T[0].tolist()]


# In[42]:

df.xs(the_index_we_want.values)


# In[30]:

df.ix[the_index_we_want.values]


# In[19]:

df


# In[15]:

df.ix[the_index_we_want['A']]


# In[20]:

df


# In[23]:

df.ix['a'].ix[3]


# In[38]:

from pylab import *
n = 10.0
x = arange(n)

y = array([1.0/(2**i) for i in arange(n)])

plot(x,y)
xscale('log')
yscale('log')


# In[32]:

from pylab import *
n = 100.0
x = arange(n)

y = array([1.0/(2**(i+1)) for i in arange(n)])
y[1:] /= y[:-1]
plot(x,y)
xscale('log')
yscale('log')


# In[51]:

y = 1
x=0

for i in range(2,100):
    #dy = -(y/2.0)#*y
    y=(1/(2.0**i))/(1-(1/(2.0**(i-1))))
    #print(y)
    x+=1
    scatter(x,y)
xscale('log')
yscale('log')


# In[33]:

y[:10]


# In[3]:

get_ipython().magic('pylab inline')


# In[4]:

import pandas as pd
df = pd.DataFrame(index=pd.date_range(pd.datetime(1970,1,1),pd.datetime(2000,1,1), freq='5AS'))
df['data']=.5
df['A'] = randint(0,2,len(df.index))

get_ipython().magic("timeit df.groupby('A').sum()")
#100 loops, best of 3: 2.72 ms per loop
get_ipython().magic("timeit df.groupby('A').tshift(-1, freq='5AS')")


# In[5]:

(2000-1970)/5.


# In[10]:

import pandas as pd
df = pd.DataFrame(index=pd.date_range(pd.datetime(1975,1,1),pd.datetime(2010,1,1),freq='5AS'))
df['data']=.5
df['A'] = randint(0,2,len(df.index))
df['B'] = randint(0,2,len(df.index))

get_ipython().magic("timeit df.groupby(['A','B']).sum()")
#100 loops, best of 3: 2.72 ms per loop
get_ipython().magic("timeit df.groupby(['A','B']).tshift(-1, freq='5AS')")


# In[2]:

import pandas as pd


# In[1]:

get_ipython().magic('pylab inline')


# In[16]:

index = pd.MultiIndex.from_product([arange(50),
                                     arange(5),
                                     pd.date_range(pd.datetime(1975,1,1),
                                                   pd.datetime(2010,1,1),
                                                   freq='5AS')],
                                   names=['A', 'B', 'Year'])

df = pd.DataFrame(index=index)
df['data']=.5

get_ipython().magic("timeit df.reset_index(['A','B']).groupby(['A','B']).sum()")
get_ipython().magic("timeit df.reset_index(['A','B']).groupby(['A','B']).tshift(-1, freq='5AS')")


# In[15]:

n_A = 500
n_B = 50
index = pd.MultiIndex.from_product([arange(n_A),
                                     arange(n_B),
                                     pd.date_range(pd.datetime(1975,1,1),
                                                   pd.datetime(2010,1,1),
                                                   freq='5AS')],
                                   names=['A', 'B', 'Year'])

df = pd.DataFrame(index=index)
df['data']=.5

get_ipython().magic("timeit df.reset_index(['A','B']).groupby(['A','B']).sum()")
#100 loops, best of 3: 2.72 ms per loop
get_ipython().magic("timeit df.reset_index(['A','B']).groupby(['A','B']).tshift(-1, freq='5AS')")


# In[6]:

n_A = 50
n_B = 5
index = pd.MultiIndex.from_product([arange(n_A),
                                     arange(n_B),
                                     arange(1975, 2010,5)],
                                   names=['A', 'B', 'Year'])

df = pd.DataFrame(index=index)
df['data']=.5

get_ipython().magic("timeit df.reset_index().groupby(['A','B'])['Year'].sum()")
#100 loops, best of 3: 2.72 ms per loop
get_ipython().magic("timeit df.reset_index().groupby(['A','B'])['Year'].transform(lambda x:x-5)")


# In[5]:

n_A = 500
n_B = 50
index = pd.MultiIndex.from_product([arange(n_A),
                                     arange(n_B),
                                     arange(1975, 2010,5)],
                                   names=['A', 'B', 'Year'])

df = pd.DataFrame(index=index)
df['data']=.5

get_ipython().magic("timeit df.reset_index().groupby(['A','B'])['Year'].sum()")
#100 loops, best of 3: 2.72 ms per loop
get_ipython().magic("timeit df.reset_index().groupby(['A','B'])['Year'].transform(lambda x:x-5)")


# In[12]:

import pandas as pd
import numpy as np

data = [[1,2],
        [1,4],
        [4,2],
        [2,3]]

df = pd.DataFrame(columns=['X', 'Y'],
                  data=data)

df_permuted = df.copy()

df_permuted.Y = np.random.permutation(df.Y)

print(df.X==df.Y)

print(df_permuted.X==df_permuted.Y)



# In[2]:

import pandas as pd
import numpy as np

data = [[1,2],
        [1,4],
        [4,2],
        [2,3]]

df = pd.DataFrame(columns=['X', 'Y'],
              data=data)


df_permuted = df.copy()

df_permuted.Y = np.random.permutation(df.Y)

print(df.X==df.Y)
#0    False
#1    False
#2    False
#3    False
#dtype: bool

print(df_permuted.X==df_permuted.Y)
#0    False
#1    False
#2    False
#3     True
#dtype: bool


# In[7]:

from numpy.random import choice
for i in df.index:
    other_rows = df[(df.ix[i].X != df.Y) * (df.ix[i].Y != df.X)]
    selected_row = choice(other_rows.index)
    original_Y = df.ix[i].Y
    df.ix[i].Y = df.ix[selected_row].Y
    df.ix[selected_row].Y = original_Y


# In[8]:

print(df.X==df.Y)


# In[1]:

import pandas as pd


# In[ ]:

get_ipython().magic('pylab inline')


# In[66]:

import pandas as pd
from pylab import *
df = pd.DataFrame(columns = ['line', 
                             'sample', 
                             'number_of_rhizoids']) 
                            #I like having column names use _ instead of spaces, so then they're accessable as an 
                            #attribute of the dataframe. Accessing them as strings is probably smarter, though. 
        
df.line = ['a', 'b', 'c', 'd']*10
df.sample = ravel([[sample]*4 for sample in range(10)])
df.number_of_rhizoids = randint(100, None, 40)


# In[71]:

df_index = df.set_index('line')


# In[73]:

df_multiindex = df.set_index(['line','sample'])


# In[75]:

what_you_want = df.pivot(index='sample', columns='line', values='number_of_rhizoids')
what_you_want.plot()


# In[ ]:

['a', 'b', 'c', 'd']*10


# In[58]:

[[sample]*4 for sample in range(10)]


# In[59]:

ravel([[sample]*4 for sample in range(10)])


# In[ ]:




# In[ ]:




# In[32]:

df.groupby('line').plot()


# In[37]:

get_ipython().magic('pinfo sns.tsplot')


# In[38]:

sns.tsplot(df, 
           time="sample",
           value="number_of_rhizoids",
           condition="line")


# In[34]:

get_ipython().magic('pinfo sns.tsplot')


# In[31]:

df.set_index('line').grouby(plot()


# In[30]:

df.plot(x='sample', y='number_of_rhizoids', subplots=True)


# In[25]:

import seaborn as sns
sns.plot(df)


# In[8]:

from numpy.random import choice


# In[9]:

get_ipython().magic('pinfo choice')


# In[4]:

### Using an index, which is what I would do.

df.set_index('line')


# In[3]:

pan = pd.Panel(df)


# In[1]:

get_ipython().magic('pylab inline')


# In[41]:

x = arange(100)

fast_slope = -1
slow_slope = -.5

y_fast = x*fast_slope+100
y_slow = x*slow_slope+100
plot(x,y_fast, label='Fast (Slope: %.1f)'%fast_slope)
plot(x,y_slow, label='Slow (Slope: %.1f)'%slow_slope)
legend()
ylabel("Number Unmobilized")
xlabel("Time Step")

text(5, 25, "Average Mobilization Time Fast/Slow: %.2f"%(mean(y_fast)/mean(y_slow)))
text(5, 20, "Average Mobilization Time Slow/Fast: %.2f"%(mean(y_slow)/mean(y_fast)))
text(5, 10, "Hazard Ratio Fast (relative to slow): %.2f"%(fast_slope/slow_slope))
text(5, 5, "Hazard Ratio Slow (relative to fast): %.2f"%(slow_slope/fast_slope))


# In[50]:

x_fast = arange(100)
x_slow = arange(200)

fast_slope = -1
slow_slope = -.5

y_fast = x*fast_slope+100
y_slow = arange(200)*slow_slope+100

plot(y_fast, label='Fast (Slope: %.1f)'%fast_slope)
plot(y_slow, label='Slow (Slope: %.1f)'%slow_slope)
legend()
ylabel("Number Unmobilized")
xlabel("Time Step")

text(5, 25, "Average Mobilization Time Fast/Slow: %.2f"%(mean(arange(len(y_fast)))/mean(arange(len(y_slow)))))
text(5, 20, "Average Mobilization Time Slow/Fast: %.2f"%(mean(arange(len(y_slow)))/mean(arange(len(y_fast)))))
text(5, 10, "Hazard Ratio Fast (relative to slow): %.2f"%(fast_slope/slow_slope))
text(5, 5, "Hazard Ratio Slow (relative to fast): %.2f"%(slow_slope/fast_slope))


# In[47]:

mean(y_fast)


# In[46]:

mean(y_slow)


# In[39]:

x = arange(100)

y_fast = array(([100]*25)+([0]*75))
y_slow = array(([100]*50)+([0]*50))
y_fast = y_slow*.5
plot(x,y_fast, label='Fast')
plot(x,y_slow, label='Slow')
legend()
ylabel("Number Unmobilized")
xlabel("Time Step")

text(10, 25,b "Average Speed Fast/Slow: %.2f"%(mean(y_fast)/mean(y_slow)))
text(10, 20, "Average Speed Slow/Fast: %.2f"%(mean(y_slow)/mean(y_fast)))
#text(10, 10, "Hazard Ratio Fast (relative to slow): 2.0)
#text(10, 5, "Hazard Ratio Slow (relative to fast): %.2f"%(slow_slope/fast_slope))


# In[3]:

get_ipython().magic('pylab inline')


# In[12]:

import igraph
from igraph import Graph

g = Graph.Erdos_Renyi(n=50,p=.1)
g.es['weight']= rand(g.vcount())

print(g.spanning_tree())


# In[2]:

get_ipython().magic('matplotlib inline')


# In[3]:

from pylab import *


# In[8]:

x = arange(0, 20, .1)


# In[12]:

plot(x, sin(x)+1)
plot(x, 2-(sin(x)+1))


# In[14]:

from scipy.stats import pearsonr
pearsonr(sin(x)+1, 2-(sin(x)+1))

