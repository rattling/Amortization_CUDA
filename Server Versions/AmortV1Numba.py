
# coding: utf-8

# In[ ]:




# In[63]:

def getSegmentsInner (start, end, cumVest,days,i):    
    maxSlope = 0
    for j in range(start,end):
        if j==0:
            tmpSlope=cumVest[i,0]/days[i,0] 
        else:
            tmpSlope=(cumVest[i,j]-cumVest[i, j-1])/(days[i,j]-days[i,j-1])
        if tmpSlope > maxSlope:
            last=j
            maxSlope=tmpSlope      
    return last
        
     
def getSegmentsOuter(end, cumVest,days, i, tmpSeg):       
    start = 0
    seg=0
    while start < end:
        last = getSegmentsInner(start, end, cumVest,days, i)
        seg=seg+1      
        tmpSeg[0,start:last+1]=seg
        start=last+1
    return tmpSeg      
   
def cumVestWrapper(rowCount, numTranches, shares, vals, cumVest):
    for i in range(0,rowCount):      
        total=0
        # GET CUMULATIVE VESTMENT AMOUNT PER GRANT
        for j in range(0,numTranches[i]+1):            
            total+=shares[i,j]*vals[i,j]
            cumVest[i,j]=total  
    return cumVest    
 
def segWrapper(rowCount,n, numTranches, segments, days, cumVest):
    for i in range(0,rowCount):            
        #ASSIGN SEGMENTS TO VESTMENT 
        #print(numTranches)   
        tmpSeg = np.array([ [ 0 for x in range(n) ] for y in range(1) ])
        end=numTranches[i]+1
        segments[i,:] = getSegmentsOuter(end,cumVest,days, i, tmpSeg) 
    return segments
    
    
    

     




# In[64]:

@numba.jit(nopython=True)        
def getSegmentsInnerN (start, end, cumVest,days,i):    
    maxSlope = 0
    for j in range(start,end):
        if j==0:
            tmpSlope=cumVest[i,0]/days[i,0] 
        else:
            tmpSlope=(cumVest[i,j]-cumVest[i, j-1])/(days[i,j]-days[i,j-1])
        if tmpSlope > maxSlope:
            last=j
            maxSlope=tmpSlope      
    return last
        
@numba.jit(nopython=True)        
def getSegmentsOuterN(end, cumVest,days, i, tmpSeg):       
    start = 0
    seg=0
    while start < end:
        last = getSegmentsInnerN(start, end, cumVest,days, i)
        seg=seg+1      
        tmpSeg[0,start:last+1]=seg
        start=last+1
    return tmpSeg  
    
@numba.jit(nopython=True)    
def cumVestWrapperN(rowCount, numTranches, shares, vals, cumVest):
    for i in range(0,rowCount):      
        total=0
        # GET CUMULATIVE VESTMENT AMOUNT PER GRANT
        for j in range(0,numTranches[i]+1):            
            total+=shares[i,j]*vals[i,j]
            cumVest[i,j]=total  
    return cumVest
    
@numba.jit(nopython=True)   
def segWrapperN(rowCount,n, numTranches, segments, days, cumVest):
    for i in range(0,rowCount):            
        #ASSIGN SEGMENTS TO VESTMENT 
        #print(numTranches)   
        tmpSeg = np.array([ [ 0 for x in range(n) ] for y in range(1) ])
        end=numTranches[i]+1
        segments[i,:] = getSegmentsOuterN(end,cumVest,days, i, tmpSeg) 
    return segments
    
    
    

     




# In[18]:

def setUp(numba_flag):
    #START PROGRAM - MAYBE PUT THIS IN A DRIVER FUNCTION FOR A SINGLE CALL WITH PARAMETERS
    global epsilon, shares,i, pd, np, dt, arange, base_dir, input_dir, output_dir, os, random, math, sys, time, timer, rowCount, df
    import pandas as pd
    import numpy as np
    import datetime as dt
    from numpy import arange
    import random
    import math
    import os
    import sys
    print(sys.version)
    import time
    from timeit import default_timer as timer
    #if numba_flag ==1:
    global numba
    import numba
    print("Using Numba")
    print(numba.__version__)
    
    #Set input_dir to where your inputs files are located
    #base_dir = r'/home/ec2-user/'
    base_dir = '/home/ec2-user/'
    input_dir = base_dir + 'inputs/'
    output_dir = base_dir + 'outputs/'


# In[19]:

def importData():
    fileName='amort.csv'
    stg= pd.read_csv(os.path.join(input_dir,fileName), parse_dates=[2,3], encoding='utf8') 
    stg['Days'] = (stg['VestDate']-stg['GrantDate']).astype(dt.timedelta).map(lambda x: np.nan if pd.isnull(x) else x.days)  
   
    #  Can push days diff calc into the numba piece - dependong on data format. If SAS days since 1960 would work grand
    #   Need to deal with slight differences in periods - could mess up the segment assignment if not regularized
    
#    TRANSPOSE AND RENAME COLUMNS TO GET ONE ROW PER GRANT
    df = stg.pivot_table(index=["GrantID"], columns=["TrancheID"])
    newNames =[]
    for i in range(0,df.shape[1]):
        newNames.append(df.columns.get_level_values(0)[i]+str(df.columns.get_level_values(1)[i]))
    df.columns=newNames  
   
    pd.options.display.float_format = '{:,.0f}'.format
    return df
        


# In[70]:

def runCode():
    
    setUp(1)
    df = importData() 
    df=pd.concat([df]*100000)
    dayCols = [col for col in df if col.startswith('Day')]
    valCols = [col for col in df if col.startswith('Share')]
    shareCols = [col for col in df if col.startswith('Vest')]     
    
    shares = df[shareCols].values
    vals = df[valCols].values
    days =df[dayCols].values
    n=shares.shape[1]    
    rowCount=len(df.index)
    
    segments = np.zeros(shape=(rowCount, n), dtype='int64')  
    cumVest = np.zeros(shape=(rowCount, n), dtype='int64') 
    tmpSeg= np.zeros(shape=(1, n), dtype='int64')
    
    #Line below might work within Numba- returns a vector which is fine or could do it per grant
    numTranches=shares.shape[1] - (~np.isnan(shares))[:, ::-1].argmax(1) -1
    
    func_start = timer()    
    cumVest=cumVestWrapper(rowCount, numTranches, shares,vals, cumVest)
    timing=timer()-func_start
    print("Function: cumVest duration (seconds):" + str(timing))
    
    func_start = timer()   
    segments=segWrapper(rowCount,n, numTranches, segments, days, cumVest)
    timing=timer()-func_start
    print("Function: segWrapper duration (seconds):" + str(timing))
    
    print(cumVest)
    print(segments)
    
    func_start = timer()    
    cumVest=cumVestWrapperN(rowCount, numTranches, shares,vals, cumVest)
    timing=timer()-func_start
    print("Function: cumVestN duration (seconds):" + str(timing))
    
    func_start = timer()   
    segments=segWrapperN(rowCount,n, numTranches, segments, days, cumVest)
    timing=timer()-func_start
    print("Function: segWrapperN duration (seconds):" + str(timing))
    
    print(cumVest)
    print(segments)


# In[71]:

runCode()


# In[34]:


np.zeros(tmpSeg)
print(tmpSeg)


# In[41]:

segments = np.zeros(shape=(4,5), dtype='int64')
segments.fill(3)



# In[42]:

print(segments)


# In[39]:

segments=3


# In[40]:

print(segments)


# In[ ]:

segments = np.zeros(shape=(4,5), dtype='int64')
segments.fill(3)
segments=[3]


# In[51]:


segments = np.array([ [ 0 for x in range(n) ] for y in range(1) ])


# In[52]:

segments


# In[69]:

df=pd.concat([df]*5)


# In[ ]:



