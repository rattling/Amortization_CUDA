
# coding: utf-8


def setUp(numba_flag):
    
    global shares,i, pd, np, dt, arange, base_dir, input_dir, output_dir, os, random, math, sys, time, timer, rowCount, df
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
    

@numba.jit(nopython=True)        
def getSegmentsInnerN (start, end, cumVest,days,i):    
    maxSlope = 0
    #The first tranche is a special case as slope is just y/x
    for j in range(start,end):
        if j==0:
            tmpSlope=cumVest[i,0]/days[i,0] 
        #Otherwise calculate the slope as (y2-y1)/(x2-x1)
        else:
            tmpSlope=(cumVest[i,j]-cumVest[i, j-1])/(days[i,j]-days[i,j-1])
        #Keep a record of the max slope so far
        if tmpSlope > maxSlope:
            last=j
            maxSlope=tmpSlope
    #Return the last tranche of the max slope line - this is the end of the new segment
    return last
      
		
@numba.jit(nopython=True)        
def getSegmentsOuterN(end, cumVest,days, i, tmpSeg):       
    start = 0
    seg=0
    #Run getSegmentsInner from first to last tranche on the first go
    #Then keep running through for remaining tranches
    while start < end:
        #The inner loop establishes a segment and returns the last tranche for the segment
        last = getSegmentsInnerN(start, end, cumVest,days, i)
        #As this is a new segment, the segment number increases
        seg=seg+1   
        #Allocate the segment number to all tranches in the segment
        tmpSeg[0,start:last+1]=seg
        #Next iteration will start from the next unallocated tranche
        start=last+1
    return tmpSeg  
    

@numba.jit(nopython=True)
def cumVestWrapperN(rowCount, numTranches, shares, vals, cumVest):
    #This is a hot loop. No dependence between obs so Numba should do in parallel
    for i in range(0,rowCount):      
        total=0
        # GET CUMULATIVE VESTMENT AMOUNT PER GRANT FOR EACH TRANCHE
        for j in range(0,numTranches[i]+1):            
            total+=shares[i,j]*vals[i,j]
            cumVest[i,j]=total  
    return cumVest
    

@numba.jit(nopython=True)   
def segWrapperN(rowCount,n, numTranches, segments, days, cumVest):
    #This is a hot loop. No dependence between obs so Numba should do in parallel
    for i in range(0,rowCount):            
        #Assign segment number to each tranche
        #Create an empty vector to hold the result for this grant  
        tmpSeg = np.array([ [ 0 for x in range(n) ] for y in range(1) ])
        #End is number of tranches for this grant (differs per grant)
        end=numTranches[i]+1
        #Update the segment numbers for this grant
        segments[i,:] = getSegmentsOuterN(end,cumVest,days, i, tmpSeg) 
    return segments
    

def importData():
    fileName='amort.csv'
    stg= pd.read_csv(os.path.join(input_dir,fileName), parse_dates=[2,3], encoding='utf8') 
    stg['Days'] = (stg['VestDate']-stg['GrantDate']).astype(dt.timedelta).map(lambda x: np.nan if pd.isnull(x) else x.days)  
   
    #  Can push days diff calc into the numba piece - depending on data format. If SAS days since 1960 would work ok
    #  Need to deal with slight differences in periods - could mess up the segment assignment if not regularized    
#    TRANSPOSE AND RENAME COLUMNS TO GET ONE ROW PER GRANT
    df = stg.pivot_table(index=["GrantID"], columns=["TrancheID"])
    newNames =[]
    for i in range(0,df.shape[1]):
        newNames.append(df.columns.get_level_values(0)[i]+str(df.columns.get_level_values(1)[i]))
    df.columns=newNames  
   
    pd.options.display.float_format = '{:,.0f}'.format
    return df        

def runCode(numba_flag, scaling):
    
    setUp(numba_flag)
    df = importData()     
    #Scale up the inputs for performance testing
    df=pd.concat([df]*scaling)
    
    #Creates sets of columns with same prefix
    dayCols = [col for col in df if col.startswith('Day')]
    valCols = [col for col in df if col.startswith('Share')]
    shareCols = [col for col in df if col.startswith('Vest')]     
    
    #Roll out to numpy series that can be used by Numba
    shares = df[shareCols].values
    vals = df[valCols].values
    days =df[dayCols].values
    n=shares.shape[1]    
    rowCount=len(df.index)
    
    #Create arrays to store results 
    segments = np.zeros(shape=(rowCount, n), dtype='int64')  
    cumVest = np.zeros(shape=(rowCount, n), dtype='int64')     
    
    #Create a vector with the number of tranches per grant as it will differ
    numTranches=shares.shape[1] - (~np.isnan(shares))[:, ::-1].argmax(1) -1
    
    #Calculate cumulative vestments for each tranche
    func_start = timer()    
    cumVest=cumVestWrapper(rowCount, numTranches, shares,vals, cumVest)
    timing=timer()-func_start
    print("Function: cumVest duration (seconds):" + str(timing))
    
    #Allocate a segment number to each tranche
    func_start = timer()   
    segments=segWrapper(rowCount,n, numTranches, segments, days, cumVest)
    timing=timer()-func_start
    print("Function: segWrapper duration (seconds):" + str(timing))
    
    print(cumVest)
    print(segments)
	
    #Repeat above using Numba
    if numba_flag == 1:
        #Calculate cumulative vestments for each tranche
        func_start = timer()    
        cumVest=cumVestWrapperN(rowCount, numTranches, shares,vals, cumVest)
        timing=timer()-func_start
        print("Function: cumVestN duration (seconds):" + str(timing))
        
        #Allocate a segment number to each tranche
        func_start = timer()   
        segments=segWrapperN(rowCount,n, numTranches, segments, days, cumVest)
        timing=timer()-func_start
        print("Function: segWrapperN duration (seconds):" + str(timing))
    
    print(cumVest)
    print(segments)

	    #Add New Data to DataFrame and Export to CSV
    for i in range(0, n ):
        cumCol="cumVest"+str(i+1) 
        df[cumCol]=cumVest[:, i]
        segCol="segment"+str(i+1)            
        df[segCol]=segments[:, i]  
    df.to_csv(os.path.join(output_dir,'amort_updated.csv'))


runCode(1, 100000)






