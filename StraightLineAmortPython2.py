# -*- coding: utf-8 -*-
"""
Created on Sat Mar 31 17:11:09 2018

@author: JohnArm
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Mar 30 18:08:51 2018

@author: JohnArm
"""

def setUp(numba_flag):
    #START PROGRAM - MAYBE PUT THIS IN A DRIVER FUNCTION FOR A SINGLE CALL WITH PARAMETERS
    global epsilon, s1,pd, np, dt, arange, base_dir, input_dir, output_dir, os, random, math, sys, time, timer, rowCount, df
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
    if numba_flag ==1:
        global numba
        import numba
        print("Using Numba")
        print(numba.__version__)
    
    #Set input_dir to where your inputs files are located
    #base_dir = r'/home/ec2-user/'
    base_dir = 'C:/Users/JohnArm/Documents/GitHub/Amortization_CUDA/'
    input_dir = base_dir + 'input/'
    output_dir = base_dir + 'output/'
    epsilon = .001
    
def importData():
#    fileName='amort' +str(s)+ '.csv'
    fileName='amort.csv'
    stg= pd.read_csv(os.path.join(input_dir,fileName), parse_dates=[2,3], encoding='utf8') 
    stg['Days'] = (stg['VestDate']-stg['GrantDate']).astype(dt.timedelta).map(lambda x: np.nan if pd.isnull(x) else x.days)
  
   
    #    Can push above calc into the numbe piece - check with paul on data format - if SAS days can be subtracted in numba
    
#    TRANSPOSE AND RENAME COLUMNS TO GET ONE ROW PER GRANT
    df = stg.pivot_table(index=["GrantID"], columns=["TrancheID"])
    newNames =[]
    for i in range(0,df.shape[1]):
        newNames.append(df.columns.get_level_values(0)[i]+str(df.columns.get_level_values(1)[i]))
    df.columns=newNames  
   
    pd.options.display.float_format = '{:,.0f}'.format
    return df

def runCode():
    setUp(0)
    df = importData()
    dayCols = [col for col in df if col.startswith('Day')]
    valCols = [col for col in df if col.startswith('Share')]
    shareCols = [col for col in df if col.startswith('Vest')]     
    
    s1 = df[shareCols].values
    s2 = df[valCols].values
    days =df[dayCols].values
    n=s1.shape[1]
    rowCount=len(df.index)
    
    segments = np.empty(shape=(rowCount, n), dtype='int64') 
    slopes = np.empty(shape=(rowCount, n), dtype='float64') 
    
    numTranches=s1.shape[1] - (~np.isnan(s1))[:, ::-1].argmax(1) -1
   
    
    for i in range(0,rowCount):      
        total=0
        #   GET CUMULATIVE VESTMENT AMOUNT PER GRANT
        for j in range(0,n):            
            total+=s1[i,j]*s2[i,j]
            s1[i,j]=total 
        # GET THE OVERALL SLOPE FOR THE GRANT
        
        overallSlope=s1[i,numTranches[i]]/days[i,numTranches[i]]
        # PROJECT POINTS TO SLOPE LINE IF UNDERNEATH IT
        for j in range(0,numTranches[i]+1): 
            if s1[i,j] < overallSlope*days[i,j]:
                s1[i,j] =  overallSlope*days[i,j]
             #GET THE SEGMENT SLOPE AND UPDATE SEGMENT IF SLOPE CHANGING
            if j==0:
                slopes[i,0]=s1[i,0]/days[i,0]
                segments[i,0] =1
            else:
                slopes[i,j]=(s1[i,j]-s1[i, j-1])/(days[i,j]-days[i,j-1])
                if abs(slopes[i,j]-slopes[i,j-1]) > epsilon:
                    segments[i,j]= segments[i,j-1]+1
                else:
                    segments[i,j]=segments[i,j-1]
                    
    print("OK")
    
    #NEXT - check hte projection piece first can calc it in Excel to check against
    #Then check slopes as above
    #Then segments
    

     
runCode()