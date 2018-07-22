# -*- coding: utf-8 -*-
"""
Created on Sat Mar 31 11:40:47 2018

@author: JohnArm
"""

from io import StringIO
import pandas as pd

txt = """UniqueID   Month   ActivityType    Score   Posts
Joe May sas 3151    1980
Tom May sas 792 690
DomPazz May sas 597 417
Reeza   May sas 549 511
Longfish    May sas 478 255
AndyHayden  May pandas  8063    1281
jezrael May pandas  7976    4754
EdChum  May pandas  6579    2501
unutbu  May python  39827   6409
piRSquared  May pandas  5024    3004
Joe May sas-macro   343 184
Tom May sas-macro   96  83
DomPazz May sas-macro   46  26
Reeza   May sas-macro   54  39
Longfish    May sql 62  39
AndyHayden  May python  7991    1360
jezrael May python  7485    4185
EdChum  May python  6439    2363
unutbu  May numpy   6382    1035
piRSquared  May python  4625    2782
Joe May sql 279 189
Tom May sql 91  79
DomPazz May sql 33  30
Reeza   May sql 32  38
Longfish    May variables   19  8
AndyHayden  May dataframe   2264    191
jezrael May dataframe   2847    1601
EdChum  May dataframe   1748    529
unutbu  May pandas  6345    1276
piRSquared  May dataframe   1696    853"""

df = pd.read_table(StringIO(txt), sep="\s+")


print(trans)                       
