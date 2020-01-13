# -*- coding: utf-8 -*-
"""
Created on Sun Jan 12 21:08:24 2020

@author: Aspire
"""

import csv

with open('cleveland.data') as input_file:
       lines = input_file.readlines()
       newLines = []
       for line in lines:
          newLine = line.strip().split('name')
          newLines.append( newLine )
          #print(newLines)
    
with open('output.csv', 'wb') as test_file:
       file_writer = csv.writer(test_file)
       file_writer.writerows( newLines )