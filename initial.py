# -*- coding: utf-8 -*-
"""
Created on Sun Jan 12 21:08:24 2020

@author: Aspire
"""

import csv

with open('clev.txt') as input_file:
       lines = input_file
       newLines = []
       newRows = []
       Rows = []
       for line in lines:
         
          newLine = line.split()
          for i in newLine:
              if i=='name':
                  newLines.append(newRows)
                  newRows=[]
                  
                 
              else:
                  newRows.append( i )
          
#          print newLines
#          print '\n'
               
       
       
              
       
          
     

   
with open('output.csv', 'wb') as test_file:
       file_writer = csv.writer(test_file)
       file_writer.writerows( newLines )


