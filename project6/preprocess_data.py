"""
For pre-processing data. Data comes as comma-separated values, however some of
the strings in the data have commas in them and throw off the SQL ingestion.
Re-printing file as pipe-separated values
"""

import pandas as pd

the_file = "pisa2012.csv"
col_heads = open(the_file).readline().split(",")
dtypes_dict = {}
for head in col_heads[1:-1]:
    dtypes_dict[head] = object

print "READING FILE %s" % the_file 
the_data = pd.read_csv(the_file, dtype=dtypes_dict)

output_file = "pisa2012_reprocessed_nohead.dat"
print "WRITING FILE %s" % output_file
the_data.to_csv(output_file, sep="|", header=False, index=False)