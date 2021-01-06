# This program generates a list of names of the files in a folder

import glob

fh = open("MetricList.txt","w+")

org_map = []

gen_map = []

for file in glob.glob("./Maps/val/*"):
    org_map.append(file)

for file in glob.glob("./Result/*"):
    gen_map.append(file)

org_map.sort()
gen_map.sort()

for i in range(len(org_map)):
    fh.write(org_map[i]+"\t"+gen_map[i]+"\n")

