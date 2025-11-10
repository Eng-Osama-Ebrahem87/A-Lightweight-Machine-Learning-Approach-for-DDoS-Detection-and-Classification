import json
import csv

#Target_File_Loc = r"E:\VeReMi Dataset\Some VeReMi files\traceJSON-9-7-A0-1-0.json"

Target_File_Loc = r"E:\VeReMi Dataset\Some VeReMi files\traceJSON-81-79-A0-22-0.json"

Dest_File_Loc =  r"E:\VeReMi Dataset\Some VeReMi files\jsonoutput--81-79-A0-22-0.csv" 


#Dest_File_Loc =  r"E:\VeReMi Dataset\Some VeReMi files\jsonoutput--537-535-A0-301-0.csv" 

with open(Target_File_Loc ) as jf:
    #jd = json.load(jf)
    jd = [json.loads(line) for line in jf]
df = open(Dest_File_Loc , 'w', newline='')
cw = csv.writer(df)

c = 0
for data in jd:
    if c == 0:
        header = data.keys()
        cw.writerow(header)
        c += 1
    cw.writerow(data.values())

df.close()
