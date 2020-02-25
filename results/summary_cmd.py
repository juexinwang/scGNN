import os
import argparse
parser = argparse.ArgumentParser(description='Read Results from results bach scripts')
parser.add_argument('--imputeMode', default=False, action='store_true',
                    help='impute or not (default: False). Caution: usually change npuDir if set imputeMode as true')
args = parser.parse_args()

reDict = {}
with open('jobinfo_usage.txt') as f:
    lines = f.readlines()
    for line in lines:
        line = line.strip()
        words = line.split()
        reDict[words[2]] = words[0]
    f.close()

for i in range(63):
    for j in range(13):
        # 'python summary.py --fileName results-19687313.out --outFileName RC_0_0.txt'        
        if args.imputeMode:
            name = 'RI_'+str(i)+'_'+str(j)
            commandStr = 'cat results-' + reDict[name] + '.out > imputation/' + name + '.txt'        
        else:
            name = 'RC_'+str(i)+'_'+str(j)
            commandStr = 'python summary.py --fileName results-' + reDict[name] + '.out --outFileName celltype/' + name + '.txt'
        # print(commandStr)
        os.system(commandStr)

for i in range(13):
    allstr = []
    for j in range(63):
        allstr.append(j)
        with open('imputation/RI_'+str(j)+'_'+str(i)+'.txt') as f:
            lines = f.readlines()
            for line in lines:
                allstr.append('\t'+line)
            f.close()
    
    with open('imputation/results_'+str(j)+'.txt','w') as fw:
        fw.writelines("%s\n" % strr for strr in allstr)
        fw.close()
    

