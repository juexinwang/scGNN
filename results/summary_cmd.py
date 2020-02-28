import os
import argparse
parser = argparse.ArgumentParser(description='Read Results from results bach scripts')
parser.add_argument('--imputeMode', default=False, action='store_true',
                    help='impute or not (default: False). Caution: usually change npuDir if set imputeMode as true')
args = parser.parse_args()

reDict = {}
with open('jobinfo_imp.txt') as f:
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

numDict={0:'z',1:'z0',2:'z1',3:'z2',4:'z3',5:'z4',6:'z5',7:'z6',8:'z7',9:'z8',10:'z9',11:'z',12:'z0',13:'z1',14:'z2',15:'z3',16:'z4',17:'z5',18:'z6',19:'z7',20:'z8',21:'z9'}
nameDict={'Z0':'','Z1':'','Z2':'','Z3':'','Z4':'','Z5':'','Z6':'','Z7':'','Z8':'','Z9':''}

# for all
for i in range(13):
    allstr = []
    selstr = []
    for j in range(63):
        tag = True
        if args.imputeMode:
            with open('imputation/RI_'+str(j)+'_'+str(i)+'.txt') as f:
                lines = f.readlines()
                count = 0
                for line in lines:
                    if line.startswith('Traceback (most recent call last):'):
                        tag = False
                    elif line.startswith('FileNotFoundError:'):
                        tag = True
                        count += 1
                    elif tag:
                        allstr.append(str(j)+'\t'+numDict[count]+'\t'+line)
                        count += 1
                f.close()
        else:
            with open('celltype/RC_'+str(j)+'_'+str(i)+'.txt') as f:
                lines = f.readlines()
                count = 0
                tmpstr = ''
                tmpstr1=''
                tstr = ''
                tstr1 = ''
                for line in lines:
                    # For all str
                    allstr.append(str(j)+'\t'+line)

                    line = line.strip()
                    words = line.split()
                    if len(words)>3:
                        if words[0]=='Graph':                        
                            if count==1:
                                tstr = tmpstr
                                tstr1 = tmpstr1
                            count+=1

                        elif words[0] in nameDict:
                            tmpstr =''
                            tmpstr1=''
                        if words[1]=='Louvain':
                            tmpstr = tmpstr + words[2] + ',' 
                            tmpstr1 = tmpstr1 + words[5] + ',' 
                        if words[0]=='KMeans' or words[0]=='SpectralClustering' or words[0]=='Birch':
                            tmpstr = tmpstr + words[1] + ','
                            tmpstr1 = tmpstr1 + words[4] + ','
                selstr.append(str(j)+','+tstr+tstr1+tmpstr+tmpstr1+'\n')
                f.close() 
            
    if args.imputeMode:
        outputfilename = 'imputation/results_'+str(i)+'.txt'
    else:
        outputfilename = 'celltype/results_'+str(i)+'.txt'
        seloutputfilename = 'celltype/sel_results_'+str(i)+'.csv'
    with open(outputfilename,'w') as fw:
        fw.writelines("%s" % strr for strr in allstr)
        fw.close()
    with open(seloutputfilename,'w') as fw:
        fw.writelines("%s" % strr for strr in selstr)
        fw.close()
