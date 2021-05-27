#!@PYTHON3@
#%%
import struct
import pickle
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd



plt.rcParams['font.family'] = 'serif'
plt.rcParams['mathtext.fontset'] = 'dejavuserif'
plt.rcParams.update({'font.size': 13})
# plt.rcParams.update({'font.size': 15})

# expected coverage for transcripts
def ReadRawCorrection(filename):
	ExpectedProb={}
	fp=open(filename, 'rb')
	numtrans=struct.unpack('i', fp.read(4))[0]
	for i in range(numtrans):
		namelen=struct.unpack('i', fp.read(4))[0]
		seqlen=struct.unpack('i', fp.read(4))[0]
		name=""
		correction=np.zeros(seqlen)
		for j in range(namelen):
			name+=struct.unpack('c', fp.read(1))[0].decode('utf-8')
		for j in range(seqlen):
			correction[j]=struct.unpack('d', fp.read(8))[0]
		ExpectedProb[name]=correction
	fp.close()
	print("Finish reading theoretical distributions for {} transcripts.".format(len(ExpectedProb)))
	return ExpectedProb

ExpectedProb=ReadRawCorrection("/home/priyamvada/data/ERR030875/correction.dat")
#ExpectedProb=ReadRawCorrection("/home/congm1/savanna/savannacong33/SADrealdata/HumanBodyMap/salmon_Full_ERR030873/correction.dat")
for key, value in ExpectedProb.items():
    print(key, ' : ', value)





df=pd.DataFrame(ExpectedProb.items(),columns=['Transcript Name', 'Expected coverage'])
print(df)



# import struct
# import pickle
# import numpy as np
# from matplotlib import pyplot as plt
# import pandas as pd

plt.rcParams['font.family'] = 'serif'
plt.rcParams['mathtext.fontset'] = 'dejavuserif'
plt.rcParams.update({'font.size': 13})
# plt.rcParams.update({'font.size': 15})
# observed data coverage for transcripts
def ReadRawStartPos(filename):
	TrueRaw={}
	fp=open(filename, 'rb')
	numtrans=struct.unpack('i', fp.read(4))[0]
	for i in range(numtrans):
		namelen=struct.unpack('i', fp.read(4))[0]
		seqlen=struct.unpack('i', fp.read(4))[0]
		name=""
		poses=np.zeros(seqlen, dtype=np.int)
		counts=np.zeros(seqlen)
		for j in range(namelen):
			name+=struct.unpack('c', fp.read(1))[0].decode('utf-8')
		for j in range(seqlen):
			poses[j]=struct.unpack('i', fp.read(4))[0]
		for j in range(seqlen):
			counts[j]=struct.unpack('d', fp.read(8))[0]
		tmp=np.zeros(poses[-1]+1)
		for j in range(len(poses)):
			tmp[poses[j]] = counts[j]
		TrueRaw[name]=tmp
	fp.close()
	print("Finish reading actual distribution for {} transcripts.".format(len(TrueRaw)))
	return TrueRaw

TrueRaw=ReadRawStartPos("/home/priyamvada/data/ERR030875/startpos.dat")
#TrueRaw=ReadRawStartPos("/home/congm1/savanna/savannacong33/SADrealdata/HumanBodyMap/salmon_Full_ERR030873/startpos.dat")
for key, value in TrueRaw.items():
    print(key, ' : ', value)

#len(TrueRaw)
	#import pickle
# key_to_lookup = 'ENST00000387405.1'
# if key_to_lookup in TrueRaw:
#   print ("Key exists")
# else:
#   print ("Key does not exist")



df1=pd.DataFrame(TrueRaw.items(),columns=['Transcript Name', 'observed coverage'])
print(df1)


dfmerge=pd.merge(df,df1,on='Transcript Name')
print(dfmerge)




# extract and label (1) adjustable anomalous transcripts
import pandas as pd

dfa = pd.read_csv("/home/priyamvada/data/ERR030875/test_correctapprox9/test_adjusted_quantification.tsv", sep="\t")
#dfa = pd.read_csv("/home/congm1/savanna/savannacong33/SADrealdata/HumanBodyMap/salmon_Full_ERR030873/test_correctapprox9/test_adjusted_quantification.tsv", sep="\t")
print (dfa)


dfa1 = dfa.iloc[:,0:1]
print(dfa1)
dfa1=dfa1.rename(columns={'# Name': 'Transcript Name'})
print(dfa1)

dfa2=dfa1.assign(Label='1')
print(dfa2)



#%%

#extract and label (2) non-adjustable anomalous transcripts
dfan = pd.read_csv("/home/priyamvada/data/ERR030875/test_correctapprox9/test_unadjustable_pvalue.tsv", sep="\t")
#dfan = pd.read_csv("/home/congm1/savanna/savannacong33/SADrealdata/HumanBodyMap/salmon_Full_ERR030873/test_correctapprox9/test_unadjustable_pvalue.tsv", sep="\t")
print (dfan)


dfan1 = dfan.iloc[:,0:1]
print(dfan1)

dfan1=dfan1.rename(columns={'#Name': 'Transcript Name'})
print(dfan1)
dfan2=dfan1.assign(Label='1')
print(dfan2)





anomalous_merge = pd.concat([dfa2, dfan2], axis=0)
print(anomalous_merge)
len(anomalous_merge)


allabnormal_merge=pd.merge(dfmerge,anomalous_merge,on='Transcript Name')
print(allabnormal_merge)


# make list of anomalous transcripts names

anomal_list = list(allabnormal_merge['Transcript Name'])
#print(anomal_list)


#%%

Normal_Transcripts=[]

for key,_ in TrueRaw.items():
	Normal_Transcripts.append(key)

#print(Normal_Transcripts)
#print(len(Normal_Transcripts))

#print('normaldone')


l3 = [x for x in Normal_Transcripts if x not in anomal_list]

print(l3)
print(len(l3))


#%%


normaldf=pd.DataFrame(l3,columns=['Transcript Name'])
print (normaldf)

normaldf1=normaldf.assign(Label='0')
print (normaldf1)

# print("normaldf done about to merge")
# all normal transcripts merged with expected and observed coverages
allnormal_merge=pd.merge(dfmerge,normaldf1,on='Transcript Name')
print(allnormal_merge)

## check to make sure no overlap between normal and abnormal transcripts

# nomal_list = list(allnormal_merge['Transcript Name'])
# m=list(set(nomal_list).intersection(anomal_list))
# print(m)



allnormal30875 = allnormal_merge.to_numpy()

#np.save('/Users/priyamvadakumar/Desktop/CARLlslab/allabnormal30875.npy', allabnormal30875)

np.save("/home/priyamvada/data/allnormal30875.npy", allnormal30875)

# allnormal_merge.to_csv("/home/priyamvada/data/{ERR030873_id}_normal.csv", index=False)

# f = open("/home/priyamvada/data/{ERR030873_id}_allnormal.dat", "wb")
# pickle.dump(allnormal_merge, f)

# f.close()



