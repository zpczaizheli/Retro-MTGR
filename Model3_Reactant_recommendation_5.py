import numpy as np
import scipy.sparse as sp
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from GetMoleculeFeature import *
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
from rdkit.Chem import AllChem, Draw
from rdkit.Chem import MACCSkeys
#from torch.nn import Linear

# 数据读取




def SymbolAtom(mol):
	Atomnum = mol.GetNumAtoms()
	for i in range(Atomnum):
		Atom = mol.GetAtomWithIdx(i)
		Atom.SetProp("atomNote", str(i))
	return mol


def GetAtomIndxBindingwithAtomlist(Mol,Atomlist):
	Result_Indx = []
	AtomNum = Mol.GetNumAtoms()
	for indx in Atomlist:
		Result_Indx.append(indx)
		for indx1 in range(AtomNum):
			Bond = Mol.GetBondBetweenAtoms(indx1,indx)
			if str(Bond) == 'None':
				continue
			else:
				Result_Indx.append(indx1)
	Result_Indx = list(set(Result_Indx))
	return Result_Indx


def Modification_AtomSymbol(Mol,Atomlist):
	AtomNum = Mol.GetNumAtoms()
	mw = Chem.RWMol(Mol)
	for indx in Atomlist:
		mw.ReplaceAtom(indx, Chem.Atom(0))  # 将列表中的原子转换成“*”
	m_edit = mw.GetMol()
	return m_edit



def GetBondStructure(Product_Mol, Bond, X):   # 得到断裂键周围X深度的结构
	AtomNum = Product_Mol.GetNumAtoms()
	bond_indx1, bond_indx2 = Bond.split(',')
	AtomnumList = [int(bond_indx1), int(bond_indx2)]
	for i in range(X):
		AtomnumList = GetAtomIndxBindingwithAtomlist(Product_Mol,AtomnumList)
	No_AtomnumList = [] 
	for i in range(AtomNum):
		if i not in AtomnumList:
			No_AtomnumList.append(i)
	Xing_Structure_Mol = Modification_AtomSymbol(Product_Mol,No_AtomnumList)  # 将非主要结构（断裂键周围深度为X的结构）的原子变为*，利于删除
	Mol_Bond_X_Structure = Chem.DeleteSubstructs(Xing_Structure_Mol, Mol_Xing)
	return Mol_Bond_X_Structure


def get_LG(mol):
	Mol_Xing = Chem.MolFromSmiles('*')
	mw = Chem.RWMol(mol)
	for atom in mol.GetAtoms():
		if atom.GetAtomMapNum() != 0:
			mw.ReplaceAtom(atom.GetIdx(), Chem.Atom(0))
	Mol = mw.GetMol()
	Mol_Bond = Chem.DeleteSubstructs(Mol, Mol_Xing)
	return Mol_Bond



def GetAtomStructure(Product_Mol, Bond, X):   # 得到断裂键周围X深度的结构
	AtomNum = Product_Mol.GetNumAtoms()
	bond_indx1, bond_indx2 = Bond.split(',')
	
	AtomnumList1 = [int(bond_indx1)]
	for i in range(X):
		AtomnumList = GetAtomIndxBindingwithAtomlist(Product_Mol,AtomnumList1)
		for j in AtomnumList:
			if j != int(bond_indx2):
				AtomnumList1.append(j)
	No_AtomnumList1 = [] 
	for i in range(AtomNum):
		if i not in AtomnumList1:
			No_AtomnumList1.append(i)
	
	Xing_Structure_Mol1 = Modification_AtomSymbol(Product_Mol,No_AtomnumList1)  # 将非主要结构（断裂键周围深度为X的结构）的原子变为*，利于删除
	mw = Chem.RWMol(Xing_Structure_Mol1)
	mw.ReplaceAtom(int(bond_indx2), Chem.Atom(33))  # 将列表中的原子转换成“*”
	Mol = mw.GetMol()
	Mol_Bond_X_Structure1 = Chem.DeleteSubstructs(Mol, Mol_Xing)
	ind1 = 0
	
	for i in range(Mol_Bond_X_Structure1.GetNumAtoms()):
		if Mol_Bond_X_Structure1.GetAtomWithIdx(i).GetSymbol() == 'As':
			ind1 = i
	
	m = Chem.RWMol(Mol_Bond_X_Structure1)
	m.ReplaceAtom(ind1, Chem.Atom(0))  # 将列表中的原子转换成“*”
	Mol1 = m.GetMol()
	#Chem.Kekulize(Mol1,True)
	
	AtomnumList2 = [int(bond_indx2)]
	for i in range(X):
		AtomnumList = GetAtomIndxBindingwithAtomlist(Product_Mol,AtomnumList2)
		for j in AtomnumList:
			if j != int(bond_indx1):
				AtomnumList2.append(j)
	No_AtomnumList2 = [] 
	for i in range(AtomNum):
		if i not in AtomnumList2:
			No_AtomnumList2.append(i)
	
	Xing_Structure_Mol2 = Modification_AtomSymbol(Product_Mol,No_AtomnumList2)  # 将非主要结构（断裂键周围深度为X的结构）的原子变为*，利于删除
	mw = Chem.RWMol(Xing_Structure_Mol2)
	mw.ReplaceAtom(int(bond_indx1), Chem.Atom(33))  # 将列表中的原子转换成“*”
	Mol = mw.GetMol()
	Mol_Bond_X_Structure2 = Chem.DeleteSubstructs(Mol, Mol_Xing)
	ind2 = 0
	for i in range(Mol_Bond_X_Structure2.GetNumAtoms()):
		if Mol_Bond_X_Structure2.GetAtomWithIdx(i).GetSymbol() == 'As':
			ind2 = i
	
	m = Chem.RWMol(Mol_Bond_X_Structure2)
	m.ReplaceAtom(ind2, Chem.Atom(0))  # 将列表中的原子转换成“*”
	Mol2 = m.GetMol()
	return Mol1, Mol2


def check_list_in_list(list1,list2):
	all_elements_in_list2 = all(element in list2 for element in list1)
	return all_elements_in_list2

def check_list_in_list(list1,list2):
	x = True
	for i in list1:
		if i in list2:
			continue
		else:
			x = False
	# ~ print(x)
	return x


def get_atoms_maplist(mol):
	list_map = []
	atoms = mol.GetAtoms()
	for atom in atoms:
		list_map.append(atom.GetAtomMapNum())
	return list_map


def GetRightMappingForSynthonAndReactant(Synthon1,Synthon2,Reactant1,Reactant2):  # 根据原子数对合成子和反应物的对应关系进行修正，使得合成子1与反应物1对应，合成子2与反应物2对应
	#print('图片')
	#mols = [Synthon1,Synthon2,Reactant1,Reactant2]
	#img = Draw.MolsToGridImage(mols, molsPerRow=4,subImgSize=(1000,1000))
	#plt.imshow(img)
	#plt.show()
	Synthon1_maplist = get_atoms_maplist(Synthon1)
	Synthon2_maplist = get_atoms_maplist(Synthon2)
	Reactant1_maplist = get_atoms_maplist(Reactant1)
	Reactant2_maplist = get_atoms_maplist(Reactant2)
	
	S1_R1 = check_list_in_list(Synthon1_maplist,Reactant1_maplist)
	S1_R2 = check_list_in_list(Synthon1_maplist,Reactant2_maplist)
	S2_R1 = check_list_in_list(Synthon2_maplist,Reactant1_maplist)
	S2_R2 = check_list_in_list(Synthon2_maplist,Reactant2_maplist)
	#print(S1_R1,S1_R2,S2_R1,S2_R2)
	
	if Chem.MolToSmiles(Reactant1) == 'O=C(O)CN(CC(=O)O)C(=O)OCc1ccccc1':
		img = Draw.MolsToGridImage([Synthon1,Synthon2,Reactant1,Reactant2], molsPerRow=4,subImgSize=(1000,1000))
		plt.imshow(img)
		plt.show()

	if S1_R1 == True and S2_R2 == True :
		#print(1)
		return [Synthon1,Synthon2,Reactant1,Reactant2]
		
	elif S1_R2 == True and S2_R1 == True :
		#print(2)
		return [Synthon1,Synthon2,Reactant2,Reactant1]
		
	elif S1_R1 == False and S2_R2 == True:  
		#print(3)
		return [Synthon1,Synthon2,Reactant1,Reactant2]
	
	elif S1_R1 == True and S2_R2 == False:
		#print(4)
		return [Synthon1,Synthon2,Reactant1,Reactant2]
		
	elif S1_R2 == False and S2_R1 == True:
		#print(5)
		return [Synthon1,Synthon2,Reactant2,Reactant1]
		
	elif S1_R2 == True and S2_R1 == False:
		#print(6)
		return [Synthon1,Synthon2,Reactant2,Reactant1]

	else:
		print(Synthon1_maplist)
		print(Synthon2_maplist)
		print(Reactant1_maplist)
		print(Reactant2_maplist)
		mols = [Synthon1,Synthon2,Reactant1,Reactant2]
		img = Draw.MolsToGridImage(mols, molsPerRow=4,subImgSize=(500,500))
		plt.imshow(img)
		plt.show()
		print('匹配错误，请检查')
		return ['多数据']


class Classify(nn.Module):
	def __init__(self, in_features, out_features, bias=True):
		super(Classify, self).__init__()
		self.in_features = in_features
		self.out_features = out_features
		self.linear = nn.Linear(in_features, out_features)
		self.linear2 = nn.Linear(out_features, 1)
		self.use_bias = bias
		if self.use_bias:
			self.bias = nn.Parameter(torch.FloatTensor(out_features))
		self.reset_parameters()
		pass
	def reset_parameters(self):
		#nn.init.normal_(self.weight)
		#nn.init.normal_(self.weight2)
		#nn.init.kaiming_uniform_(self.weight)
		if self.use_bias:
			nn.init.zeros_(self.bias)
		pass
	def forward(self, input_features, Labels_Features_GCN):
		#print(input_features)
		output = self.linear(input_features)
		#output = torch.mm(output, self.weight2)
		#output = torch.sigmoid(output)
		#print(output.size())

		f = nn.Softmax(dim = 1)
		if self.use_bias: 
			output = output + self.bias
		#output = torch.mm(output, Labels_Features_GCN)
		#print('shape',output.shape)
		output = f(output)
		#print('shape2', output.shape)
		#output = self.linear2(output)
		#print('Model3_output:')
		#print(output)
		return output


'''
def Feature_Build(mol):  # 得到结构的特征向量
	Feature_list = []
	smi = Chem.MolToSmiles(mol)
	#Draw.ShowMol(mol, size=(500,500), kekulize=False)
	#Feature_string = MACCSkeys.GenMACCSKeys(x)
	#print(Feature_string)
	
	New_smi_Fra = ''
	for w in smi:
		if w == 'c':
			New_smi_Fra += 'C'
		elif w == 's':
			New_smi_Fra += 'S'
		elif w == 'n':
			New_smi_Fra += 'N'
		elif w == 'o':
			New_smi_Fra += 'O'
		elif w == 'p':
			New_smi_Fra += 'P'
		else:
			New_smi_Fra += w
	New_smi_Fra = New_smi_Fra.replace('[SN]','[Sn]')
	mol_ = Chem.MolFromSmiles(New_smi_Fra)
	Feature_string = AllChem.GetMorganFingerprintAsBitVect(mol_, 1, nBits=100).ToBitString()
	#Feature_string = AllChem.GetMACCSKeysFingerprint(mol_).ToBitString()
	for i in Feature_string:
		Feature_list.append(float(i))
	return Feature_list
'''

def SymbolAtom(mol):
	for i, atom in enumerate(mol.GetAtoms()):
		atom.SetProp("atomNote", str(i))
	return mol

def exchangedata(x,y):
	a = x
	x = y
	y = a
	return x,y
	

Mol_Xing = Chem.MolFromSmiles('*')

def Data_Processing(data,Labels_dict):   # 得到数据的标签，one-hot形式
	smi_P = data[0]
	smi_R1 = data[2]
	smi_R2 = data[3]
	Bond = data[1]
	mol_P = Chem.MolFromSmiles(smi_P)
	mol_P = SymbolAtom(mol_P)
	mol_R1 = Chem.MolFromSmiles(smi_R1)
	mol_R2 = Chem.MolFromSmiles(smi_R2)
	'''
	if smi_P == 'C=CC(=O)OCCCCCCOc1ccc(C(C)C(=O)O)cc1':
		Draw.ShowMol(mol_P, size=(1000,1000), kekulize=False)
		Draw.ShowMol(mol_R1, size=(1000,1000), kekulize=False)
		Draw.ShowMol(mol_R2, size=(1000,1000), kekulize=False)
	'''
	x,y = Bond.split(',')
	bond = mol_P.GetBondBetweenAtoms(int(x), int(y))
	if str(bond) == 'None':
		print("断裂键不存在,断裂位点标记错误")
		print(data)
		img = Draw.MolsToGridImage([mol_P,mol_R1,mol_R2], molsPerRow=3,subImgSize=(1000,1000))
		plt.title(data[1])
		plt.imshow(img)
		plt.show()
	
	Bond_indx = [bond.GetIdx()]
	Fragments = Chem.FragmentOnBonds(mol_P, Bond_indx, addDummies=False)  # 切断产物，生成两个合成子
	# print(smi_P,Chem.MolToSmiles(Fragments))

	if len(Chem.MolToSmiles(Fragments).split('.')) == 1:
		# ~ print(smi_P)
		print('分子切割出现错误，只有一个产物', Bond)
		img = Draw.MolsToGridImage([mol_P, mol_R1, mol_R2], molsPerRow=3,subImgSize=(1000, 1000))
		plt.title(data[1])
		plt.imshow(img)
		plt.show()

	mol_list = Chem.GetMolFrags(Fragments, asMols=True)
	smi1 = Chem.MolToSmiles(mol_list[0])
	smi2 = Chem.MolToSmiles(mol_list[1])

		
	Synthon1 = Chem.MolFromSmiles(smi1)
	Synthon2 = Chem.MolFromSmiles(smi2)
	
	result = GetRightMappingForSynthonAndReactant(Synthon1, Synthon2, mol_R1, mol_R2)
	if len(result) == 1:
		print("合成子在反应物中多次出现")
		print(data[0], data[1], data[2], data[3])
		# Draw.ShowMol(mol_P, size=(1000,1000), kekulize=False)
		img = Draw.MolsToGridImage([mol_P, mol_R1, mol_R2], molsPerRow=3, subImgSize=(1000, 1000))
		plt.title(data[1])
		plt.imshow(img)
		plt.show()

	Synthon_1, Synthon_2, Reactant_1, Reactant_2 = result[0], result[1], result[2], result[3]
	
	# ~ if smi_R1 == 'O=C(O)CN(CC(=O)O)C(=O)OCc1ccccc1':
		# ~ print('smi_R1:',smi_R1)
		# ~ print('smi1:',smi1)
		# ~ print('smi_R2:',smi_R2)
		# ~ print('smi2:',smi2)
		# ~ img = Draw.MolsToGridImage([Reactant_1, Synthon_1,Reactant_2, Synthon_2], molsPerRow=4,subImgSize=(1000,1000))
		# ~ plt.imshow(img)
		# ~ plt.show()
	
	# ~ extra_structure1 = Chem.DeleteSubstructs(Reactant_1, Synthon_1)
	# ~ extra_structure2 = Chem.DeleteSubstructs(Reactant_2, Synthon_2)
	extra_structure1 = get_LG(Reactant_1)
	extra_structure2 = get_LG(Reactant_2)
	
	smi_extra1 = Chem.MolToSmiles(extra_structure1)
	smi_extra2 = Chem.MolToSmiles(extra_structure2)
	
	if smi_extra1 == '':
		smi_extra1 = 'None'
	if smi_extra2 == '':
		smi_extra2 = 'None'
	# if smi_extra1=='O=S(=O)(O)C(F)(F)' or smi_extra2=='O=S(=O)(O)C(F)(F)':
	# 	print('------', smi_P)
	# 	return sdasasd
	# if smi_extra1 not in Labels_dict.keys():
	# 	print(data)
	# 	img = Draw.MolsToGridImage([mol_P,mol_R1,mol_R2], molsPerRow=3,subImgSize=(1000,1000))
	# 	plt.title(data[1])
	# 	plt.imshow(img)
	# 	plt.show()
	# if smi_extra2 not in Labels_dict.keys():
	# 	print(data)
	# 	img = Draw.MolsToGridImage([mol_P,mol_R1,mol_R2], molsPerRow=3,subImgSize=(1000,1000))
	# 	plt.title(data[1])
	# 	plt.imshow(img)
	# 	plt.show()
	Label_id1 = Labels_dict[smi_extra1]
	Label_id2 = Labels_dict[smi_extra2]
	return Label_id1, Label_id2
	















