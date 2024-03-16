from numpy import *
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import Draw



#  键能表
BondEnergy_Table = {}
with open('data/USPT-50K/other_data/键能表.txt', 'r',encoding='UTF-8') as file_object:   
	for line in file_object: 
		line = line.rstrip()
		BondName, Energy = line.split('	')
		BondEnergy_Table[BondName] = int(Energy)
AtomQuality_Table = {}
with open('data/USPT-50K/other_data/原子质量表.txt', 'r',encoding='UTF-8') as file_object:   
	for line in file_object: 
		line = line.rstrip()
		AtomName, Quality = line.split('	')
		AtomQuality_Table[AtomName] = float(Quality)

# 构建原子符号的one-hot编码对照表
AtomSymbles = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'Al', 'I', 'B', 'K', 'Se', 'Zn','H', 'Cu', 'Mn', '*','unknown']
AtomSymble_One_Hot = {}
for i in range(len(AtomSymbles)):
	AtomSymble_One_Hot[AtomSymbles[i]] = [0]*i + [1] + [0]*(len(AtomSymbles)-i-1)



def GetAtomFeature(Atom_Rdkit):
	Feature = []
	Atom_Symbol = Atom_Rdkit.GetSymbol()   
	Degree = Atom_Rdkit.GetDegree()  # 该原子的度
	IsAromatic = Atom_Rdkit.GetIsAromatic()  # 是否是芳香环
	TotalNumHs = Atom_Rdkit.GetTotalNumHs()  # 与该原子连接的氢原子个数
	fc = Atom_Rdkit.GetFormalCharge()   # 该原子所带电荷数(1维)
	
	if Atom_Symbol not in AtomSymble_One_Hot.keys():
		Atom_Symbol = 'unknown'
	if str(IsAromatic) == 'False':
		IsAromatic = 0
	elif str(IsAromatic) == 'True':
		IsAromatic = 1
	Feature += AtomSymble_One_Hot[Atom_Symbol]   # 用one-hot编码表示原子的符号 (23维)
	Feature.append(TotalNumHs)    # 与该原子连接的氢原子个数(1维)
	Feature.append(Degree)  # 该原子的度(1维)
	Feature.append(IsAromatic)  # 该原子是否是芳香环上的原子(1维)
	Feature.append(fc)  # 该原子所带电荷数(1维)
	Feature.append(AtomQuality_Table[Atom_Symbol])  # 该原子的原子质量(1维)
	return Feature
	
def GetBondenergy(M_Mol,i,j):
	atom1_Symbol = M_Mol.GetAtomWithIdx(i).GetSymbol()
	atom2_Symbol = M_Mol.GetAtomWithIdx(j).GetSymbol()
	BondType = M_Mol.GetBondBetweenAtoms(i,j).GetBondType()
	BondSymbol = ''
	if str(BondType) == 'SINGLE':
		BondSymbol = '-'

	elif str(BondType) == 'DOUBLE':
		BondSymbol = '='

	elif str(BondType) == 'TRIPLE':
		BondSymbol = '#'

	elif str(BondType) == 'AROMATIC':
		BondSymbol = '~'
	else:
		print('其他键类型')
		
	if atom1_Symbol == '*' or atom1_Symbol == '*':
		BondEnergy_Table[atom1_Symbol + BondSymbol + atom2_Symbol] = '0'
		BondEnergy_Table[atom2_Symbol + BondSymbol + atom1_Symbol] = '0'
	Bond1 = atom1_Symbol + BondSymbol + atom2_Symbol
	Bond2 = atom2_Symbol + BondSymbol + atom1_Symbol
	if Bond1 in BondEnergy_Table.keys():
		BondEnergy = BondEnergy_Table[Bond1]
	else:
		BondEnergy = BondEnergy_Table[Bond2]
	return BondEnergy



def GetFeatureForMoleculeFromSmiles(Smiles): 
	M_Features = []
	M_Mol = Chem.MolFromSmiles(Smiles)
	AtomsNum = M_Mol.GetNumAtoms()
	for i in range(AtomsNum):   
		atom_rdkit = M_Mol.GetAtomWithIdx(i)
		F = GetAtomFeature(atom_rdkit)
		M_Features.append(F)
	return M_Features

def GetAdjForMoleculeFromSmiles(Smiles):
	Adj = []
	M_Mol = Chem.MolFromSmiles(Smiles)
	AtomsNum = M_Mol.GetNumAtoms()
	for i in range(AtomsNum):   
		A_Row = [] 
		for j in range(AtomsNum):
			
			if str(M_Mol.GetBondBetweenAtoms(i,j)) == 'None':
				if i == j:
					A_Row.append(1)
				else:
					A_Row.append(0)
				continue
			else:
				#A_Row.append(float(GetBondenergy(M_Mol,i,j)))
				A_Row.append(1)
		Adj.append(A_Row)
	return Adj


def GetLabelForBondsFromMoleculeSmiles(Smiles,BreakBondID):
	Labels = []
	BondId = []
	BondId.append(BreakBondID)
	M_Mol = Chem.MolFromSmiles(Smiles)
	AtomsNum = M_Mol.GetNumAtoms()
	#print(Smiles)
	#print(BreakBondID)
	id1, id2 = BreakBondID.split(',')
	for num1 in range(AtomsNum):
		for num2 in range(num1+1, AtomsNum):
			
			if  str(num1) == id1 and str(num2) == id2:
				Labels.append(1)
			else:
				if str(M_Mol.GetBondBetweenAtoms(num1,num2)) == 'None':
					continue
				else:
					
					IsAromatic_1 = M_Mol.GetAtomWithIdx(num1).GetIsAromatic()
					IsAromatic_2 = M_Mol.GetAtomWithIdx(num2).GetIsAromatic()

					#if str(IsAromatic_1) == 'True'  and str(IsAromatic_2) == 'True':
					Labels.append(0)
					BondId.append(str(num1)+','+str(num2))
	return Labels,BondId


def GetAtomDegreeListMoleculeSmiles(Smiles):
	Degree_list = []
	M_Mol = Chem.MolFromSmiles(Smiles)
	AtomsNum = M_Mol.GetNumAtoms()
	for num1 in range(AtomsNum):
		Atom_Rdkit = M_Mol.GetAtomWithIdx(num1)
		Degree = Atom_Rdkit.GetDegree()
		Degree_list.append(Degree)
	return Degree_list
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
