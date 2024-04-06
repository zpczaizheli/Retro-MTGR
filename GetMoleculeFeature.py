from numpy import *
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import Draw
from Extract import *


BondEnergy_Table = {}
with open('data/USPT-50K/other_data/Bond_Energy.txt', 'r',encoding='UTF-8') as file_object:
	for line in file_object: 
		line = line.rstrip()
		BondName, Energy = line.split('	')
		BondEnergy_Table[BondName] = int(Energy)
AtomQuality_Table = {}
with open('data/USPT-50K/other_data/Atom_weight.txt', 'r',encoding='UTF-8') as file_object:
	for line in file_object: 
		line = line.rstrip()
		AtomName, Quality = line.split('	')
		AtomQuality_Table[AtomName] = float(Quality)

AtomSymbles = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'Al', 'I', 'B', 'K', 'Se', 'Zn','H', 'Cu', 'Mn', '*','unknown']
AtomSymble_One_Hot = {}
for i in range(len(AtomSymbles)):
	AtomSymble_One_Hot[AtomSymbles[i]] = [0]*i + [1] + [0]*(len(AtomSymbles)-i-1)



def GetAtomFeature(Atom_Rdkit):
	Feature = []
	Atom_Symbol = Atom_Rdkit.GetSymbol()   
	Degree = Atom_Rdkit.GetDegree()
	IsAromatic = Atom_Rdkit.GetIsAromatic()
	TotalNumHs = Atom_Rdkit.GetTotalNumHs()
	fc = Atom_Rdkit.GetFormalCharge()
	
	if Atom_Symbol not in AtomSymble_One_Hot.keys():
		Atom_Symbol = 'unknown'
	if str(IsAromatic) == 'False':
		IsAromatic = 0
	elif str(IsAromatic) == 'True':
		IsAromatic = 1
	Feature += AtomSymble_One_Hot[Atom_Symbol]
	Feature.append(TotalNumHs)
	Feature.append(Degree)
	Feature.append(IsAromatic)
	Feature.append(fc)
	Feature.append(AtomQuality_Table[Atom_Symbol])
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
				A_Row.append(1)
		Adj.append(A_Row)
	return Adj


def GetLabelForBondsFromMoleculeSmiles(Smiles,BreakBondID):
	Labels = []
	BondId = []
	BondId.append(BreakBondID)
	M_Mol = Chem.MolFromSmiles(Smiles)
	AtomsNum = M_Mol.GetNumAtoms()
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

	
def Getsubstructure(smiles, bond, Teamplate_step):
	Mol_product = Chem.MolFromSmiles(smiles)
	num_atoms_P = Mol_product.GetNumAtoms()
	Index_list = []
	B= [int(i) for i in bond.split(',')]
	for atom_idx in range(num_atoms_P):
		mid_list = []
		if atom_idx in B:
			mid_list.append(atom_idx)
			for i in range(Teamplate_step):
				fin_list = mid_list[:]
				for j in fin_list:
					ato = Mol_product.GetAtomWithIdx(j)
					neighbor_atoms = ato.GetNeighbors()
					neighbor_index_list = []
					for neighbor_atom in neighbor_atoms:
						neighbor_index_list.append(neighbor_atom.GetIdx())
					mid_list += neighbor_index_list
			mid_list = list(set(mid_list))
		else:
			continue
		Index_list += mid_list
		Index_list = list(set(Index_list))
	Index_list = complete_ring(Index_list, smiles)
	save_index = Index_list[:]
	Mol_Xing = Chem.MolFromSmiles('*')
	mw = Chem.RWMol(Mol_product)
	for atom_idx in range(num_atoms_P):
		if atom_idx in save_index:
			continue
		else:
			mw.ReplaceAtom(atom_idx, Chem.Atom(0))
	m = mw.GetMol()
	Mm = Chem.DeleteSubstructs(m, Mol_Xing)
	S = removeHs(Chem.MolToSmiles(Mm))
	return S

	
	
	
	
	
	
	
	
	
	
	
	
	
