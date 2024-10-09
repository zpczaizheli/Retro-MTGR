import csv
from rdkit import Chem
from rdkit.Chem import AllChem,Draw
import matplotlib.pyplot as plt
from PIL import Image
import os


def SymbolAtom(mol):
	Atomnum = mol.GetNumAtoms()
	for i in range(Atomnum):
		Atom = mol.GetAtomWithIdx(i)
		Atom.SetProp("atomNote", str(i))
	return mol
def check_numbers_in_lists(a, b, list1, list2):
    # 检查a和b是否分别出现在两个列表中
    a_in_list1 = a in list1 and b in list2
    b_in_list1 = b in list1 and a in list2
    
    # 输出True或False
    return a_in_list1 or b_in_list1
def get_map_list(reactant):
	list1 = []
	for atom in reactant.GetAtoms():
		map_num = atom.GetAtomMapNum()
		if map_num is not None:
			list1.append(map_num)
	return list1

BondEnergy_Table = {}
filename = 'test-MIT.txt'

data_list = []
with open(filename, 'r',encoding='UTF-8') as file_object:
	for line in file_object:
		line = line.strip()
		data_list.append(line)

print(len(data_list))
count = 0
count2 = 0
count3 = 0
Finally_Data = {}

Resultdata = []

for data in data_list[1:-1]:
	print(count)
	count += 1
	#reaction_class = data[0]
	reaction_smarts = data.split('	')[0]
	#print(reaction_smarts)
	reaction = AllChem.ReactionFromSmarts(reaction_smarts)
	reactants = reaction.GetReactants()
	products = reaction.GetProducts()
	product = products[0]
	if len(products) > 1:
		an = product.GetNumAtoms()
		for p in products[1:-1]:
			atomnum = p.GetNumAtoms()
			if atomnum > an:
				product = p
				an = atomnum


	if len(reaction.GetProducts()) > 1:
		print(reaction_smarts)
		for p in reaction.GetProducts():
			print(Chem.MolToSmiles(p))
	result = ''
	map_list_p = get_map_list(product)
	Real_reactants_list = []
	for react in reactants:
		map_list_r = get_map_list(react)
		set1 = set(map_list_r)
		set2 = set(map_list_p)
		common_elements = set1.intersection(set2)
		if common_elements:
			Real_reactants_list.append(react)
	if len(Real_reactants_list) == 1:
		count3 += 1
		ringnum_r = Chem.GetSSSR(reactants[0])
		ringnum_p = Chem.GetSSSR(product)
		result = Chem.MolToSmiles(product) + '	' + str(0) + ',' + str(1) + '	' + Chem.MolToSmiles(reactants[0]) + '	*'
		Resultdata.append(result)

	if len(Real_reactants_list) == 2:
		map_list1 = get_map_list(Real_reactants_list[0])
		map_list2 = get_map_list(Real_reactants_list[1])
		#list_pbond = []
		for bond in product.GetBonds():
			atom1_map = bond.GetBeginAtom().GetAtomMapNum()
			atom2_map = bond.GetEndAtom().GetAtomMapNum()
			if check_numbers_in_lists(atom1_map,atom2_map,map_list1,map_list2):

				ind1 = bond.GetBeginAtom().GetIdx()
				ind2 = bond.GetEndAtom().GetIdx()
				if ind1 > ind2 :
					a = ind1
					ind1 = ind2
					ind2 = a
				result = Chem.MolToSmiles(product) + '	' + str(ind1) + ',' + str(
					ind2) + '	' + Chem.MolToSmiles(
					Real_reactants_list[0]) + '	' + Chem.MolToSmiles(
					Real_reactants_list[1])
				Resultdata.append(result)
				break


txt = open('output-' + filename, 'w', encoding='UTF-8')
for i in Resultdata:
	txt.write('0	' + i + '\n')













