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

    a_in_list1 = a in list1 and b in list2
    b_in_list1 = b in list1 and a in list2

    return a_in_list1 or b_in_list1
def get_map_list(reactant):
	list1 = []
	for atom in reactant.GetAtoms():
		map_num = atom.GetAtomMapNum()
		if map_num is not None:
			list1.append(map_num)
	return list1

BondEnergy_Table = {}
filename = '50k-all.csv'

data_list = []
with open(filename, mode='r', newline='') as file:
	csv_reader = csv.reader(file)
	for row in csv_reader:
		data_list.append(row)

print(len(data_list))
count = 0
count2 = 0
count3 = 0
Finally_Data = {}
for i in range(1,11):
	Finally_Data[str(i)] = []

for data in data_list[1:-1]:
	#print(count2)
	count2 += 1
	reaction_class = data[0]
	reaction_smarts = data[2]
	#print(reaction_smarts)
	reaction = AllChem.ReactionFromSmarts(reaction_smarts)
	reactants = reaction.GetReactants()
	product = reaction.GetProducts()[0]
	if len(reaction.GetProducts()) > 1:
		print('xxxxxxxxxxxxxxx')
	result = ''
	if len(reactants) == 1:
		count3 += 1
		result = Chem.MolToSmiles(product) + '	' + str(0) + ',' + str(1) + '	' + Chem.MolToSmiles(reactants[0]) + '	*'
		Finally_Data[reaction_class].append(result)

	elif len(reactants) == 2:
		map_list1 = get_map_list(reactants[0])
		map_list2 = get_map_list(reactants[1])
		list_pbond = []
		for bond in product.GetBonds():
			atom1_map = bond.GetBeginAtom().GetAtomMapNum()
			atom2_map = bond.GetEndAtom().GetAtomMapNum()
			if check_numbers_in_lists(atom1_map,atom2_map,map_list1,map_list2):
				if int(bond.GetBeginAtom().GetIdx()) < int(bond.GetEndAtom().GetIdx()):
					result = Chem.MolToSmiles(product) + '	' + str(bond.GetBeginAtom().GetIdx()) + ',' + str(bond.GetEndAtom().GetIdx()) + '	' + Chem.MolToSmiles(reactants[0]) + '	' +Chem.MolToSmiles(reactants[1])
				else:
					result = Chem.MolToSmiles(product) + '	' + str(bond.GetEndAtom().GetIdx()) + ',' + str(
						bond.GetBeginAtom().GetIdx()) + '	' + Chem.MolToSmiles(
						reactants[0]) + '	' + Chem.MolToSmiles(reactants[1])
				Finally_Data[reaction_class].append(result)
				break
		
	else:
		combined_img = Image.new("RGB", (300 * (len(reactants)+1), 300))
		mol_img = Draw.MolToImage(product, size=(300, 300))
		combined_img.paste(mol_img, (len(reactants) * 300, 0))
		count += 1
		if product is None:
			print('sss')
		for i in range(len(reactants)):
			mol_img = Draw.MolToImage(reactants[i], size=(300, 300))
			combined_img.paste(mol_img, (i * 300, 0))


txt = open('output-50k.txt', 'w', encoding='UTF-8')
for i in range(1,11):
	print(len(Finally_Data[str(i)]))
	for data in Finally_Data[str(i)]:
		txt.write(str(i) + '	' + data + '\n')







