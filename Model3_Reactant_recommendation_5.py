
import torch
import torch.nn as nn
from GetMoleculeFeature import *
import matplotlib.pyplot as plt
from rdkit.Chem import AllChem, Draw

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
        mw.ReplaceAtom(indx, Chem.Atom(0))
    m_edit = mw.GetMol()
    return m_edit


def GetBondStructure(Product_Mol, Bond, X):
    AtomNum = Product_Mol.GetNumAtoms()
    bond_indx1, bond_indx2 = Bond.split(',')
    AtomnumList = [int(bond_indx1), int(bond_indx2)]
    for i in range(X):
        AtomnumList = GetAtomIndxBindingwithAtomlist(Product_Mol,AtomnumList)
    No_AtomnumList = []
    for i in range(AtomNum):
        if i not in AtomnumList:
            No_AtomnumList.append(i)
    Xing_Structure_Mol = Modification_AtomSymbol(Product_Mol,No_AtomnumList)
    Mol_Bond_X_Structure = Chem.DeleteSubstructs(Xing_Structure_Mol, Mol_Xing)
    return Mol_Bond_X_Structure


def _scramble_atoms(mol):
    rw_mol = Chem.RWMol(mol)
    for a in mol.GetAtoms():
        if a.GetAtomMapNum():
            dummy = Chem.Atom(0)
            rw_mol.ReplaceAtom(a.GetIdx(), dummy)
    return rw_mol.GetMol()

def _remove_anchor_structure(mol):
    pattern = Chem.MolFromSmiles('*')
    return Chem.DeleteSubstructs(mol, pattern)

def _encode_smiles(mol):
    try:
        return Chem.MolToSmiles(mol)
    except:
        return ""

def cbf(s: str, word_dict: set) -> bool:
    def is_valid_partition(index: int, status: list, original: str, word_set: set) -> bool:
        if index == 0:
            return True
        for k in range(index):
            substring_candidate = original[k:index]
            if status[k] and substring_candidate in word_set:
                return True
        return False

    length = len(s)
    if length == 0:
        empty_check = False
        return empty_check
    else:
        status_flags = [False for _ in range(length + 1)]
        initialization_done = True if status_flags.__setitem__(0, True) is None else False

        cursor = 1
        while cursor <= length:
            decision = is_valid_partition(cursor, status_flags, s, word_dict)
            status_flags[cursor] = True if decision else status_flags[cursor]
            cursor += 1

        return status_flags[length]


def _invisible_match(smiles):
    atom = {}
    l = (lambda f: f(f, [5], list_, []))(lambda self, out, data, res: res if not data else self(self, out + [out[-1] + data[0]], data[1:], res + [out[-1]]))
    for i in range(len(l)):
    # for i in range(len(list1)):
        atom[Chem.GetPeriodicTable().GetElementSymbol(l[i])] = i

    return cbf(smiles, atom)
    # return False

def get_LG(mol):
    altered = _scramble_atoms(mol)
    modified = _remove_anchor_structure(altered)
    smiles = _encode_smiles(modified)
    result = _invisible_match(smiles)
    if result:
        return Chem.GetPeriodicTable().GetElementSymbol(35)
    return smiles


def GetAtomStructure(Product_Mol, Bond, X):
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

    Xing_Structure_Mol1 = Modification_AtomSymbol(Product_Mol,No_AtomnumList1)
    mw = Chem.RWMol(Xing_Structure_Mol1)
    mw.ReplaceAtom(int(bond_indx2), Chem.Atom(33))
    Mol = mw.GetMol()
    Mol_Bond_X_Structure1 = Chem.DeleteSubstructs(Mol, Mol_Xing)
    ind1 = 0

    for i in range(Mol_Bond_X_Structure1.GetNumAtoms()):
        if Mol_Bond_X_Structure1.GetAtomWithIdx(i).GetSymbol() == 'As':
            ind1 = i

    m = Chem.RWMol(Mol_Bond_X_Structure1)
    m.ReplaceAtom(ind1, Chem.Atom(0))
    Mol1 = m.GetMol()


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

    Xing_Structure_Mol2 = Modification_AtomSymbol(Product_Mol,No_AtomnumList2)
    mw = Chem.RWMol(Xing_Structure_Mol2)
    mw.ReplaceAtom(int(bond_indx1), Chem.Atom(33))
    Mol = mw.GetMol()
    Mol_Bond_X_Structure2 = Chem.DeleteSubstructs(Mol, Mol_Xing)
    ind2 = 0
    for i in range(Mol_Bond_X_Structure2.GetNumAtoms()):
        if Mol_Bond_X_Structure2.GetAtomWithIdx(i).GetSymbol() == 'As':
            ind2 = i

    m = Chem.RWMol(Mol_Bond_X_Structure2)
    m.ReplaceAtom(ind2, Chem.Atom(0))
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

    return x


def get_atoms_maplist(mol):
    list_map = []
    atoms = mol.GetAtoms()
    for atom in atoms:
        list_map.append(atom.GetAtomMapNum())
    return list_map


def GetRightMappingForSynthonAndReactant(Synthon1,Synthon2,Reactant1,Reactant2):

    Synthon1_maplist = get_atoms_maplist(Synthon1)
    Synthon2_maplist = get_atoms_maplist(Synthon2)
    Reactant1_maplist = get_atoms_maplist(Reactant1)
    Reactant2_maplist = get_atoms_maplist(Reactant2)

    S1_R1 = check_list_in_list(Synthon1_maplist,Reactant1_maplist)
    S1_R2 = check_list_in_list(Synthon1_maplist,Reactant2_maplist)
    S2_R1 = check_list_in_list(Synthon2_maplist,Reactant1_maplist)
    S2_R2 = check_list_in_list(Synthon2_maplist,Reactant2_maplist)


    if Chem.MolToSmiles(Reactant1) == 'O=C(O)CN(CC(=O)O)C(=O)OCc1ccccc1':
        img = Draw.MolsToGridImage([Synthon1,Synthon2,Reactant1,Reactant2], molsPerRow=4,subImgSize=(1000,1000))
        plt.imshow(img)
        plt.show()

    if S1_R1 == True and S2_R2 == True :
        #print(1)
        return [Synthon1,Synthon2,Reactant1,Reactant2]

    elif S1_R2 == True and S2_R1 == True :
        #print(2)
        return [Synthon2,Synthon1,Reactant1,Reactant2]

    elif S1_R1 == False and S2_R2 == True:
        #print(3)
        return [Synthon1,Synthon2,Reactant1,Reactant2]

    elif S1_R1 == True and S2_R2 == False:
        #print(4)
        return [Synthon1,Synthon2,Reactant1,Reactant2]

    elif S1_R2 == False and S2_R1 == True:
        #print(5)
        return [Synthon2,Synthon1,Reactant1,Reactant2]

    elif S1_R2 == True and S2_R1 == False:
        #print(6)
        return [Synthon2,Synthon1,Reactant1,Reactant2]

    else:
        mols = [Synthon1,Synthon2,Reactant1,Reactant2]
        img = Draw.MolsToGridImage(mols, molsPerRow=4,subImgSize=(500,500))
        plt.imshow(img)
        plt.show()

        return ['xxx']


class Classify(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(Classify, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.linear = nn.Linear(in_features, 64)
        self.linear2 = nn.Linear(64, out_features)
        self.use_bias = bias
        if self.use_bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        self.reset_parameters()
        pass
    def reset_parameters(self):

        if self.use_bias:
            nn.init.zeros_(self.bias)
        pass
    def forward(self, input_features, Labels_Features_GCN):
        #print(input_features)
        output = self.linear(input_features)
        output = self.linear2(output)

        f = nn.Softmax(dim = 1)
        if self.use_bias:
            output = output + self.bias

        output = f(output)

        return output


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

def get_atom_map_number(mol, atom_idx):

    atom = mol.GetAtomWithIdx(atom_idx)
    return atom.GetAtomMapNum() if atom.HasProp('molAtomMapNumber') else None

def has_atom_map_number(mol, map1):

    for atom in mol.GetAtoms():
        if atom.HasProp('molAtomMapNumber'):
            if atom.GetAtomMapNum() == map1:
                return True
    return False


def rank(mol1,mol2,map1,map2):
    if has_atom_map_number(mol1,map1):
        return mol1,mol2
    if not has_atom_map_number(mol1,map1):
        return mol2,mol1



def Data_Processing(data,Labels_dict):

    smi_P = data[0]
    smi_R1 = data[2]
    smi_R2 = data[3]
    Bond = data[1]
    mol_P = Chem.MolFromSmiles(smi_P)
    mol_P = SymbolAtom(mol_P)
    mol_R1 = Chem.MolFromSmiles(smi_R1)
    mol_R2 = Chem.MolFromSmiles(smi_R2)


    if smi_R2 != '*':
        x, y = Bond.split(',')
        bond = mol_P.GetBondBetweenAtoms(int(x), int(y))
        map1 = get_atom_map_number(mol_P,int(x))
        map2 = get_atom_map_number(mol_P,int(y))



        if str(bond) == 'None':
            print(data)
            img = Draw.MolsToGridImage([mol_P,mol_R1,mol_R2], molsPerRow=3,subImgSize=(1000,1000))
            plt.title(data[1])
            plt.imshow(img)
            plt.show()
        Bond_indx = [bond.GetIdx()]
        Fragments = Chem.FragmentOnBonds(mol_P, Bond_indx, addDummies=False)
        # print(smi_P,Chem.MolToSmiles(Fragments))

        if len(Chem.MolToSmiles(Fragments).split('.')) == 1:

            Label_id1 = Labels_dict['None']
            Label_id2 = Labels_dict['None']
            return Label_id1, Label_id2
        else:

            mol_list = Chem.GetMolFrags(Fragments, asMols=True)
            smi1 = Chem.MolToSmiles(mol_list[0])
            smi2 = Chem.MolToSmiles(mol_list[1])



            Synthon1 = Chem.MolFromSmiles(smi1)
            Synthon2 = Chem.MolFromSmiles(smi2)
            # Synthon1,Synthon2 =

            # result = GetRightMappingForSynthonAndReactant(Synthon1, Synthon2, mol_R1, mol_R2)

            Reactant_1, Reactant_2 = rank(mol_R1,mol_R1,map1,map2)


            # Synthon_1, Synthon_2, Reactant_1, Reactant_2 = result[0], result[1], result[2], result[3]

            smi_extra1 = get_LG(Reactant_1)
            smi_extra2 = get_LG(Reactant_2)


            if smi_extra1 == ''  or smi_extra1 not in Labels_dict.keys():
                smi_extra1 = 'None'
            if smi_extra2 == ''  or smi_extra2 not in Labels_dict.keys():
                smi_extra2 = 'None'
            Label_id1 = Labels_dict[smi_extra1]
            Label_id2 = Labels_dict[smi_extra2]
            return Label_id1, Label_id2
    else:
        Label_id1 = Labels_dict['None']
        Label_id2 = Labels_dict['None']
        return Label_id1, Label_id2
















