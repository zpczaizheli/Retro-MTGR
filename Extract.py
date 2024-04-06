
from rdkit import Chem
from rdkit.Chem import Draw
if_img = False

step = 2
def removeHs(smiles):
    atoms = smiles.split(']')
    smart = ''
    for atom in atoms:
        if len(atom.split(':')) < 2:
            if len(atom.split('[')) < 2:
                smart += atom

            if len(atom.split('[')) == 2:
                smart += atom + ']'
            continue

        w1, w2 = atom.split(':')
        num = 0
        if len(w1) < 3:
            for i in range(len(w1)-2, len(w1)):
                if w1[i] == 'H':
                    num = i
                    break
        else:
            for i in range(len(w1)-3, len(w1)):
                if w1[i] == 'H':
                    num = i
                    break
        if num != 0:
            smart += atom[0:num] + ':' + w2
        else:
            smart += atom
        smart += ']'
    return smart


def complete_ring(Index_list, smiles1):

    mol = Chem.MolFromSmiles(smiles1)
    mid_list = Index_list[:]
    aromatic_atoms = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetIsAromatic()]
    Ring_atom_index = []
    for index in mid_list:
        if index in aromatic_atoms:
            Ring_atom_index.append(index)

    ring_info = mol.GetRingInfo()
    num_rings = ring_info.NumRings()

    for index1 in Ring_atom_index:
        for index2 in range(mol.GetNumAtoms()):
            for i in range(num_rings):
                ring_atoms = ring_info.AtomRings()[i]
                if index1 in ring_atoms and index2 in ring_atoms:
                    mid_list.append(index2)
                    break
    return list(set(mid_list))


def SymbolAtom(mol):
    Atomnum = mol.GetNumAtoms()
    for i in range(Atomnum):
        Atom = mol.GetAtomWithIdx(i)
        Atom.SetProp("atomNote", str(i))
    return mol


def Get_changed_atoms_map(reaction):
    reactants,products = reaction.split('>>')
    Changed_Atoms_map = []
    for reactant in reactants.split('.'):

        Mol_reactant = Chem.MolFromSmiles(reactant)
        num_atoms_R = Mol_reactant.GetNumAtoms()
        for atom_idx in range(num_atoms_R):
            Condition_list_R = {}
            atom = Mol_reactant.GetAtomWithIdx(atom_idx)
            atom_map = atom.GetAtomMapNum()

            neighbor_atoms = atom.GetNeighbors()
            neighbor_map = []

            for neighbor_atom in neighbor_atoms:
                neighbor_map.append(neighbor_atom.GetAtomMapNum())
            Condition_list_R[atom_map] = neighbor_map

            x = False
            for product in products.split('.'):
                Mol_product = Chem.MolFromSmiles(product)
                num_atoms_P = Mol_product.GetNumAtoms()
                for i in range(num_atoms_P):
                    Condition_list_P = {}
                    atom_P = Mol_product.GetAtomWithIdx(i)
                    atom_map_P = atom_P.GetAtomMapNum()
                    neighbor_atoms_P = atom_P.GetNeighbors()
                    neighbor_map_P = []
                    for neighbor_atom_P in neighbor_atoms_P:
                        neighbor_map_P.append(neighbor_atom_P.GetAtomMapNum())
                    Condition_list_P[atom_map_P] = neighbor_map_P

                    if set(Condition_list_P[atom_map_P]) == set(Condition_list_R[atom_map]) and atom_map_P == atom_map:

                        x = True
                        break
                if x:
                    break
            if x == False:
                Changed_Atoms_map.append(atom_map)

    return Changed_Atoms_map


def Get_Reaction_template(reaction,step):
    Changed_Atoms_map = Get_changed_atoms_map(reaction)
    Save_atom_Index_list_R = []
    Save_atom_Index_list_P = []
    reactants, products = reaction.split('>>')
    tem = ''
    for reactant in reactants.split('.'):
        Mol_reactant = Chem.MolFromSmiles(reactant)

        if if_img:
            img = Draw.MolToImage(SymbolAtom(Mol_reactant))
            img.show()
        num_atoms_R = Mol_reactant.GetNumAtoms()
        Index_list = []
        for atom_idx in range(num_atoms_R):
            mid_list = []
            atom = Mol_reactant.GetAtomWithIdx(atom_idx)
            atom_map = atom.GetAtomMapNum()
            if atom_map in Changed_Atoms_map:

                mid_list.append(atom_idx)
                for i in range(step):
                    fin_list = mid_list[:]
                    for j in fin_list:
                        ato = Mol_reactant.GetAtomWithIdx(j)
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

        Index_list = complete_ring(Index_list, reactant)

        Save_atom_Index_list_R.append(Index_list)

    for product in products.split('.'):
        Mol_product = Chem.MolFromSmiles(product)
        if if_img:
            img = Draw.MolToImage(SymbolAtom(Mol_product))
            img.show()
        num_atoms_P = Mol_product.GetNumAtoms()
        Index_list = []
        for atom_idx in range(num_atoms_P):
            mid_list = []
            atom = Mol_product.GetAtomWithIdx(atom_idx)
            atom_map = atom.GetAtomMapNum()

            if atom_map == 0:
                Index_list.append(atom_idx)
                continue
            if atom_map in Changed_Atoms_map:

                mid_list.append(atom_idx)
                for i in range(step):
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
        Index_list = complete_ring(Index_list, product)
        Save_atom_Index_list_P.append(Index_list)

    Mol_Xing = Chem.MolFromSmiles('*')
    for i in range(len(reactants.split('.'))):
        save_index = Save_atom_Index_list_R[i]
        Mol_reactant = Chem.MolFromSmiles(reactants.split('.')[i])
        num_atoms_R = Mol_reactant.GetNumAtoms()
        mw = Chem.RWMol(Mol_reactant)
        for atom_idx in range(num_atoms_R):
            if atom_idx in save_index:
                continue
            else:
                mw.ReplaceAtom(atom_idx, Chem.Atom(0))
        m = mw.GetMol()
        Mm = Chem.DeleteSubstructs(m, Mol_Xing)

        if if_img:
            img = Draw.MolToImage(Mm)
            img.show()

        tem += removeHs(Chem.MolToSmiles(Mm))
        if i < len(reactants.split('.'))-1:
            tem += '.'
    tem +='>>'
    for i in range(len(products.split('.'))):
        save_index = Save_atom_Index_list_P[i]
        Mol_product = Chem.MolFromSmiles(products.split('.')[i])
        num_atoms_P = Mol_product.GetNumAtoms()
        mw = Chem.RWMol(Mol_product)
        for atom_idx in range(num_atoms_P):
            if atom_idx in save_index:
                continue
            else:
                mw.ReplaceAtom(atom_idx, Chem.Atom(0))
        m = mw.GetMol()

        Mm = Chem.DeleteSubstructs(m, Mol_Xing)

        if if_img:
            img = Draw.MolToImage(Mm)
            img.show()

        tem += removeHs(Chem.MolToSmiles(Mm))
        if i < len(products.split('.'))-1:
            tem += '.'
    return tem

def reaction_weather_equal(reaction1,reaction2):
    products1, reactants1 = reaction1.split('>>')
    products2, reactants2 = reaction2.split('>>')
    mol_products1 = Chem.MolFromSmarts(products1)
    mol_products2 = Chem.MolFromSmarts(products2)
    mol_reactants1 = Chem.MolFromSmarts(reactants1)
    mol_reactants2 = Chem.MolFromSmarts(reactants2)
    X = False
    if mol_products1 and mol_products2 and mol_reactants1 and mol_reactants2:
        if mol_products1.HasSubstructMatch(mol_products2) and mol_products2.HasSubstructMatch(mol_products1):
            if mol_reactants1.HasSubstructMatch(mol_reactants2) and mol_reactants2.HasSubstructMatch(mol_reactants1):
                X = True
    else:
        print(reaction1)
        print(reaction2)
    return X












