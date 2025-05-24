import torch
import numpy as np
#from torch_geometric.graphgym import GATConv
from Model3_Reactant_recommendation_5 import *
torch.autograd.set_detect_anomaly(True)
from Extract import *



def get_BondtypeOneHot(bond):
	BondType = bond.GetBondType()
	if str(BondType) == 'SINGLE':
		return torch.Tensor([1, 0, 0, 0]).to(torch.float32)
	elif str(BondType) == 'DOUBLE':
		return torch.Tensor([0, 1, 0, 0]).to(torch.float32)
	elif str(BondType) == 'TRIPLE':
		return torch.Tensor([0, 0, 1, 0]).to(torch.float32)
	elif str(BondType) == 'AROMATIC':
		return torch.Tensor([0, 0, 0, 1]).to(torch.float32)
	else:
		print('键类型错误')



def normalize_Adj(mx_list):    
	mx = np.matrix(mx_list)
	max_ = mx.max()
	sum_ = 0
	#for x in mx_list:
	#	sum_ += sum(x)
	if max_ == 0:
		print('###################################################################################')
	return (mx/max_).tolist()
	#return (mx/sum_).tolist()


def normalize_Features(F_mx):
	result_T = []
	mx = np.matrix(F_mx)
	mx_T_list = mx.T.tolist()
	for row in mx_T_list:
		max_ = max(row)
		min_ = min(row)
		if max_ == 0:
			result_T.append([x for x in row])
		else:
			result_T.append([x/max_ for x in row])
	result_T = np.matrix(result_T)
	result = result_T.T.tolist()
	return result

def getbondneighbortype(mol,i):
	atomnum = mol.GetNumAtoms()
	bondtype = []
	count = 0
	for j in range(atomnum):
		bond = mol.GetBondBetweenAtoms(i, j)
		if str(bond) == 'None':
			continue
		count +=1
		bondtype.append(get_BondtypeOneHot(bond))
	zero_tensor = torch.zeros(4)
	for tensor in bondtype:
		zero_tensor += tensor
	return zero_tensor/count



def accuracy(output, labels):
	output = output.tolist()
	labels = labels.tolist()
	pred = output.max(1)[1].type_as(labels)
	correct = pred.eq(labels).double()
	correct = correct.sum()
	return correct / len(labels),pred

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
	sparse_mx = sparse_mx.tocoo().astype(np.float32)
	indices = torch.from_numpy(
	np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
	values = torch.from_numpy(sparse_mx.data)
	shape = torch.Size(sparse_mx.shape)
	return torch.sparse.FloatTensor(indices, values, shape)


def GetBondenergy_margin(Adj,i,j): 
	Neighbor_Bond_i = [] 
	Neighbor_Bond_j = [] 
	for num in range(len(Adj[i])):
		if num == j:
			continue
		if Adj[i][num] == 0:
			continue
		Neighbor_Bond_i.append(Adj[i][num])
	for num2 in range(len(Adj[j])):
		if num2 == i:
			continue
		if Adj[j][num2] == 0:
			continue
		Neighbor_Bond_j.append(Adj[j][num2])
	if len(Neighbor_Bond_i) == 0:
		Neighbor_Bond_i.append(Adj[i][j])
	if len(Neighbor_Bond_j) == 0:
		Neighbor_Bond_j.append(Adj[i][j])
	return max(Neighbor_Bond_i)-Adj[i][j]+0.001,max(Neighbor_Bond_j)-Adj[i][j]+0.001
	


def load_data(Datapath):
	Smiles_P_dict = {} #
	Smiles_R1_dict = {} #
	Smiles_R2_dict = {}
	BreakBondID_dict = {} #
	Feature_Dict_P = {} #
	Feature_Dict_R1 = {} #
	Feature_Dict_R2 = {} #
	Adj_Dic_P = {} #
	Adj_Dic_R1 = {} #
	Adj_Dic_R2 = {} #
	Reaction_class_dict = {}
	count = 0
	with open(Datapath, 'r',encoding='UTF-8') as file_object:   
		for line in file_object:
			line = line.rstrip()
			words = line.split('	')
			reaction_class = words[0]
			Smiles_P = words[1]
			Smiles_R1 = words[3]
			Smiles_R2 = words[4]
			
			BreakBondID = words[2]
			if BreakBondID == 'None':
				continue
			if words[-1] == 'None':
				continue
			Smiles_P_dict[str(count)] = Smiles_P
			Smiles_R1_dict[str(count)] = Smiles_R1
			Smiles_R2_dict[str(count)] = Smiles_R2
			BreakBondID_dict[str(count)] = BreakBondID
			Reaction_class_dict[str(count)] = reaction_class
			count += 1
	
	Adj_Dic_P_Normal = {}
	for num in range(count):
		# print(num)
		# BreakBondID = BreakBondID_dict[str(num)]
		Smi_P = Smiles_P_dict[str(num)]  
		Smi_R1 = Smiles_R1_dict[str(num)]
		Smi_R2 = Smiles_R2_dict[str(num)]
		# print(Smi_P)
		features_P = GetFeatureForMoleculeFromSmiles(Smi_P)
		features_R1 = GetFeatureForMoleculeFromSmiles(Smi_R1)
		features_R2 = GetFeatureForMoleculeFromSmiles(Smi_R2)
		
		Adj_P = GetAdjForMoleculeFromSmiles(Smi_P)
		Adj_R1 = GetAdjForMoleculeFromSmiles(Smi_R1)
		Adj_R2 = GetAdjForMoleculeFromSmiles(Smi_R2)
		
		features_P = torch.FloatTensor(normalize_Features(features_P))
		features_R1 = torch.FloatTensor(normalize_Features(features_R1))
		features_R2 = torch.FloatTensor(normalize_Features(features_R2))
		
		
		Adj_P_Normal = torch.FloatTensor(normalize_Adj(Adj_P))
		Adj_R1 = torch.FloatTensor(normalize_Adj(Adj_R1))
		Adj_R2 = torch.FloatTensor(normalize_Adj(Adj_R2))
		
		Feature_Dict_P[str(num)] = features_P  
		Feature_Dict_R1[str(num)] = features_R1  
		Feature_Dict_R2[str(num)] = features_R2  
		
		Adj_Dic_P[str(num)]  = Adj_P  
		Adj_Dic_R1[str(num)]  = Adj_R1  
		Adj_Dic_R2[str(num)]  = Adj_R2  
		Adj_Dic_P_Normal[str(num)] = torch.FloatTensor(Adj_P_Normal)
	Result = [Smiles_P_dict, Smiles_R1_dict, Smiles_R2_dict, BreakBondID_dict,
			Feature_Dict_P,Feature_Dict_R1, Feature_Dict_R2,Adj_Dic_P, 
			Adj_Dic_R1,Adj_Dic_R2,Adj_Dic_P_Normal,Reaction_class_dict]
	return Result




class GraphConvolution(nn.Module):

	def __init__(self, in_features, out_features, device, bias=False):
		super(GraphConvolution, self).__init__()
		self.weight = nn.Linear(in_features, out_features).to(device)
		self.use_bias = bias
		if self.use_bias:
			self.bias = nn.Parameter(torch.FloatTensor(out_features)).to(device)
		self.reset_parameters()

	def reset_parameters(self):
		if self.use_bias:
			nn.init.zeros_(self.bias)
		pass

	def forward(self, input_features, adj):
		support = self.weight(input_features)
		output = torch.mm(adj, support)
		output = torch.tanh(output)
		output = output.clone()
		if self.use_bias:
			return output + self.bias
		else:
			return output

class GraphConvolution2(nn.Module):
	def __init__(self, in_features, out_features, device, bias=True):
		super(GraphConvolution2, self).__init__()
		self.weight = nn.Linear(in_features, out_features).to(device)
		self.eye = torch.eye(in_features)
		self.use_bias = bias
		if self.use_bias:
			self.bias = nn.Parameter(torch.FloatTensor(out_features)).to(device)
		self.reset_parameters()

	def reset_parameters(self):
		if self.use_bias:
			nn.init.zeros_(self.bias)
		pass

	def forward(self, input_features):
		support = self.weight(input_features)
		output = torch.mm(support,self.eye)
		output = torch.tanh(support)
		if self.use_bias:
			return output + self.bias
		else:
			return output


def get_Atom_EP(M_Mol,x,EP_list,y):
	atomsnum = M_Mol.GetNumAtoms()
	ep = 0
	smiles_atomx = M_Mol.GetAtomWithIdx(x).GetSymbol()
	if smiles_atomx not in EP_list.keys():
		smiles_atomx = 'unknown'
	ep_atomx = EP_list[smiles_atomx]
	for i in range(atomsnum):
		if i == x:
			continue
		if i == y:
			continue
		if str(M_Mol.GetBondBetweenAtoms(i, x)) == 'None':
			continue
		smiles_atomi = M_Mol.GetAtomWithIdx(i).GetSymbol()
		if smiles_atomi not in EP_list.keys():
			smiles_atomi = 'unknown'
		ep_atomi = EP_list[smiles_atomi]
		ep += ep_atomx - ep_atomi
	return ep



class RedOut(nn.Module):
	def __init__(self, device,input_dim=29):
		super(RedOut, self).__init__()
		self.gcn1 = GraphConvolution(input_dim, 64,device)
		self.gcn2 = GraphConvolution(64, input_dim,device)
		self.gcn3 = GraphConvolution(input_dim, 64,device)
		self.gcn4 = GraphConvolution(64, input_dim,device)
		self.gcn5 = GraphConvolution(input_dim, 64,device)
		self.gcn6 = GraphConvolution(64, input_dim,device)
		self.gcn7 = GraphConvolution2(input_dim, input_dim,device)
		self.weight_BondEnergy = nn.Linear(1, 1)
		#self.weight_FC = nn.Parameter(torch.FloatTensor(29,1)).to(device)
		self.weight_FC = nn.Linear(input_dim + 34, 1)
		self.weight_BondType= nn.Linear(4, 4)
		self.weight_BondType2= nn.Linear(4, 4)
		self.liner = nn.Linear(1, 1)
		#self.fc_2 = nn.Linear(28, 4).to(device)
		#self.reset_parameters()
		pass
	
	def reset_parameters(self):
		#nn.init.normal_(self.weight_FC)
		#nn.init.normal_(self.weight_BondEnergy)
		#nn.init.kaiming_uniform_(self.weight_BondEnergy)
		#nn.init.kaiming_uniform_(self.weight_FC)
		pass

	def Mpnn1(self,Feature,adj):
		out = torch.relu(self.gcn1(Feature, adj))
		out = self.gcn2(out, adj)
		#out = self.gat1(Feature,adj)
		#out = self.gcn7(Feature, adj)
		return out

	def Mpnn2(self,Feature_R1,Feature_R2,adj_R1,adj_R2):
		Feature = torch.cat([Feature_R1,Feature_R2], dim = 0)
		len_R1 = len(adj_R1)
		len_R2 = len(adj_R2)
		adj = []
		for i in adj_R1:
			R1 = torch.cat([i,torch.zeros(len_R2)],dim = 0)
			adj.append(R1)
		for j in adj_R2:
			R2 = torch.cat([torch.zeros(len_R1),j],dim = 0)
			adj.append(R2)
		adj = torch.stack(adj)
		out = torch.relu(self.gcn3(Feature, adj))
		out = self.gcn4(out, adj)

		return out

	def Mpnn3(self,Feature,adj):
		out = torch.relu(self.gcn5(Feature, adj))
		out = self.gcn6(out, adj)
		return out
	
	def Redout_P_Bond_Train(self, output, Smiles, BondID, adj):
		output_RedOut = []
		Labels = []
		Bond_list = []
		M_Mol = Chem.MolFromSmiles(Smiles)
		Positive_bondfeature = {}
		for i in range(output.shape[0]):
			for j in range(i+1,output.shape[0]):
				if str(M_Mol.GetBondBetweenAtoms(i, j)) == 'None':
					continue

				add = output[i] + output[j]
				#add = torch.cat((output[i],output[j]),dim = 0)
				BondEnergy = self.weight_BondEnergy(torch.Tensor([adj[i, j]]).to(device))
				output_RedOut.append(torch.cat((add, BondEnergy), dim=0))
				#output_RedOut.append(add)
				# Positive_bondfeature[str(i)+','+str(j)] = torch.cat((add,BondEnergy), dim=0)
				Bond_list.append(str(i)+','+str(j))

				id1, id2 = BondID.split(',')
				if str(i) == id1 and str(j) == id2:
					Labels.append(1) 
					Positive_bondfeature[id1] = torch.stack([output[i]])
					Positive_bondfeature[id2] = torch.stack([output[j]])
					
				else:
					Labels.append(0)

		output_RedOut = torch.stack(output_RedOut)
		outputs_list = output_RedOut.tolist()
		output_RedOut = self.weight_FC(output_RedOut) 
		
		output_RedOut = output_RedOut.reshape(-1)
		Labels = torch.tensor(Labels).to(torch.float32)
		output_RedOut = torch.sigmoid(output_RedOut)
		return Positive_bondfeature, output_RedOut, Labels,Bond_list,outputs_list
	
	def Redout_P_Bond_Test(self,output,Smiles,BondID,adj,AtomSymble_One_Hot,EP_list):
		output_RedOut = []
		Labels = []
		Bond_list = []
		M_Mol = Chem.MolFromSmiles(Smiles)
		Positive_bondfeature = {}
		Atom_symbol = []
		Bond_symbol = []
		#print(Smiles)
		#print(output.tolist())
		BE_list = []
		for i in range(output.shape[0]):
			for j in range(i + 1, output.shape[0]):
				if str(M_Mol.GetBondBetweenAtoms(i, j)) == 'None':
					continue
				BE_list.append(GetBondenergy(M_Mol, i, j))
		max_BE = max(BE_list)
		# print(type(max_BE))
		a,b = BondID.split(',')
		if a==b:
			print('yyyyyyyy')
			i = a
			j = b
			X = 1
			# Mol_Bond_X_Structure = GetBondStructure(M_Mol, BondID, X)
			# smi = Chem.MolToSmiles(Mol_Bond_X_Structure)
			# Bond_symbol.append(smi)
			#
			# Mol_Bond_X_Structure1, Mol_Bond_X_Structure2 = GetAtomStructure(M_Mol, str(i) + ',' + str(j), X)
			# smi1 = Chem.MolToSmiles(Mol_Bond_X_Structure1)
			# smi2 = Chem.MolToSmiles(Mol_Bond_X_Structure2)
			# Atom_symbol.append([smi1, smi2])

			atom1_symbol = M_Mol.GetAtomWithIdx(i).GetSymbol()
			if atom1_symbol not in AtomSymble_One_Hot.keys():
				atom1_symbol = 'unknown'
			atom1_feature = torch.Tensor(AtomSymble_One_Hot[atom1_symbol]).to(torch.float32)

			atom2_symbol = M_Mol.GetAtomWithIdx(j).GetSymbol()
			if atom2_symbol not in AtomSymble_One_Hot.keys():
				atom2_symbol = 'unknown'
			atom2_feature = torch.Tensor(AtomSymble_One_Hot[atom2_symbol]).to(torch.float32)

			bond_type_OneHot = self.weight_BondType(get_BondtypeOneHot(M_Mol.GetBondBetweenAtoms(0, 1)))

			Nighbors_BondType_Feature = self.weight_BondType2(getbondneighbortype(M_Mol, i) + getbondneighbortype(M_Mol, j))
			#
			EP_BondXY = self.liner(
				torch.Tensor([abs(get_Atom_EP(M_Mol, i, EP_list, j) - get_Atom_EP(M_Mol, j, EP_list, i))]).to(
					torch.float32))
			add = torch.cat((atom1_feature, output[i]), dim=0) + torch.cat((atom2_feature, output[j]), dim=0)
			add = torch.cat((bond_type_OneHot, add), dim=0)
			add = torch.cat((add, Nighbors_BondType_Feature), dim=0)
			#
			add = torch.cat((add, EP_BondXY), dim=0)

			BondEnergy = self.weight_BondEnergy(torch.Tensor([GetBondenergy(M_Mol, i, 1) / max_BE]).to(device))

			output_RedOut.append(torch.cat((add, BondEnergy), dim=0))

			Bond_list.append(str(i) + ',' + str(j))

			id1, id2 = BondID.split(',')

			if str(i) == id1 and str(j) == id2:
				Labels.append(1)
				Positive_bondfeature[id1] = torch.stack(
					[torch.cat((output[i], torch.cat((add, BondEnergy), dim=0)), dim=0)])
				Positive_bondfeature[id2] = torch.stack(
					[torch.cat((output[j], torch.cat((add, BondEnergy), dim=0)), dim=0)])
			else:
				Labels.append(0)
		else:
			for i in range(output.shape[0]):
				#print(output[i].tolist())
				for j in range(output.shape[0]):
					if str(M_Mol.GetBondBetweenAtoms(i, j)) == 'None':
						continue
					bond_type = M_Mol.GetBondBetweenAtoms(i, j).IsInRing()
					if bond_type == True and str(i)+',' +str(j) != BondID:
						continue
					X = 1
					Mol_Bond_X_Structure = GetBondStructure(M_Mol, BondID, X)
					smi = Chem.MolToSmiles(Mol_Bond_X_Structure)
					Bond_symbol.append(smi)

					Mol_Bond_X_Structure1, Mol_Bond_X_Structure2 = GetAtomStructure(M_Mol, str(i)+','+str(j), X)
					smi1 = Chem.MolToSmiles(Mol_Bond_X_Structure1)
					smi2 = Chem.MolToSmiles(Mol_Bond_X_Structure2)
					Atom_symbol.append([smi1,smi2])


					atom1_symbol = M_Mol.GetAtomWithIdx(i).GetSymbol()
					if atom1_symbol not in AtomSymble_One_Hot.keys():
						atom1_symbol = 'unknown'
					atom1_feature = torch.Tensor(AtomSymble_One_Hot[atom1_symbol]).to(torch.float32)

					atom2_symbol = M_Mol.GetAtomWithIdx(j).GetSymbol()
					if atom2_symbol not in AtomSymble_One_Hot.keys():
						atom2_symbol = 'unknown'
					atom2_feature = torch.Tensor(AtomSymble_One_Hot[atom2_symbol]).to(torch.float32)

					bond_type_OneHot = self.weight_BondType(get_BondtypeOneHot(M_Mol.GetBondBetweenAtoms(i, j)))

					Nighbors_BondType_Feature = self.weight_BondType2(getbondneighbortype(M_Mol, i) + getbondneighbortype(M_Mol, j))
					#
					EP_BondXY = self.liner(torch.Tensor([abs(get_Atom_EP(M_Mol, i, EP_list, j) - get_Atom_EP(M_Mol, j, EP_list, i))]).to(torch.float32))
					add = torch.cat((atom1_feature, output[i]), dim=0) + torch.cat((atom2_feature, output[j]), dim=0)
					add = torch.cat((bond_type_OneHot,add),dim=0)
					add = torch.cat((add, Nighbors_BondType_Feature),dim=0)
					#
					add = torch.cat((add, EP_BondXY), dim=0)

					BondEnergy = self.weight_BondEnergy(torch.Tensor([GetBondenergy(M_Mol, i, j)/max_BE]).to(device))

					output_RedOut.append(torch.cat((add, BondEnergy), dim=0))

					Bond_list.append(str(i) + ',' + str(j))

					id1, id2 = BondID.split(',')
					# print("qqqqqqqqqqqqqqqqqqqqqq")
					# print(torch.stack([torch.cat((output[i], torch.cat((add, BondEnergy), dim=0)), dim=0)]).shape)
					# print(output[i].shape)



					if str(i) == id1 and str(j) == id2:
						Labels.append(1)
						# Positive_bondfeature[id1] = torch.stack([torch.cat((output[i], torch.cat((add, BondEnergy), dim=0)), dim=0)])
						# Positive_bondfeature[id2] = torch.stack([torch.cat((output[j], torch.cat((add, BondEnergy), dim=0)), dim=0)])
						Positive_bondfeature[id1] = torch.cat((atom1_feature, output[i]), dim=0)
						Positive_bondfeature[id2] = torch.cat((atom2_feature, output[j]), dim=0)
					# if str(i) == id1 and str(j) == id2:
					# 	Labels.append(1)
					# 	Positive_bondfeature[id1] = atom1_feature.unsqueeze(0)
					# 	Positive_bondfeature[id2] = atom2_feature.unsqueeze(0)
					else:
						Labels.append(0)

		output_RedOut = torch.stack(output_RedOut)
		outputs_list = output_RedOut.tolist()
		output_RedOut = self.weight_FC(output_RedOut) 
		
		output_RedOut = output_RedOut.reshape(-1)
		Labels = torch.tensor(Labels).to(torch.float32)
		output_RedOut = torch.sigmoid(output_RedOut)
		return Positive_bondfeature, output_RedOut, Labels,Bond_list,outputs_list,Atom_symbol,Bond_symbol

	def Pooling(self, Output):
		l = len(Output)
		result = Output[0]
		for i in range(1,l):
			result += Output[i]

		return torch.div(result,l)
	
	def forward(self,EP_list, AtomSymble_One_Hot, Feature_P, adj_P, BreakBondID,Smiles_P, Feature_R1 = 'None',Feature_R2 = 'None',adj_R1 = 'None',adj_R2 = 'None', Neg_Feature_P = 'None', Neg_adj_P = 'None'):
		if Feature_R1 == 'None':
			Atom_output_P = self.Mpnn1(Feature_P,adj_P)
			# print(Atom_output_P.shape)
			Positive_bond_feature, output_Bond_RedOut_P, Label_P,Bond_list,outputs_list,Atom_symbol,Bond_symbol = self.Redout_P_Bond_Test(Atom_output_P,Smiles_P,BreakBondID,adj_P,AtomSymble_One_Hot,EP_list)  # 输出目标分子键的特征和标签

			Atom_feature_id1 = torch.cat((Positive_bond_feature[BreakBondID.split(',')[0]], Atom_output_P[int(BreakBondID.split(',')[0])]), dim=0).unsqueeze(0)
			Atom_feature_id2 = torch.cat((Positive_bond_feature[BreakBondID.split(',')[1]], Atom_output_P[int(BreakBondID.split(',')[1])]), dim=0).unsqueeze(0)


			return Atom_feature_id1, Atom_feature_id2, output_Bond_RedOut_P,Label_P, Bond_list,outputs_list,Atom_symbol,Bond_symbol
		else:
			Atom_output_R = self.Mpnn2(Feature_R1,Feature_R2,adj_R1,adj_R2)
			#Atom_output_R2 = self.Mpnn2(Feature_R2,adj_R2)
			#R1_Feature = self.Pooling(Atom_output_R1)
			R_Feature = self.Pooling(Atom_output_R)

			Neg_Atom_output_P = self.Mpnn3(Neg_Feature_P, Neg_adj_P)
			Neg_P_Feature = self.Pooling(Neg_Atom_output_P)

			Atom_output_P = self.Mpnn1(Feature_P,adj_P)
			Positive_bond_feature, output_Bond_RedOut_P, Label_P, Bond_list,outputs_list,Atom_symbol,Bond_symbol = self.Redout_P_Bond_Test(Atom_output_P,Smiles_P,BreakBondID,adj_P,AtomSymble_One_Hot,EP_list)  # 输出目标分子键的特征和标签

			P_Feature = self.Pooling(Atom_output_P)
			# print()
			Atom_feature_id1 = torch.cat((Positive_bond_feature[BreakBondID.split(',')[0]], Atom_output_P[int(BreakBondID.split(',')[0])]), dim=0).unsqueeze(0)
			Atom_feature_id2 = torch.cat((Positive_bond_feature[BreakBondID.split(',')[1]], Atom_output_P[int(BreakBondID.split(',')[1])]), dim=0).unsqueeze(0)

			return Atom_feature_id1, Atom_feature_id2, output_Bond_RedOut_P,Label_P,P_Feature,R_Feature,Neg_P_Feature,outputs_list,Atom_symbol,Bond_symbol


def GetTop5Result(Atom_feature_id1, Atom_feature_id2, Value_Bonds, Labels, Bonds_list, atom1_label, atom2_label):
	Value_Bonds_list = Value_Bonds.tolist()
	Labels_list = Labels.tolist()
	idx_value = {}

	for m in range(len(Bonds_list)):
		idx_value[str(m)] = Value_Bonds_list[m]
	Result = sorted(idx_value.items(), key=lambda x: x[1], reverse=True)
	Result_label = {}
	Result_Substructure = {}
	Bond = {}
	for i in range(len(Result)):
		indx = int(Result[i][0])
		label = Labels_list[indx]
		bond = Bonds_list[indx]
		Bond[i] = bond
		Result_label[i] = label
	
	Atom_id1 = Atom_feature_id1.tolist()
	Atom_id2 = Atom_feature_id2.tolist()
	
	indx_id1 = {}
	indx_id2 = {}

	print('xxxxxxxxxxxxxxxxxxxxxxxxxx')
	print(len(Atom_id1), Atom_id1)
	print(len(atom1_label), atom1_label)
	print(len(Atom_id2),Atom_id2)
	print(len(atom2_label), atom2_label)
	for i in range(len(Atom_id1)):
		indx_id1[str(i)] = Atom_id1[i]
		indx_id2[str(i)] = Atom_id2[i]
	resu_id1 = sorted(indx_id1.items(), key=lambda x: x[1], reverse=True)
	resu_id2 = sorted(indx_id2.items(), key=lambda x: x[1], reverse=True)
	for j in range(len(atom1_label)):

		id1 = int(resu_id1[j][0])
		id2 = int(resu_id2[j][0])
		label1 = atom1_label[id1]
		label2 = atom2_label[id2]
		Result_Substructure[j] = [label1,label2]
	return Result_label, Result_Substructure, Bond


def GetLabelsTensor(Labels_dict, Labels):
	result = []
	for label in Labels:
		result.append(Labels_dict[label])
	return torch.FloatTensor(result)


def GetTopXIndx(list1,x):
	result = []
	idx_value = {}
	for m in range(len(list1)):
		idx_value[str(m)] = list1[m]
	Result = sorted(idx_value.items(), key=lambda x: x[1], reverse=True)
	for i in range(x):
		result.append(int(Result[i][0]))
	return result


def train(epochs, Datalist,device,Train_list,Test_list,model,model2,optimizer1,optimizer2,loss_fn,loss_fn_3,Labels_dict, Labels, Labels_Features_GCN, model3,AtomSymble_One_Hot,EP_list):
	print("Labels_dict")
	for x,y in Labels_dict.items():
		print(x,y)
	Smiles_P_dict = Datalist[0]
	Smiles_R1_dict = Datalist[1]
	Smiles_R2_dict = Datalist[2]
	BreakBondID_dict = Datalist[3]
	Feature_Dict_P = Datalist[4]
	Feature_Dict_R1 = Datalist[5]
	Feature_Dict_R2 = Datalist[6]
	Adj_Dic_P = Datalist[7]
	Adj_Dic_R1 = Datalist[8]
	Adj_Dic_R2 = Datalist[9]
	Adj_Dic_P_Normal = Datalist[10]
	Reaction_class_dict = Datalist[11]
	Result = []
	test_result = {}
	T = {}
	for w in Train_list:
		Smiles_P = Smiles_P_dict[str(w)]
		Smiles_R1 = Smiles_R1_dict[str(w)]
		Smiles_R2 = Smiles_R2_dict[str(w)]
		r = Smiles_P + '>>' + Smiles_R1 + '.' + Smiles_R2
		t = Get_Reaction_t(r, step)
		if t.split('>>')[0] not in T.keys():
			T[t.split('>>')[0]] = [t]
		else:
			T[t.split('>>')[0]].append(t)
	for epoch in range(epochs):
		print('epoch:',epoch)
		optimizer1.zero_grad()
		optimizer2.zero_grad()
		outputs_train = []
		Labels_train = []
		P_Feature_list = []
		Neg_P_Feature_list = []
		R_Feature_list = []
		Featuer_atom_id1 = []
		Label_atom_id1= []
		for i_train in Train_list:
			features_P = Feature_Dict_P[str(i_train)]
			features_R1 = Feature_Dict_R1[str(i_train)]
			features_R2 = Feature_Dict_R2[str(i_train)]
			adj_P = Adj_Dic_P_Normal[str(i_train)]
			adj_R1 = Adj_Dic_R1[str(i_train)]
			adj_R2 = Adj_Dic_R2[str(i_train)]
			Smiles_P = Smiles_P_dict[str(i_train)]
			Smiles_R1 = Smiles_R1_dict[str(i_train)]
			Smiles_R2 = Smiles_R2_dict[str(i_train)]
			BreakBondID = BreakBondID_dict[str(i_train)]
			a = random.randint(1, 20)
			Neg_features_P = Feature_Dict_P[str(i_train)]
			Neg_adj_P = Adj_Dic_P_Normal[str(i_train)]

			Atom_feature_id1, Atom_feature_id2, output_Bond_RedOut_P,Label_P,P_Feature,R_Feature,Neg_P_Feature,outputs_list,Atom_symbol,Bond_symbol = model(EP_list,AtomSymble_One_Hot,features_P,adj_P,BreakBondID,Smiles_P,features_R1,features_R2,adj_R1,adj_R2,Neg_features_P,Neg_adj_P)

			Atom_feature_id1 = model2(Atom_feature_id1, Labels_Features_GCN)
			Atom_feature_id2 = model2(Atom_feature_id2, Labels_Features_GCN)
			Neg_P_Feature_list.append(Neg_P_Feature)
			outputs_train.append(output_Bond_RedOut_P)
			Labels_train.append(Label_P)
			P_Feature_list.append(P_Feature)
			R_Feature_list.append(R_Feature)
			#  model3的训练
			Featuer_atom_id1.append(Atom_feature_id1.reshape(-1))
			Featuer_atom_id1.append(Atom_feature_id2.reshape(-1))

			data = [Smiles_P,BreakBondID,Smiles_R1,Smiles_R2]
			label_id1, label_id2 = Data_Processing(data, Labels_dict)
			Label_atom_id1.append(torch.tensor(label_id1).to(torch.float32))
			Label_atom_id1.append(torch.tensor(label_id2).to(torch.float32))
		Featuer_atom_id1 = torch.cat(Featuer_atom_id1, dim=0).to(device)
		Label_atom_id1 = torch.cat(Label_atom_id1, dim = 0)

		L7 = loss_fn_3(Featuer_atom_id1, Label_atom_id1)
		outputs_train = torch.cat(outputs_train, dim = 0).to(device)

		Labels_train = torch.cat(Labels_train, dim = 0).to(device)
		P_FeatureS = torch.cat(P_Feature_list, dim = 0)
		R_FeatureS = torch.cat(R_Feature_list, dim = 0)

		Neg_P_Features = torch.cat(Neg_P_Feature_list, dim = 0)
		L1 = loss_fn(outputs_train, Labels_train)
		L2 = loss_fn_3(P_FeatureS, R_FeatureS)
		L4 = loss_fn_3(P_FeatureS, Neg_P_Features)

		Loss = L7

		#Loss = 0.2*L1 + 0.2*(L2/L4) + L7

		Loss.backward(retain_graph=True)
		#Loss2.backward()
		optimizer1.step()
		optimizer2.step()
		outputs_test = []
		Labels_test = []

		count_all = 0
		count_bond_top1 = 0
		count_bond_top3 = 0
		count_bond_top5 = 0
		count_model2 = 0


		for i_test in Test_list:
			features_P = Feature_Dict_P[str(i_test)]
			adj_P = Adj_Dic_P_Normal[str(i_test)]
			Smiles_P = Smiles_P_dict[str(i_test)]
			BreakBondID = BreakBondID_dict[str(i_test)]
			Smiles_R1 = Smiles_R1_dict[str(i_test)]
			Smiles_R2 = Smiles_R2_dict[str(i_test)]
			Atom_feature_id1, Atom_feature_id2, Output_P_Bond, label, Bond_list, outputs_list, Atom_symbol, Bond_symbol = model(EP_list, AtomSymble_One_Hot, features_P, adj_P, BreakBondID, Smiles_P)
			Atom_feature_id1 = model2(Atom_feature_id1, Labels_Features_GCN)
			Atom_feature_id2 = model2(Atom_feature_id2, Labels_Features_GCN)

			outputs_test.append(Output_P_Bond)
			Labels_test.append(label)
			data = [Smiles_P,BreakBondID,Smiles_R1,Smiles_R2]
			label_id1,label_id2 = Data_Processing(data,Labels_dict)
			Top5_label, Top5_Substructure, Bond = GetTop5Result(Atom_feature_id1.reshape(-1), Atom_feature_id2.reshape(-1), Output_P_Bond, label, Bond_list,label_id1,label_id2)
			for num1 in range(1):
				if Top5_Substructure[num1][0] == 1: 
					for num2 in range(1):
						if Top5_Substructure[num2][1] == 1: 
							count_model2 += 1
							for num in range(len(Top5_label)):
								if Top5_label[num] == 1:
									if num == 0:
										count_bond_top1 += 1
									if num < 3:
										count_bond_top3 += 1
									if num < 5 :
										count_bond_top5 += 1
			count_all += 1
		# torch.save(model, 'Model_Save/Class_all_model.pt')
		# torch.save(model2, 'Model_Save/Class_all_model2.pt')
		# torch.save(model3, 'Model_Save/Class_all_model3.pt')

		print((str(epoch) + ': Accuracy：' + 'Top1:' + str(count_bond_top1/count_all)+'  Top3:' + str(count_bond_top3/count_all)+'   Top5:' + str(count_bond_top5/count_all)+'\n'))
		print(str(epoch), ":", count_model2, '-', count_all)
		Result.append(str(epoch) + ': Accuracy：' + 'Top1:' + str(count_bond_top1/count_all)+'  Top3:' + str(count_bond_top3/count_all)+'   Top5:' + str(count_bond_top5/count_all)+'\n')
		Result.append('\n')
		tx = open('Test-result.txt', 'w', encoding='UTF-8')
		for p in Result:
			tx.write(p)
device = torch.device('cpu')










