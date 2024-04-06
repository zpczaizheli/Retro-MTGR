

from Model_AFPNN_Retrosynthesis_5 import *

def remove_zero_rows_and_columns(matrix):
    if not matrix:
        return matrix

    num_rows = len(matrix)
    num_cols = len(matrix[0])

    zero_rows = set()
    zero_cols = set()

    for row in range(num_rows):
        for col in range(num_cols):
            if matrix[row][col] != 0:
                break
        else:
            zero_rows.add(row)


    for col in range(num_cols):
        for row in range(num_rows):
            if matrix[row][col] != 0:
                break
        else:
            zero_cols.add(col)

    new_matrix = [matrix[i] for i in range(num_rows) if i not in zero_rows]
    new_matrix = [[new_matrix[i][j] for j in range(num_cols) if j not in zero_cols] for i in range(len(new_matrix))]

    return new_matrix


def Listadd(list1, list2):
	result = []
	for i in range(len(list1)):
		result.append(list1[i] + list2[i])
	return result


def GetLabelsrelationship(Labels,Labels_dict_OneHot,Data):
	LabelsDict_relasionship = {}
	for i in range(len(Labels)):
		LabelsDict_relasionship[Labels[i]] = [0.0]*len(Labels)
	for num in range(len(Data[0])):
		Smiles_P = Data[0][str(num)]
		Smiles_R1 = Data[1][str(num)]
		Smiles_R2 = Data[2][str(num)]
		BreakBondID = Data[3][str(num)]
		data = [Smiles_P, BreakBondID, Smiles_R1, Smiles_R2]
		label1, label2 = Data_Processing(data, Labels_dict_OneHot)
		x = 0
		y = 0
		for i in range(len(label1)):
			if label1[i] == 1:
				x = i
			if label2[i] == 1:
				y = i
		LabelsDict_relasionship[Labels[x]] = Listadd(LabelsDict_relasionship[Labels[x]],label2)
		LabelsDict_relasionship[Labels[y]] = Listadd(LabelsDict_relasionship[Labels[y]],label1)
	
	for x,y in LabelsDict_relasionship.items():
		print(x,y)
	Adj_Labels = []
	mini_labels = []
	for num in range(len(Labels)):
		if sum(LabelsDict_relasionship[Labels[num]]) != 0:
			mini_labels.append(Labels[num])
		Adj_Labels.append(LabelsDict_relasionship[Labels[num]])

	matrix = remove_zero_rows_and_columns(Adj_Labels)

	mini_adj = []
	mini_Labels_dict = {}
	print('mini_labels', len(mini_labels))
	coun = 1
	for coun in range(len(mini_labels)):
		row = [0.0]*(coun) +  [1.0] + [0.0]*(len(mini_labels) - coun - 1)
		if mini_labels[coun] not in mini_Labels_dict.keys():
			mini_Labels_dict[mini_labels[coun]] = row
		mini_adj.append(row)
	v = 0
	result = []
	for i in matrix:
		normalized_list = [x / sum(i) for x in i]
		count = sum(i)
		result.append(normalized_list)
		v += count
		#print(i,count)
	print(v)
	return mini_Labels_dict, mini_labels,len(mini_labels),mini_adj,result


class GraphConvolution(nn.Module):
	def __init__(self, in_features, out_features, device, bias=True):
		super(GraphConvolution, self).__init__()
		self.weight = nn.Linear(in_features, out_features).to(device)
		self.use_bias = bias
		if self.use_bias:
			self.bias = nn.Parameter(torch.FloatTensor(out_features)).to(device)
		self.reset_parameters()
		
	def reset_parameters(self):
		#nn.init.normal_(self.weight)
		#nn.init.kaiming_uniform_(self.weight)
		if self.use_bias:
			nn.init.zeros_(self.bias)
		pass
	
	def forward(self, input_features, adj):
		support = self.weight(input_features)
		output = torch.mm(adj, support)
		output = torch.tanh(output)
		#print('output:',output)
		if self.use_bias: 
			#print(self.bias)
			return output + self.bias
		else:
			return output


class Model3(nn.Module):
	def __init__(self,input_dim,output_dim,device):
		super(Model3, self).__init__()
		self.gcn1 = GraphConvolution(input_dim, input_dim * 2,device)
		self.gcn2 = GraphConvolution(input_dim * 2, output_dim,device)
		#self.fc1 = nn.Linear(56, 28).to(device)  # 从两个原料分子拼接的向量映射到目标分子的向量
		self.reset_parameters()
		pass
	def reset_parameters(self):
		#nn.init.normal_(self.weight_FC)
		#nn.init.normal_(self.weight_BondEnergy)
		#nn.init.kaiming_uniform_(self.weight_BondEnergy)
		pass
	def forward(self,Nodes_Feature,Adj):
		output = self.gcn1(Nodes_Feature,Adj)
		#output = torch.relu(output)
		output = self.gcn2(output,Adj)
		return output




		
	
