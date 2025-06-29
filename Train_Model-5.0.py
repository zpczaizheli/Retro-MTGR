# from Model_AFPNN_Retrosynthesis_5 import *
# from torch.optim.lr_scheduler import StepLR
from model3 import *
# from GetMoleculeFeature import *
import torch.nn.functional as F
import torch.optim as optim
EP_list = {}
with open('data/USPT-50K/other_data/Atoms_character.txt', 'r',encoding='UTF-8') as file_object:
	for line in file_object:
		line = line.rstrip()
		Name, ep = line.split('	')
		EP_list[Name] = float(ep)

Mol_Xing = Chem.MolFromSmiles('*')
Labels_dict = {}
Labels = []
with open('data/USPT-50K/other_data/Labels.txt', 'r',encoding='UTF-8') as file_object:   
	for line in file_object: 
		line = line.rstrip()
		if line not in Labels:
			Labels.append(line)

count = 1
for num1 in range(len(Labels)):
	if Labels[num1] not in Labels_dict.keys():
		Labels_dict[Labels[num1]] = [0.0]*(count-1) + [1.0] + [0.0]*(len(Labels) - count)
		count += 1

Datapath = 'data/USPT-50K/50k-all.txt'
Data = load_data(Datapath)
device = torch.device('cpu')

Train_list = list(range(0, 31833))
Test_list = list(range(31833, 35370))

# Train_list = list(range(0, 700))
# Test_list = list(range(700, 800))


mini_Labels_dict, mini_Labels, Label_num, Initial_Features, Adj_Labels = GetLabelsrelationship(Labels, Labels_dict, Data)
model = RedOut(device,len(Data[4]['0'][0])).to(device)


# model2 = Classify(len(Data[4]['0'][0])*2+34, Label_num)
model2 = Classify(82, Label_num)
model3 = Model3(Label_num,Label_num,device)
Adj_Labels = torch.FloatTensor(Adj_Labels)
Initial_Features = torch.FloatTensor(Initial_Features)
Initial_Features = model3(Initial_Features, Adj_Labels)

optimizer1 = optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
optimizer2 = optim.Adam(model2.parameters(), lr=0.01, weight_decay=5e-4)
optimizer3 = optim.Adam(model3.parameters(), lr=0.01, weight_decay=1e-4)

loss_fn = nn.BCELoss()  #
loss_fn_3 = nn.MSELoss(reduce=True, size_average=True)


if __name__ == "__main__":
	train(5000,Data,device,Train_list,Test_list,model,model2,optimizer1,optimizer2,loss_fn,loss_fn_3,mini_Labels_dict, Labels, Initial_Features,model3,AtomSymble_One_Hot,EP_list)




