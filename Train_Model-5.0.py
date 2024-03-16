# 在版本4的基础上增加标签之间的相关性，标签的编码方式不再是One-Hot，而是带有权重的向量

# from Model_AFPNN_Retrosynthesis_5 import *
# from torch.optim.lr_scheduler import StepLR
from model3 import *
# from GetMoleculeFeature import *


EP_list = {}
with open('data/USPT-50K/other_data/原子电负性表.txt', 'r',encoding='UTF-8') as file_object:
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
		Labels_dict[Labels[num1]] = [0.0]*(count-1) +  [1.0] + [0.0]*(len(Labels) - count)
		count += 1

Datapath = 'data/USPT-50K/class2.txt'
Data = load_data(Datapath) #  S_P_d = {}, S_R1_d = {}, S_R2_d = {}, Bond_dict = {},F_D_P = {},F_D_R1 = {} F_D_R2 = {},A_D_P = {}, A_D_R1 = {},A_D_R2 = {}
device = torch.device('cpu')

# 总数居Train(0,27019); Valid(27019,30408); Test(30408,33803);
Train_list = list(range(0, 3375))
Test_list = list(range(3375,3975))

mini_Labels_dict, mini_Labels, Label_num, Initial_Features, Adj_Labels = GetLabelsrelationship(Labels, Labels_dict, Data)



model = RedOut(device,len(Data[4]['0'][0])).to(device)
#model = torch.load('Model_Save/Class_all_model_2.pt')

model2 = Classify(len(Data[4]['0'][0])*2+34, Label_num)
#model2 = torch.load('Model_Save/Class_al_model2.pt')

model3 = Model3(Label_num,Label_num,device)
#model3 = torch.load('Model_Save/Class_all_model3.pt')

#print(len(Labels_dict)) 


Adj_Labels = torch.FloatTensor(Adj_Labels)
Initial_Features = torch.FloatTensor(Initial_Features)
print(Initial_Features.shape)
print(Adj_Labels.shape)
#Labels_Features_GCN = model3(Initial_Features, Adj_Labels)

txt = open('output/Adj_Labels.txt', 'w',encoding='UTF-8')
for i in Adj_Labels:
	for j in i:
		txt.write(str(j.item()) + '	')
	txt.write('\n')


optimizer1 = optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
optimizer2 = optim.Adam(model2.parameters(), lr=0.01, weight_decay=5e-4)
optimizer3 = optim.Adam(model3.parameters(), lr=0.01, weight_decay=1e-4)


#scheduler = StepLR(optimizer1, step_size=10, gamma=0.5)
#scheduler = StepLR(optimizer2, step_size=10, gamma=0.5)
loss_fn = nn.BCELoss()  # 
#loss_fn_2 = ContrastiveLoss()  # 对比学习loss
loss_fn_3 = nn.MSELoss(reduce=True, size_average=True)


# Labels_Features_Gcn



if __name__ == "__main__":
	train(1000,Data,device,Train_list,Test_list,model,model2,optimizer1,optimizer2,loss_fn,loss_fn_3,mini_Labels_dict, Labels, Initial_Features,model3,AtomSymble_One_Hot,EP_list)




