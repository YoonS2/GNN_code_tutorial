# %matplotlib inline

import json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import  tabulate
import collections
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv




###---------------깃헙 이용자들이 웹개발자인지, 머신러닝 개발자인지 예측하는 것임.--Binary classification

###---------------node feature로는 깃헙 이용자들의 별표시한 repository, 사는곳, 메일주소임.


folder="C:/Users/ydb80/Desktop/GNN/GNN_EX2_data"  ##데이터 
with open(folder+"/musae_git_features.json") as json_data:
        data_raw = json.load(json_data)
edges=pd.read_csv(folder+"/musae_git_edges.csv")
target_df=pd.read_csv(folder+"/musae_git_target.csv")

print("5 top nodes labels")
print(target_df.head(5).to_markdown())
print()
print("5 last nodes")
print(target_df.tail(5).to_markdown())

plt.hist(target_df.ml_target,bins=4);
plt.title("Classes distribution")
plt.show()


###-------------class값이 unbalance하긴함

#노드별 features수 파악하기
feat_counts=list(map(len,list(data_raw.values())))

plt.hist(feat_counts,bins=20)
plt.title("Number of features per graph distribution")
plt.show()

###--------------대부분 15~23개의 피쳐를 가짐

import itertools
feats=list(itertools.chain.from_iterable(list(data_raw.values())))

plt.hist(feats,bins=50)
plt.title("Features distribution")
plt.show()


###-----------각 사용자간 어느정도 feature가 공유되는것을 알수 있음


def encode_data(light=False,n=60):  ##시각화를 위해서는 노드 60개만 사용하기로함
    if light==True:
        nodes_included=n
    elif light==False:
        nodes_included=len(data_raw)

    data_encoded={}
    for i in range(nodes_included):# 
        one_hot_feat=np.array([0]*(len(set(feats))))
        this_feat=data_raw[str(i)] #data_raw에 문자형으로 들어가있으니 문자형 취해줌
        one_hot_feat[this_feat]=1 #노드의 피처에 해당하는 것은 1로 넣어주기
        data_encoded[str(i)]=list(one_hot_feat) #딕셔너리형태로 만들어주기

    if light==True: # 노드 60개만 사용한 경우, 해당 되는 노드만 정리 
        sparse_feat_matrix=np.zeros((1,len(set(feats)))) #피처 매트릭스 형태잡아주기위해 0으로 임시 생성
        for j in range(nodes_included):
            temp=np.array(data_encoded[str(j)]).reshape(1,-1)
            sparse_feat_matrix=np.concatenate((sparse_feat_matrix,temp),axis=0)
        sparse_feat_matrix=sparse_feat_matrix[1:,:]  #처음에 만들어두었던 0행은 제외
        return(data_encoded,sparse_feat_matrix)
    elif light==False:
        return(data_encoded, None)
        
        
        
#처음 60개 노드만 시각화 해보기
data_encoded_vis,sparse_feat_matrix_vis=encode_data(light=True,n=60)
plt.figure(figsize=(25,25));
plt.imshow(sparse_feat_matrix_vis[:,:250],cmap='Greys');  #피처 250개에 대해서만 보여주기


###---------------feature들이 얼마나 sparse한지 대충 확인 가능

from torch_geometric.data import Data #그래프 만들어주기 위함


###--------------------torch_geometiic.data의 Data를 사용하면 그래프에서 표현하고자 하는 속성을 추가하기에 좋음
###------------무방향 그래프를 만들것임.
###            방향이 없다는 것은 반대로 말하면 서로 방향을 가진다는 의미이므로, 각 노드사이의 edge를 만들때 두개를 만들어야함
###            예를 들어서 노드1과 노드2가 연결되어있을때,
###             1 -> 2    2 -> 1
###             이렇게 두개를 만들어야함



def construct_graph(data_encoded,light=False):
    import torch
    node_features_list=list(data_encoded.values())#data_encoded는 dic형태임 
    node_features=torch.tensor(node_features_list) #tensor로 만들어줌 -onehot인코딩된 형태
    node_labels=torch.tensor(target_df['ml_target'].values) #label을 tensor형태로
    edges_list=edges.values.tolist() #edges== 서로 연결되어있는 노드 정보 데이터프레임 형태
    edge_index01=torch.tensor(edges_list, dtype = torch.long).T # 연결 노드 정보 tensor형태로 
    edge_index02=torch.zeros(edge_index01.shape, dtype = torch.long)#.T #두노드 간의 엣지는 두개여야하므로. 
    edge_index02[0,:]=edge_index01[1,:] #1에서의 노드 순서를 반대로 넣어줌
    edge_index02[1,:]=edge_index01[0,:]
    edge_index0=torch.cat((edge_index01,edge_index02),axis=1) #옆으로(열로) 붙임
    g = Data(x=node_features, y=node_labels, edge_index=edge_index0) #그래프 만들기
    g_light = Data(x=node_features[:,0:2], #가벼운 그래프 feature 두개씩만
                     y=node_labels   ,
                     edge_index=edge_index0[:,:55]) #edge 연결 정보는 55개까지만 
    if light:
        return(g_light)
    else:
        return(g)
        
def draw_graph(data0):  #그래프를 시각화
    from torch_geometric.utils.convert import to_networkx #networkx 그래프로 그려주기 위함
    import networkx as nx
    #node_labels=data0.y
    if data0.num_nodes>100:
        print("This is a big graph, can not plot...")
        return

    else:
        data_nx = to_networkx(data0)
        node_colors=data0.y[list(data_nx.nodes)] #색깔로 라벨들 구분해줌
        pos= nx.spring_layout(data_nx,scale =1)
        plt.figure(figsize=(12,8))
        nx.draw(data_nx, pos,  cmap=plt.get_cmap('Set1'),
                node_color =node_colors,node_size=600,connectionstyle="angle3",
                width =1, with_labels = False, edge_color = 'k', arrowstyle = "-")
                
                
g_sample=construct_graph(data_encoded=data_encoded_vis,light=True)
draw_graph(g_sample)


#######전체 데이터로 트레인해보기
data_encoded,_=encode_data(light=False)
g=construct_graph(data_encoded=data_encoded,light=False)




#데이터 크기가 커서 그래프를 훈련용/검증용/테스트용 등으로 나누기가 쉽지는 않아보임

#torch_geometric.transforms.AddTrainValTestMask 를 사용할것임. (RandomNodeSplit으로 이름이 바꼈다고함)

#이class는 0과 1을 이용해서 그래프를 나눔. 그 이진 벡터를 mask라고 한다.

#train mask val mask test mask등 나눌수 있음 (해당하지 않는 애들을 0으로 만들거나 해서 마스킹함)

#여기서는 훈련은 0.1, 검증은 0.3 테스트셋을 0.6으로 쓸거임.

#일반적이진 않지만 현실적이라고 생각되어서..



from torch_geometric.transforms import RandomNodeSplit  as masking #AddTrainValTestMask에서 RandomNodeSplit으로바뀜

msk=masking(split="train_rest", num_splits = 1, num_val = 0.3, num_test= 0.6)
g=msk(g)
print(g)
print()
print("training samples",torch.sum(g.train_mask).item())  #train에 해당하는 애들은 train_mast가 True 값을 가지고.. 이런식
print("validation samples",torch.sum(g.val_mask ).item())
print("test samples",torch.sum(g.test_mask ).item())


#이제 SocialGNN 클래스를 만들어볼거임!!

#GCNConv는 일단 2개를 씀.

#첫번째 layer는 우리 그래프에서의 feature수와 동일한 수만큼의 input features와 임의의 output features를 갖는다.

#relu 활성함수를 적용한 후 두번째 layer로 넘겨줌

#두번째 layer는 output수가 label수와 동일하게 2개를 가짐.

#forward function에서 많은 인수를 받을수 있지만 여기서는 edge index와 edge weight만 받겠음.

import torch.nn.Mudule
class SocialGNN(torch.nn.Module):
    def __init__(self,num_of_feat,f):
        super(SocialGNN, self).__init__()
        self.conv1 = GCNConv(num_of_feat, f)
        self.conv2 = GCNConv(f, 2)

    def forward(self, data):
        x = data.x.float()
        edge_index =  data.edge_index
        x = self.conv1(x=x, edge_index=edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x
        
#우리는 모든 노드에 대해서 라벨을 예측할거임.

#그래서 정확도와 손실을 계산할때, 그 set에서만 계산할거임. train동안은 train set으로만 정확도를 구함.

#이렇게 각 mask마다 정확도와 손실을 구할것이다. (masked_loss 함수를 만들것임)

#이건 모든 노드에 손실과 정확도를 구해놓고 마스크를 곱해서 해당 셋에 해당하지 않는 필요없는 노드는 0으로 만드는 식으로 할거임
#왜이렇게하지...???




def masked_loss(predictions,labels,mask):
    mask=mask.float() #True값을 1로 False를 0으로 변환
    mask=mask/torch.mean(mask)
    loss=criterion(predictions,labels)  #손실함수 
    loss=loss*mask
    loss=torch.mean(loss)
    return (loss)    

def masked_accuracy(predictions,labels,mask):
    mask=mask.float()
    mask/=torch.mean(mask)
    accuracy=(torch.argmax(predictions,axis=1)==labels).long()
    accuracy=mask*accuracy
    accuracy=torch.mean(accuracy)
    return (accuracy)
    
def train_social(net,data,epochs=10,lr=0.01):  #학습함수
    optimizer = torch.optim.Adam(net.parameters(), lr=lr) # 00001 
    best_accuracy=0.0

    train_losses=[]
    train_accuracies=[]

    val_losses=[]
    val_accuracies=[]

    test_losses=[]
    test_accuracies=[]

    for ep in range(epochs+1):
        optimizer.zero_grad()
        out=net(data) #여기서 net=model임
        loss=masked_loss(predictions=out,
                         labels=data.y,
                         mask=data.train_mask)
        loss.backward()
        optimizer.step()
        train_losses+=[loss]
        train_accuracy=masked_accuracy(predictions=out,
                                       labels=data.y, 
                                       mask=data.train_mask)
        train_accuracies+=[train_accuracy]

        val_loss=masked_loss(predictions=out,
                             labels=data.y, 
                             mask=data.val_mask)
        val_losses+=[val_loss]

        val_accuracy=masked_accuracy(predictions=out,
                                     labels=data.y, 
                                     mask=data.val_mask)
        val_accuracies+=[val_accuracy]

        test_accuracy=masked_accuracy(predictions=out,
                                      labels=data.y, 
                                      mask=data.test_mask)
        test_accuracies+=[test_accuracy]
        if np.round(val_accuracy,4)> np.round(best_accuracy ,4):
            print("Epoch {}/{}, Train_Loss: {:.4f}, Train_Accuracy: {:.4f}, Val_Accuracy: {:.4f}, Test_Accuracy: {:.4f}"
                               .format(ep+1,epochs, loss.item(), train_accuracy, val_accuracy,  test_accuracy))
            best_accuracy=val_accuracy
         plt.plot(list(map(float,train_losses))) 
    plt.plot(list(map(float,val_losses)))
    plt.plot(list(map(float,test_losses)))
    plt.show()

    plt.plot(list(map(float,train_accuracies)))
    plt.plot(list(map(float,val_accuracies)))
    plt.plot(list(map(float,test_accuracies)))
    plt.show()



num_of_feat=g.num_node_features
net=SocialGNN(num_of_feat=num_of_feat,f=16)
criterion=nn.CrossEntropyLoss()
train_social(net,g,epochs=50,lr=0.1)
