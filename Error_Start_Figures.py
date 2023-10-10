import torch
import torch.nn as nn
import numpy as np
from Train_Gamma import Train_NN
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

from sklearn.decomposition import PCA

from sklearn.metrics import mean_squared_error 


class ESN(nn.Module):
    "Implements the base ESN class"
    def __init__(self, sdim, spectral_radius=0.9, leaky_rate=0.3):
        torch.manual_seed(5)
        super(ESN, self).__init__()
        self.activation = nn.Tanh()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.dim = sdim**2
        self.spectral_radius = spectral_radius
        self.leaky_rate = leaky_rate
    
        self.W = torch.from_numpy(np.loadtxt('W.txt')).float().to(self.device)
        self.W = self.W/torch.max(torch.abs(torch.linalg.eigvalsh(self.W)))
        self.W_in = (torch.rand(self.dim, self.dim, requires_grad=False, device = self.device)-0.5)*2
        self.model_h = None
        self.model_r = None
        self.pca = None
        self.pca_x = None

        

        

    def forward(self, x, u):
        u = torch.from_numpy(np.pad(u.detach().cpu().numpy(), (0, self.dim-1), 'constant', constant_values=(0, 0))).float().to(self.device)
        return (1-self.leaky_rate)*x + self.leaky_rate*self.activation(self.spectral_radius*self.W @ x + self.W_in @ u)
    
    def collect_states(self, u, length, discard):
        x = torch.zeros(self.dim, device=self.device)
        print('Collecting states')
        states = []
        for i in range(length+discard):
            x = self.forward(x, u[i])
            if i > discard:
                states.append(x)
            if i%100 == 0:
                print('Collecting states: ', i, '/', length+discard)    
        return torch.stack(states).T
    
    def fit(self, u, length, discard, k):
        #readout
        states = self.collect_states(u, length, discard)

        self.model_r = Train_NN(states.T.to(self.device),u[discard+2:discard+length+1].to(self.device),hidden_layers=4,layer_dimension=500,epochs=500,batch_size=500)
        #h'
        # collect u in delays [u_n-k, u_n-k+1, ..., u_n-1, u_n]
        u_delays = torch.zeros(length-1, k)
        for i in range(k):
            u_delays[:,i] = u[discard-k+i+1:discard+length-k+i].reshape(-1)
        self.pca = PCA(n_components=100)
        self.scaler = MinMaxScaler(feature_range=(-0.2,0.2))
        y_train = torch.from_numpy(self.scaler.fit_transform(self.pca.fit_transform(states.T.detach().cpu().numpy()))).float()

        # train u_delays to x_n

        self.model_h = Train_NN(u_delays.to(self.device),y_train.to(self.device),hidden_layers=4,layer_dimension=500,epochs=500,batch_size=500)

        self.pca_x = PCA(n_components=3)
        act_states = self.pca_x.fit_transform(states.T.detach().cpu().numpy())

        return act_states
    
    def predict(self, u, pred_len, warmup_len, k, err):

        prediction = torch.zeros(pred_len, 1)
        states_PCA = torch.zeros(pred_len, 3)

        x0 = torch.from_numpy(self.pca.inverse_transform(self.scaler.inverse_transform(self.model_h.predict(u[warmup_len-k:warmup_len].T.to(self.device)).detach().cpu().numpy()))).float().reshape(-1).to(device=self.device)
        
        x = x0 + torch.rand(x0.shape, device=self.device)*err

        
        prediction[0] = self.model_r.predict(x.to(self.device))
        for i in range(pred_len-1):
            states_PCA[i] = torch.from_numpy(self.pca_x.transform(x.detach().cpu().numpy().reshape(1,-1))).float()
            x = self.forward(x, prediction[i])

            prediction[i+1] = self.model_r.predict(x.to(self.device))
        
        return prediction, states_PCA


spectral_radius = 0.9
leaky_rate = 0.7
train_len = 30000
discard = 1000
pred_len = 3000
warmup_len = 400
k=7
err = 1

scaler = MinMaxScaler(feature_range=(-0.2,0.2))

esn = ESN(30, spectral_radius, leaky_rate)
print('ESN created')
train_data = torch.from_numpy(scaler.fit_transform(np.loadtxt('Lorenz_Train.txt').reshape(-1,1))).float() 
act_states = esn.fit(train_data, train_len, discard, k)
test_data = torch.from_numpy(scaler.fit_transform(np.loadtxt('Lorenz_Test.txt').reshape(-1,1))).float()
prediction, states_PCA = esn.predict(test_data, pred_len, warmup_len, k, err)

#save data in files
np.savetxt('Pred_Data/Lorenz_Actual_States.txt', act_states)
np.savetxt('Pred_Data/Lorenz_PCA_States.txt', states_PCA)
np.savetxt('Pred_Data/Lorenz_Prediction.txt', prediction.detach().cpu().numpy())
np.savetxt('Pred_Data/Lorenz_Test_Wrong.txt', test_data[1:pred_len+1].detach().cpu().numpy())

