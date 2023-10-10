import  matplotlib.pyplot as plt
import numpy as np

#For the cold start figures
name = 57
pred_len = 200
warmup_len = 7

data = np.load(f'Pred_Data/Lorenz_{name}.npy').T
test_data = data[:pred_len+warmup_len]
prediction_c = data[pred_len+warmup_len:2*(pred_len) + warmup_len]
prediction_w = data[2*(pred_len) + warmup_len:]

print(test_data.shape)


plt.rcParams['figure.dpi'] = 200
plt.rcParams['figure.figsize'] = [10, 8]
plt.rcParams['axes.titley'] = 1.0    
plt.rcParams['axes.titlepad'] = -10  
fig = plt.figure(figsize=(10, 4))
ax = fig.subplots()
time = np.linspace(21, 25, 200)
time2 = np.linspace(21-0.02*7, 21, 7)

ax.plot(time2, test_data[1:8], c='red', lw = 1)
ax.plot(time, test_data[7:pred_len+warmup_len], label='Actual', c='red', lw = 1)
ax.plot(time, prediction_c, label='Cold', c='blue', lw = 1)
ax.plot(time2, prediction_w[:7], label='Warmup', c='green', lw = 1)
ax.plot(time, prediction_w[6:-1], label='Warm', c='orange', lw = 1)
ax.legend(fontsize = 11, loc = 'upper right')   

plt.savefig(f'Img/Lorenz_{name}.png')




# for the scatterplot

mse = np.loadtxt(f'Pred_Data/Lorenz_Error_MSE_{9999}.txt')
err = np.loadtxt(f'Pred_Data/Lorenz_Error_{9999}.txt')
fig = plt.figure()
ax = fig.subplots(1,1)

ax.scatter(err[:1000], mse[:1000], s=0.2, c='blue', label='MSE')
ax.scatter(err[1000:2000], mse[1000:2000], s=0.2, c='red', label='MSE')
ax.scatter(err[2000:3000], mse[2000:3000], s=0.2, c='green', label='MSE')
ax.scatter(err[3000:4000], mse[3000:4000], s=0.2, c='orange', label='MSE')
ax.scatter(err[4000:5000], mse[4000:5000], s=0.2, c='purple', label='MSE')
ax.scatter(err[5000:6000], mse[5000:6000], s=0.2, c='black', label='MSE')
ax.scatter(err[6000:7000], mse[6000:7000], s=0.2, c='yellow', label='MSE')
ax.scatter(err[7000:8000], mse[7000:8000], s=0.2, c='pink', label='MSE')
ax.scatter(err[8000:9000], mse[8000:9000], s=0.2, c='brown', label='MSE')
ax.scatter(err[9000:], mse[9000:], s=0.2, c='gray', label='MSE')
plt.savefig(f'Img/Lorenz_Error_MSE.png')


#For the wrong cold start figures
act_states = np.loadtxt('Pred_Data/Lorenz_Actual_States.txt')
states_PCA = np.loadtxt('Pred_Data/Lorenz_PCA_States.txt')
prediction = np.loadtxt('Pred_Data/Lorenz_Prediction.txt')
test_data = np.loadtxt('Pred_Data/Lorenz_Test_Wrong.txt')

pred_len = 3000

plt.rcParams['figure.dpi'] = 600
plt.rcParams['figure.figsize'] = [8, 2]
plt.rcParams['axes.titley'] = 1.0    
plt.rcParams['axes.titlepad'] = -10  
fig = plt.figure()
ax = fig.subplots(1,1)
ax.set(xlim = (0, 3000), ylim = (-0.22, 0.28))
ax.plot(range(pred_len), test_data[:pred_len], label='Actual', c='red', lw = 0.5)
ax.plot(range(pred_len), prediction, label='Prediction', c='blue', lw = 0.5)
plt.savefig(f'Img/Lorenz_Wrong_Prediction.png')
plt.rcParams['figure.figsize'] = [8, 8]
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(act_states[:,0], act_states[:,1], act_states[:,2], lw = 1, label='Actual', alpha=0.3, c='red')
ax.scatter(states_PCA[0,0], states_PCA[0,1], states_PCA[0,2], s = 10, c='blue')
ax.plot(states_PCA[:-2,0], states_PCA[:-2,1], states_PCA[:-2,2], lw = 0.5, c='blue', label='Prediction', alpha=1)

plt.savefig(f'Img/Lorenz_Wrong_Prediction_PCA.png')
