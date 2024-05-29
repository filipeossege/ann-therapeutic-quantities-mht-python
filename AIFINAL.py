import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

#Parameters
Runs = 30
kfolds = 5

#Dataset
A = np.loadtxt("DS.txt") # (fi, H_0 [A/m], f [Hz], a [m]) and  (Tc(°C), t(s))
X = A[:,0:4]
Y = A[:,4:6]

XTrain, XTest, YTrain, YTest = train_test_split(X, Y, test_size=0.2)
YTrain1 = YTrain[:,0]
YTrain2 = YTrain[:,1]
YTest1 = YTest[:,0]
YTest2 = YTest[:,1]

#Standardize
scaler = StandardScaler()
XTrain_scaled = scaler.fit_transform(XTrain)
XTest_scaled = scaler.transform(XTest)
scaler_Y1 = StandardScaler()
YTrain1_scaled = scaler_Y1.fit_transform(YTrain1.reshape(-1, 1)).ravel()
scaler_Y2 = StandardScaler()
YTrain2_scaled = scaler_Y2.fit_transform(YTrain2.reshape(-1, 1)).ravel()

Mdl1 = MLPRegressor(hidden_layer_sizes=(161, 10, 300),
                    activation='logistic', #logistic = sigmoid
                    alpha=5.0754e-06,
                    max_iter=1000,
                    solver='lbfgs')
Mdl1.fit(XTrain_scaled, YTrain1_scaled)

testPredictions1 = scaler_Y1.inverse_transform(Mdl1.predict(XTest_scaled).reshape(-1, 1)).ravel()

plt.figure()
plt.plot(YTest1, testPredictions1, ".")
plt.plot(YTest1, YTest1)
plt.xlabel("True Temperature (°C)")
plt.ylabel("Predicted Temperature (°C)")
plt.savefig('Temperature.png', format='png', dpi=1200, bbox_inches='tight')


R1 = np.concatenate((XTest, YTest1.reshape(-1, 1), testPredictions1.reshape(-1, 1)), axis=1)

# Temperature Surfaces
xlin = np.linspace(np.min(R1[:,0]), np.max(R1[:,0]), 50)
ylin = np.linspace(np.min(R1[:,1]), np.max(R1[:,1]), 50)
X1, Y1 = np.meshgrid(xlin, ylin)
Z_predict = griddata((R1[:,0], R1[:,1]), R1[:,-1], (X1, Y1), method='nearest')  # predict
Z_real = griddata((R1[:,0], R1[:,1]), R1[:,4], (X1, Y1), method='nearest')  # real
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(Y1, X1, Z_predict, alpha=0.3, facecolor='r', linewidth=0.1, edgecolor='r')
ax.plot_surface(Y1, X1, Z_real, alpha=0.3, facecolor='b', linewidth=0.1, edgecolor='b')
ax.legend(['ANN', 'Real'])
ax.set_ylabel(r'$\phi$')
ax.set_xlabel(r'$H_0$ (A/m)')
ax.set_xlim(4000,1000)
ax.set_zlabel(r'$T_c$ (°C)')
ax.zaxis.labelpad=-0.7
ax.view_init(elev=20, azim=70, roll=0)
plt.savefig('1.png', format='png', dpi=1200, bbox_inches='tight')

xlin = np.linspace(np.min(R1[:,2]), np.max(R1[:,2]), 50)
ylin = np.linspace(np.min(R1[:,3]), np.max(R1[:,3]), 50)
X1, Y1 = np.meshgrid(xlin, ylin)
Z_predict = griddata((R1[:,2], R1[:,3]), R1[:,-1], (X1, Y1), method='nearest')  # predict
Z_real = griddata((R1[:,2], R1[:,3]), R1[:,4], (X1, Y1), method='nearest')  # real
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(Y1, X1, Z_predict, alpha=0.3, facecolor='r', linewidth=0.1, edgecolor='r')
ax.plot_surface(Y1, X1, Z_real, alpha=0.3, facecolor='b', linewidth=0.1, edgecolor='b')
ax.legend(['ANN', 'Real'])
ax.set_ylabel(r'$f$ (Hz)')
ax.set_xlabel(r'$a$ (m)')
ax.set_xlim(8e-9,5e-9)
ax.set_zlabel(r'$T_c$ (°C)')
ax.zaxis.labelpad=-0.7
ax.view_init(elev=20, azim=20, roll=0)
plt.savefig('2.png', format='png', dpi=1200, bbox_inches='tight')

