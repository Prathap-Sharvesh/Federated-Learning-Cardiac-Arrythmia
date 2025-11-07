import numpy as np

data = np.load ('client1_data.npy', mmap_mode='r')
data1 = np.load('ecg_train_data.npy')
print("Data shape:", data1.shape)

print(len(data1))