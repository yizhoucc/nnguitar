# import argparse
# import pickle
# from scipy.io import wavfile
# import numpy as np

# # from cases import datelib_spec
# # import importlib
# # importlib.reload(datelib_spec) # every run


# # make dataset
# in_rate, in_data = wavfile.read('C:/Users/24455/Desktop/clean.wav')
# out_rate, out_data = wavfile.read('C:/Users/24455/Desktop/effect.wav')
# assert in_rate == out_rate, "wrong length"
# sample_size = int(in_rate * 100e-3)
# length = len(in_data) - len(in_data) % sample_size
# x = in_data[:length].reshape((-1, 1, sample_size)).astype(np.float32)
# y = out_data[:length].reshape((-1, 1, sample_size)).astype(np.float32)
# split = lambda d: np.split(d, [int(len(d) * 0.6), int(len(d) * 0.8)])
# d = {}
# d["x_train"], d["x_valid"], d["x_test"] = split(x)
# d["y_train"], d["y_valid"], d["y_test"] = split(y)
# d["mean"], d["std"] = d["x_train"].mean(), d["x_train"].std()
# # d.keys()
# # standardize
# for key in "x_train", "x_valid", "x_test":
#     d[key] = (d[key] - d["mean"]) / d["std"]
# pickle.dump(d, open("data.pickle", "wb"))


# test training
import pytorch_lightning as pl
from model import PedalNet
model = PedalNet({
        'num_channels':16,
        'dilation_depth':8,
        'num_repeat':3,
        'kernel_size':3,
        'batch_size':64,
        'learning_rate':1e-3,
        # 'gpus':'0',
        'data':"data.pickle"
        }
    )
# num_epochs = 2
# learning_rate = 0.001
# criterion = torch.nn.MSELoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# # Train the model
# import torch
# from torch.utils.data import TensorDataset, DataLoader
# ds = lambda x, y: TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
# train_ds = ds(d["x_train"], d["y_train"])

# for epoch in range(num_epochs):
#     outputs=model(torch.tensor(d["x_train"][:,:,:]))
#     optimizer.zero_grad()   
#     loss = criterion(outputs, torch.tensor(d["y_train"][:,:,:]))
#     loss.backward()
#     optimizer.step()
# if epoch % 1 == 0:
#     print("Epoch: %d, loss: %1.5f" % (epoch, loss.item()))


trainer = pl.Trainer(
        max_epochs=500, 
        # gpus='0',
        # row_log_interval=100
    )
trainer.fit(model)


    


# testing
import pickle
import torch
from tqdm import tqdm
from scipy.io import wavfile
import argparse
import numpy as np
from model import PedalNet

def save(name, data):
    wavfile.write(name, 44100, data.flatten().astype(np.int16))

@torch.no_grad()



model = PedalNet.load_from_checkpoint('C:/Users/24455/iCloudDrive/misc\pedalnet/lightning_logs/version_3/checkpoints/epoch=2-step=152.ckpt')
model.eval()
# train_data = pickle.load(open(args.train_data, "rb"))
# mean, std = train_data["mean"], train_data["std"]

in_rate, in_data = wavfile.read("C:/Users/24455/iCloudDrive/misc/pedalnet/data/clean.wav")
# in_rate, in_data = wavfile.read("clean.wav")
assert in_rate == 44100, "input data needs to be 44.1 kHz"
sample_size = int(in_rate * 100e-3)
mean=np.mean(in_data)
std=np.std(in_data)
length = len(in_data) - len(in_data) % sample_size
in_data=in_data[:length]
in_data.shape
# plt.plot(in_data)
in_data = (in_data - mean) / std
# plt.plot(in_data)
# split into samples
in_data = in_data.reshape((-1, 1, sample_size)).astype(np.float32)
in_data.shape # sampleind, ch, sampledata
# pad each sample with previous sample
prev_sample = np.concatenate((np.zeros_like(in_data[0:1]), 
        in_data[:-1]), axis=0)
pad_in_data = np.concatenate((prev_sample, in_data), axis=2)
# sampleind, ch, 2xsampledata with prevsample
pred = []
with torch.no_grad():
    batches = pad_in_data.shape[0] // 64
    for x in tqdm(np.array_split(pad_in_data, batches)):
        pred.append(model(torch.from_numpy(x)).numpy())
pred = np.concatenate(pred)
pred = pred[:, :, -in_data.shape[2] :]
pred.shape
# plt.plot(in_data[345,0,:])
# plt.plot(pred[345,0,:])


result=torch.tensor(pred).view(-1)
# result.shape
# result=result.tolist()
# plt.plot(result)
result=np.array(result)
wavfile.write('predicteff.wav', 44100, result)


