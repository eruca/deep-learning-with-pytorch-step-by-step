
torch.manual_seed(13)

# 在拆分之前从numpy数组构建张量
x_tensor = torch.from_numpy(x).float()
y_tensor = torch.from_numpy(y).float()

# 构建包含所有数据点的数据集
dataset = TensorDataset(x_tensor, y_tensor)

# 执行拆分
ratio = .8
n_total = len(dataset)
n_train = int(n_total * ratio)
n_val = n_total - n_train

train_data, val_data = random_split(dataset, [n_train, n_val])

# 构建每个集合的加载器
train_loader = DataLoader(dataset=train_data, batch_size=16, shuffle=True)
val_loader = DataLoader(dataset=val_data, batch_size=16)
