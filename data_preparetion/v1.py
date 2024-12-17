
# 数据在numpy数组中
# 但需要将它们转化为PyTorch的张量
x_train_tensor = torch.from_numpy(x_train).float()
y_train_tensor = torch.from_numpy(y_train).float()

# 构建Dataset
train_data = TensorDataset(x_train_tensor, y_train_tensor)

# 构建DataLoader
train_loader = DataLoader(dataset=train_data, batch_size=16, shuffle=True)
