device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'device:{device}')

# 数据在Numpy数组中
# 需要将它们转化为PyTorch的张量
# 然后将它们发送到所选设备
x_train_tensor = torch.as_tensor(x_train).float().to(device)
y_train_tensor = torch.as_tensor(y_train).float().to(device)
