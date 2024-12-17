
device = 'cude' if torch.cuda.is_available() else 'cpu'

# 设置学习率
lr = 0.1

torch.manual_seed(42)
# 现在可以创建模型并立即发送到设备
model = nn.Sequential(nn.Linear(1, 1)).to(device)

# 定义GSD优化器来更新
optimizer = optim.SGD(model.parameters(), lr=lr)

# 定义MSE损失函数
loss_fn = nn.MSELoss(reduction='mean')

# 为模型、损伤函和优化器创建train_step函数
train_step = make_train_step(model, loss_fn, optimizer)

# 为模型和损失函数创建val_step函数
val_step = make_val_step(model, loss_fn)
