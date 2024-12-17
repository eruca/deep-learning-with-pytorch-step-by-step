
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 设置学习率
lr = 0.1

torch.manual_seed(42)
# 现在可以创建一个模型并立即将其发送到设备
model = nn.Sequential(nn.Linear(1,1)).to(device)

# 定义SGD优化器来更新参数
optimizer = optim.SGD(model.parameters(), lr=lr)

# 定义MSE损伤函数
loss_fn = nn.MSELoss(reduction='mean')

# 为模型、损伤函数和优化器创建train_step函数
train_step = make_train_step(model, loss_fn, optimizer)
