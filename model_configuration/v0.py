
device = 'cuda' if torch.cuda.is_available() else 'cpu'
lr = 0.1

torch.manual_seed(42)
# 现在可以创建一个模型，并立即将其发送到设备
model = nn.Sequential(nn.Linear(1, 1)).to(device)

# 定义SGD优化器来更新参数
# 现在直接从模型找那个检索
optimizer = optim.SGD(model.parameters(), lr=lr)

# 定义MSE损失函数
loss_fn = nn.MSELoss(reduction='mean')
