
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 设置学习率
lr = 0.1

torch.manual_seed(42)

model = nn.Sequential(nn.Linear(1, 1)).to(device)

optimizer = optim.SGD(model.parameters(), lr=lr)

loss_fn = nn.MSELoss(reduction='mean')

train_step_fn = make_train_step(model, loss_fn, optimizer)

val_step_fn = make_val_step(model, loss_fn)

writer = SummaryWriter('runs/simple_linear_regression')

# 获取单个小样本，一遍可以使用add_graph
x_sample, y_sample = next(iter(train_loader))
writer.add_graph(model, x_sample.to(device))
