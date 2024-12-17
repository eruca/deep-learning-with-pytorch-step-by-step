
# 定义周期数
n_epochs = 1000

losses = []

# 对于每个周期...
for epoch in range(n_epochs):
    # 执行一个训练步骤并返回相应的损失
    loss = train_step(x_train_tensor, y_train_tensor)
    losses.append(loss)
