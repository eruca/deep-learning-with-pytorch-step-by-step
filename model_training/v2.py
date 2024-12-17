
# 定义周期数
n_epochs = 1000

losses = []

for epoch in range(n_epochs):
    # 内循环
    mini_batch_losses = []
    for x_batch, y_batch in train_loader:
        # 数据集存在于CPU中，小批量也是如此
        # 因此需要将这些小批量，发送到模型存在的设备
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        # 执行一个训练步骤
        # 并返回小批量的相应损失
        mini_batch_loss = train_step(x_batch, y_batch)
        mini_batch_losses.append(mini_batch_loss)

    # 计算所有小批量的平均损失-- 这是周期损失
    loss = np.mean(mini_batch_losses)

    losses.append(loss)
