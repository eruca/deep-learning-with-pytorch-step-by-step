
# 定义周期数
n_epochs = 200

losses = []
val_losses = []

for epoch in range(n_epochs):
    # 内循环
    loss = mini_batch(device, train_loader, train_step)
    losses.append(loss)

    # 验证--验证中没有梯度
    with torch.no_grad():
        val_loss = mini_batch(device, val_loader, val_step)
        val_losses.append(val_loss)
