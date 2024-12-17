# 定义周期数
n_epochs = 1000

for epoch in range(n_epochs):
    # 设置训练模式
    model.train()

    # 第一步：计算模型的预测输出-- 前向传递
    yhat = model(x_train_tensor)

    # 第2步：计算损失
    loss = loss_fn(yhat, y_train_tensor)

    # 第3步：计算参数b和w的梯度
    loss.backward()

    # 第4步：使用梯度和学习率更新参数
    optimizer.step()
    optimizer.zero_grad()
