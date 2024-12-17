
# 定义周期数
n_epochs = 200

losses = []
val_losses = []

for epoch in range(n_epochs):
    loss = mini_batch(device, train_loader, train_step_fn)
    losses.append(loss)

    # 验证
    with torch.no_grad():
        val_loss = mini_batch(device, val_loader, val_step_fn)
        val_losses.append(val_loss)

    # 在主标签“损失”下记录每个周期的损失
    writer.add_scalars(
        main_tag='loss',
        tag_scalar_dict={'training':loss, 'validation':val_loss},
        global_step = epoch
    )

# 关闭编写器
writer.close()
