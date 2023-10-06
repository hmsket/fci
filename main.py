import jax.numpy as jnp

from common import get_mnist_dataset, im2col


# おもみをファイルから読み込む
conv_w = jnp.load(f'./params/conv_w.npy') # shape: (2, 25)

# MNISTを取得する
(train_x, train_y), (test_x, test_y) = get_mnist_dataset(labels=[6, 7])

idx = 0
x = train_x[idx:idx+1] # idx枚目だけを使う

# 画像をおもみで畳み込む
col = im2col(x, kernel_size=[5, 5])
y = jnp.matmul(conv_w, col) # shape: (1, 2, 576)

# おもみと最も反応した場所のインデックスを取得する
idxs = jnp.argmax(y, axis=2)
print(idxs)
