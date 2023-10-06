import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from common import get_mnist_dataset, im2col


w_len = 5 # おもみフィルタの１辺の長さ

# おもみをファイルから読み込む
conv_w = jnp.load(f'./params/conv_w.npy') # shape: (2, 25)

# MNISTを取得する
(train_x, train_y), (test_x, test_y) = get_mnist_dataset(labels=[6, 7])


# １次元のインデックスをxy座標に変換する関数
def idx2xy(idx, w_len):
    q = idx // (28 - w_len + 1)
    tmp = idx + q * (w_len - 1)
    x = tmp // 28
    y = tmp % 28
    return x, y


fig = plt.figure()
colors = ['r', 'g']

for i in range(16):
    ax = fig.add_subplot(4, 4, i+1)
    ax.set_xticks([])
    ax.set_yticks([])

    # 画像を１枚だけ取得する
    image = train_x[i:i+1]

    # 画像をおもみで畳み込む
    col = im2col(image, kernel_size=[w_len, w_len])
    y = jnp.matmul(conv_w, col)

    # おもみと最も反応した場所のインデックスを取得する
    tmp_idxs = jnp.argmax(y, axis=2)
    idxs = jnp.reshape(tmp_idxs, [-1]) # １次元配列に変換する

    for j, idx in enumerate(idxs):
        # インデックスをxy座標に変換する
        x, y = idx2xy(idx, w_len)
        
        # 枠を描画する
        r = patches.Rectangle(xy=(y-0.5,x-0.5), width=w_len, height=w_len, fill=False, color=colors[j%2])
        ax.add_patch(r)

    ax.imshow(jnp.reshape(image, [28, 28]), cmap=plt.cm.gray_r)

plt.show()
