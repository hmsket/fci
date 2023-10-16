import numpy as np
import jax.numpy as jnp

from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical


# MNISTのうち，配列labelsで指定したラベルのみのデータセットを作成する関数
def get_mnist_dataset(labels):
    (train_x, train_y), (test_x, test_y) = mnist.load_data()

    train_idx = jnp.zeros(len(train_y))
    test_idx = jnp.zeros(len(test_y))

    for label in labels:
        train_idx = train_idx + (train_y == label)
        test_idx = test_idx + (test_y == label)

    train_x = train_x[jnp.where(train_idx)]
    train_y = train_y[jnp.where(train_idx)]

    test_x = test_x[jnp.where(test_idx)]
    test_y = test_y[jnp.where(test_idx)]

    # ラベルの値を変える
    # e.g. labels=[6,7]のとき，6を0,7を1に変える
    # こうしないと，to_categorical()が使えない
    for i, label in enumerate(labels):
        np.place(train_y, (train_y==label)>0, i)
        np.place(test_y, (test_y==label)>0, i)
 
    # 画像のピクセル値を，0以上1以下に正規化
    train_x = train_x.astype('float32') / 255
    test_x = test_x.astype('float32') / 255

    # 望ましい値を，one-hotベクトルにする
    N = len(labels)
    train_y = to_categorical(train_y, N) # ２番目の引数で，次元数を指定する
    test_y = to_categorical(test_y, N)

    # numpyの配列を，jax.numpyの配列に型変換する
    train_x = jnp.array(train_x)
    train_y = jnp.array(train_y)
    test_x = jnp.array(test_x)
    test_y = jnp.array(test_y)

    return (train_x, train_y), (test_x, test_y)


def im2col(x, kernel_size):
    batch_size = x.shape[0]
    image_size = x.shape[1:] # (28, 28)
    conv_size = (image_size[0]-kernel_size[0]+1, image_size[1]-kernel_size[1]+1) # 畳み込みをしたあとの行列のサイズ
    
    tmp_col = jnp.empty([batch_size, kernel_size[0], kernel_size[1], conv_size[0], conv_size[1]])

    for i in range(kernel_size[0]):
        for j in range(kernel_size[1]):
            tmp_col = tmp_col.at[:,i,j].set(x[:,i:i+conv_size[0], j:j+conv_size[1]])

    tmp_col = jnp.transpose(tmp_col, [1, 2, 0, 3, 4])
    tmp_col = jnp.reshape(tmp_col, (kernel_size[0]*kernel_size[1], batch_size*conv_size[0]*conv_size[1]))
    tmp_col = jnp.transpose(tmp_col)
    tmp_col = jnp.reshape(tmp_col, (batch_size, conv_size[0]*conv_size[1], kernel_size[0]*kernel_size[1]))
    col = jnp.transpose(tmp_col, [0, 2, 1])
    return col