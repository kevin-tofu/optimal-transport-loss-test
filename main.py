import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
import numpy as np
import tensorflow as tf

def cross_entropy(a, b):

    ret = list()
    for al, bl in zip(a, b):
        temp = [ - al2 * np.log(bl2) for al2, bl2 in zip(al, bl)]
        ret.append(temp)
    return ret


def binary_cross_entropy(a, b):

    ret = list()
    for al, bl in zip(a, b):
        temp = list()
        for al2, bl2 in zip(al, bl):
            if al2 == 1:
                temp.append(- al2 * np.log(bl2))
            else:
                temp.append(- ( 1 - al2) * np.log(1-bl2))

        ret.append(temp)
    return ret


def main(args: argparse.Namespace):

    n_groups = 7
    x = np.arange(n_groups)
    v_peak = 0.8
    v = (1 - v_peak) / 6.0
    y_true = [[0, 1, 0, 0, 0, 0, 0]]
    # y_pred_1 = [[v, v_peak, v, v, v, v, v]]
    y_pred_1 = [[v, v, v_peak, v, v, v, v]]
    y_pred_2 = [[v, v, v, v, v, v, v_peak]]
    loss1_b = tf.keras.losses.binary_crossentropy(y_true, y_pred_1)
    loss2_b = tf.keras.losses.binary_crossentropy(y_true, y_pred_2)
    loss1_c = tf.keras.losses.categorical_crossentropy(y_true, y_pred_1)
    loss2_c = tf.keras.losses.categorical_crossentropy(y_true, y_pred_2)
    print(loss1_b.numpy())
    print(loss2_b.numpy())
    print(loss1_c.numpy())
    print(loss2_c.numpy())

    loss_cross_1 = cross_entropy(y_true, y_pred_1)
    loss_cross_2 = cross_entropy(y_true, y_pred_2)
    loss_binary_1 = binary_cross_entropy(y_true, y_pred_1)
    loss_binary_2 = binary_cross_entropy(y_true, y_pred_2)


    fig = plt.figure(figsize= (16, 8))
    
    ax1 = fig.add_subplot(2, 2, 1)
    n_groups_data = 2  # グループの数
    bar_width = 0.8 / n_groups_data  # バーの幅を調整
    group_data = [y_true[0], y_pred_1[0]]
    for i in range(n_groups_data):
        ax1.bar(x + (i - n_groups_data/2) * bar_width, group_data[i], width=bar_width, label=f'Group {i+1}')
    ax1.grid()
    ax1.legend(['Ground-Truth', 'Prediction'])
    
    ax2 = fig.add_subplot(2, 2, 2)
    n_groups_data = 2  # グループの数
    bar_width = 0.8 / n_groups_data  # バーの幅を調整
    group_data = [loss_cross_1[0], loss_binary_1[0]]
    for i in range(n_groups_data):
        ax2.bar(x + (i - n_groups_data/2) * bar_width, group_data[i], width=bar_width, label=f'Group {i+1}')
    ax2.grid()
    ax2.legend(['Cross Entropy', 'Binary Cross Entropy'])

    

    ax3 = fig.add_subplot(2, 2, 3)
    
    n_groups_data = 3  # グループの数
    bar_width = 0.8 / n_groups_data  # バーの幅を調整
    group_data = [y_true[0], y_pred_1[0], y_pred_2[0]]
    for i in range(n_groups_data):
        ax3.bar(x + (i - n_groups_data/2) * bar_width, group_data[i], width=bar_width, label=f'Group {i+1}')

    ax3.grid()
    ax3.legend(['Ground-Truth', 'Prediction-1', 'Prediction-2'])


    ax4 = fig.add_subplot(2, 2, 4)
    
    n_groups_data = 4  # グループの数
    bar_width = 0.8 / n_groups_data  # バーの幅を調整
    group_data = [loss_cross_1[0], loss_binary_1[0], loss_cross_2[0], loss_binary_2[0]]
    for i in range(n_groups_data):
        ax4.bar(x + (i - n_groups_data/2) * bar_width, group_data[i], width=bar_width, label=f'Group {i+1}')

    ax4.grid()
    ax4.legend([
        'Cross Entropy of pred1', 'Binary Cross Entropy of pred1', 
        'Cross Entropy of pred2', 'Binary Cross Entropy of pred2'
    ])

    fig.savefig(args.image_dst_path)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--image_dst_path',
        '-idp',
        type=str,
        default='./image_dst.png',
        help=''
    )
    args = parser.parse_args()

    main(args)