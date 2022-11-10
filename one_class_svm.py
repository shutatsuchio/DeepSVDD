import numpy as np
import random

def meet_limit_condition(alpha_i, data_i, a, R, C, toler):
    # alphas[i]が最適化条件を満たすかどうかをテストする。
    # :param alpha_i:alphas[i]
    # :param data_i:data_array[i]
    # :param a:中心点
    # :param R:半径
    # :param C:マージンパラメータ
    # :param toler:許容度
    # :return:最適化条件を満たした場合はTrueを、それ以外の場合はFalseを返す
    # if abs(R ** 2 - np.dot((data_i - a), (data_i - a))) > toler and 0 < alpha_i < C:
    Ei = R ** 2 - np.dot((data_i - a), (data_i - a))
    if (Ei < -toler and alpha_i < C) or (Ei > toler and alpha_i > 0):
        return True
    else:
        return False

def selectJrand(i, m):
    # ランダムな整数を選択する
    # Args:
    #     i  最初のαの添え字
    #     m  全αの数
    # Returns:
    #     j  iではない乱数（0〜mまでの整数値）を返す。
    j = i
    while j == i:
        j = int(random.uniform(0, m))
    return j


def calculate_alpha_j(data_array, alphas, i, j, a):
    # data_array: テストセット
    # alphas:旧式α值
    # i, j: 現在選択されている最適化されるアルファの添え字
    # 返り值: 新しいalphas[j]の値
    a1 = np.array(a)
    x1 = np.array(data_array[i])
    x2 = np.array(data_array[j])

    x12 = np.dot(x1, x2)
    x1_2 = np.dot(x1, x1)
    x2_2 = np.dot(x2, x2)

    nu = np.dot(a1, x2) - x2_2 - np.dot(a1, x1) + x1_2 + \
        alphas[i] * (x12 + x1_2) + alphas[j] * (x1_2 - x2_2 + 3 * x12)
    de = 2 * (x1_2 + x2_2 - 2 * x12)

    if de == 0:
        return 0, False

    return -nu / de, True


def calculate_alpha_i(alphas, i):
    # alphas: 新しいα配列
    # i: 更新されるアルファ値の添え字
    # 返り值： 新しいalphas[i]
    alpha_sum = alphas.sum() - alphas[i]
    return 1 - alpha_sum

def smo(train_data, C, toler, maxIter):
    data_array = np.array(train_data)
    m, n = np.shape(data_array)

    alphas = np.array([1 / m] * m)
    R = 0

    a = np.array([0.0] * n)
    for i in range(m):
        a += alphas[i] * data_array[i]

    iter = 0
    while iter < maxIter:
        changed_flag = 0
        for i in range(m):
            if meet_limit_condition(alphas[i], data_array[i], a, R, C, toler):
                j = selectJrand(i, m)

                L = max(0, alphas[i] + alphas[j] - C)
                H = min(C, alphas[i] + alphas[j])
                if L == H:
                    continue

                new_alpha_j, valid = calculate_alpha_j(
                    data_array, alphas, i, j, a)
                if not valid:
                    continue

                if new_alpha_j < L:
                    new_alpha_j = L
                elif new_alpha_j > H:
                    new_alpha_j = H

                if abs(new_alpha_j - alphas[j]) < 0.001:
                    continue
                else:
                    alphas[j] = new_alpha_j
                    alphas[i] = calculate_alpha_i(alphas, i)
                    changed_flag += 1

                # check_alphas(alphas, C)

                a, R = calculate_a_and_R(data_array, alphas, i, j, C)

        if changed_flag == 0:
            iter += 1
        else:
            iter = 0

    return a, R,alphas


def check_alphas(alphas, C):
    """
    アルファの適合性判定
    :param alphas:alphas
    :param C:マージンパラメータ
    :return:True、それ以外はFalse
    """
    a_sum = 0
    for i in range(alphas.shape[0]):
        if alphas[i] < -0.0001:
            print("alphas" + str(i) + ":" + str(alphas[i]) + " < 0")
        if alphas[i] > C + 0.0001:
            print("alphas" + str(i) + ":" + str(alphas[i]) + " > C")
        a_sum += alphas[i]

    if abs(a_sum - 1) > 0.0001:
        print("alphas sum != 1")
        return False
    else:
        return True


def calculate_a_and_R(data_array, alphas, i, j, C):
    # 算出方法a, R
    # :param data_array:
    # :param alphas:
    # :param i:
    # :param j:
    # :param C:
    # :return:
    m, n = np.shape(data_array)
    a = [0] * n
    for l in range(m):
        a += data_array[l] * alphas[l]

    R1 = np.sqrt(np.dot(data_array[i] - a, data_array[i] - a))
    R2 = np.sqrt(np.dot(data_array[j] - a, data_array[j] - a))
    if 0 < alphas[i] < C:
        R = R1
    elif 0 < alphas[j] < C:
        R = R2
    else:
        R = (R1 + R2) / 2.0

    return a, R