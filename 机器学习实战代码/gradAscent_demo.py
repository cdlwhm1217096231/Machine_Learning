# -*- coding：utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt


def Gradient_Ascent_test():
    def f_prime(x_old):                                    # f(x)的导数
        return -2 * x_old + 4
    x_old = -1                                            # 初始值，给一个小于x_new的值
    x_new = 0                                            # 梯度上升算法初始值，即从(0,0)开始
    alpha = 0.01                                        # 步长，也就是学习速率，控制更新的幅度
    presicion = 0.00000001                                # 精度，也就是更新阈值
    while abs(x_new - x_old) > presicion:
        x_old = x_new
        x_new = x_old + alpha * f_prime(x_old)            # 上面提到的公式
    print('梯度上升法求得的极大值点:', x_new)                                        # 打印最终求解的极值近似值


if __name__ == '__main__':
    Gradient_Ascent_test()
