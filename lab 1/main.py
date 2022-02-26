import math
import numpy as np


def golden_section_search(func, l, r, eps):
    phi = 1.618
    current_l = l
    current_r = r
    while current_r - current_l > eps:
        diff = current_r - current_l
        candidate_l = current_r - diff / phi
        candidate_r = current_l + diff / phi
        if func(candidate_l) > func(candidate_r):
            current_l = candidate_l
        else:
            current_r = candidate_r
    return (current_l + current_r) / 2


def grad_descent(derivative_func, step_func, start_point, iterations):
    current_point = start_point
    for it in range(iterations):
        step = step_func(it)
        cur_der = derivative_func(current_point)
        current_point -= step * cur_der
    return current_point


def const_learning_rate(value):
    return lambda i: value


def exp_learning_rate():
    return lambda i: math.exp(-i)


def stair_learning_rate(value):
    return lambda i: 1 / (1 + i // value)


def sample_1():
    print("Sample 1")

    func = lambda xs: xs[0] ** 2
    print("Function: [x^2]")

    derivative = lambda xs: np.array([2 * xs[0]])
    print("Derivative function: [2*x]")

    start_point = np.array([1.05])
    print(f"Start point: {start_point}")

    lr_v = 0.1
    learning_rate_func = const_learning_rate(lr_v)
    print(f"Constant learning rate {lr_v}")

    iterations = 100
    print(f"Iterations count {iterations}")

    result = grad_descent(derivative, learning_rate_func, start_point, iterations)
    print(f"Found minimum point: {result}")

    print()


def sample_2():
    print("Sample 2")

    func = lambda xs: ((xs[0] + 3) ** 2) + ((xs[1] - 5) ** 2)
    print("Function: [((x+3)^2)+((y-5)^2)]")

    derivative = lambda xs: np.array([2 * (xs[0] + 3), 2 * (xs[1] - 5)])
    print("Derivative function: [2*(x+3), 2*(y-5)]")

    start_point = np.array([10.0, -3.0])
    print(f"Start point: {start_point}")

    learning_rate_func = exp_learning_rate()
    print(f"Exponential learning rate")

    iterations = 50
    print(f"Iterations count {iterations}")

    result = grad_descent(derivative, learning_rate_func, start_point, iterations)
    print(f"Found minimum point: {result}")

    print()


def sample_3():
    print("Sample 3")

    func = lambda x: (x + 5) ** 4
    print("Function: [(x+5)^4]")

    eps = 0.2
    print(f"Epsilon {eps}")

    l = -100
    r = 100
    print(f"Interval [{l}, {r}]")

    result = golden_section_search(func, l, r, eps)
    print(f"Found minimum point: {result}")

    print()


def main():
    sample_1()
    sample_2()
    sample_3()


if __name__ == '__main__':
    main()
