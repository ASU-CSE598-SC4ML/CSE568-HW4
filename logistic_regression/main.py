import crypten
import torch

from logistic_regression import LogisticRegression

training_samples = [[4, 10],
                    [6, 12],
                    [14, 20],
                    [16, 22]]
alpha = 0.3
init_w = 10


def main():
    # Init Crypten and disable OpenMP threads (needed by @mpc.run_multiprocess
    crypten.init()
    torch.set_num_threads(1)

    lr = LogisticRegression()
    lr.train(init_w, training_samples, alpha)


if __name__ == '__main__':
    main()
