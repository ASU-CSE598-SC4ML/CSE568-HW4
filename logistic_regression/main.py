import crypten
from logistic_regression import logistic_regression
from random import choice

training_samples = [[4, 10],
                    [6, 12],
                    [14, 20],
                    [16, 22]]
alpha = 0.3


def main():
    crypten.init()

    print('Secure logistic regression training starting!')

    # Randomly iterate through all training samples (without repetition)
    while len(training_samples) != 0:
        sample = choice(training_samples)
        training_samples.remove(sample)
        logistic_regression(10, sample, 0.3)

    print('Training finished!')


if __name__ == '__main__':
    main()
