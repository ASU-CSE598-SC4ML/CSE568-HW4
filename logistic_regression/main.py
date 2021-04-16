import crypten
from logistic_regression import logistic_regression
from random import choice

training_samples = [[4, 10],
                    [6, 12],
                    [14, 20],
                    [16, 22]]
alpha = 0.3
init_w = 10


def main():
    crypten.init()
    w_enc = crypten.cryptensor([init_w])
    print('Secure logistic regression training starting!')

    # Randomly iterate through all training samples (without repetition)
    while len(training_samples) != 0:
        sample = choice(training_samples)
        training_samples.remove(sample)
        logistic_regression(w_enc, sample, alpha)

    print('Training finished!')
    print(f'The end result (w) is: {w_enc}, plaintext: {w_enc.get_plain_text()}')


if __name__ == '__main__':
    main()
