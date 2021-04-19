from random import choice

import crypten
import crypten.communicator as comm
import crypten.mpc as mpc


class LogisticRegression:
    # Simulate two parties running secure logistic regression
    @mpc.run_multiprocess(world_size=2)
    def train(self, init_w, training_samples, alpha):
        w_enc = crypten.cryptensor([init_w])
        itn = 0

        print(f'Initial w is {init_w}')
        print('Secure logistic regression training starting!')

        # Randomly iterate through all training samples (without repetition)
        while len(training_samples) != 0:
            itn += 1
            print(f'\n\nIteration {itn}')

            sample = choice(training_samples)
            training_samples.remove(sample)
            print(f'The training sample is: {sample}')

            party = comm.get().get_rank()

            # Choose a random sample and secret share it
            sample_enc = crypten.cryptensor(sample, ptype=mpc.arithmetic)
            print(f'Party {party}: has sample share {sample_enc}')

            # Step 1: compute x * w
            s1 = sample_enc[0] * w_enc

            # Step 2: compute f(x * w), where f is a custom function that replaces the sigmoid function
            # If s1 < -0.5 -> f = 0
            #       >  0.5 -> f = 1
            #         else -> f = s1 + 0.5
            if (s1 < -0.5).get_plain_text().item() == 1.0:
                s2 = crypten.cryptensor([0], ptype=mpc.arithmetic)
            elif (s1 > 0.5).get_plain_text().item() == 1.0:
                s2 = crypten.cryptensor([1], ptype=mpc.arithmetic)
            else:
                s2 = s1 + 0.5

            # Step 3: compute f(x * w) - y
            s3 = s2 - sample_enc[1]

            # Step 4: compute (f(x * w) - y)*w
            s4 = s3 * w_enc

            # Step 5: compute a((f(x * w) - y)*w)
            s5 = s4 * alpha

            # Step 6: update w
            w_enc = w_enc - s5
            print(f'\nParty {party}: the updated w value is: {w_enc}, plaintext: {w_enc.get_plain_text()}')

        print('\nTraining finished!')
        print(f'\nThe end result (w) is: {w_enc}, plaintext: {w_enc.get_plain_text()}')
