import crypten
import crypten.mpc as mpc
import crypten.communicator as comm


# Simulate two parties running secure logistic regression
@mpc.run_multiprocess(world_size=2)
def logistic_regression(w_enc, sample, alpha):
    party = comm.get().get_rank()
    # Create shares of the initial w value (weights)
    print(f'Party {party}: has w share {w_enc}')

    # Choose a random sample and secret share it
    print(f'The training sample is: {sample}')
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
    print(f'Party {party}: the updated w value is: {w_enc}, plaintext: {w_enc.get_plain_text()}')
