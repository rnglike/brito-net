import numpy as np
import matplotlib.pyplot as plt
import func as f
import helper

def train(input_size, hidden_size, output_size, lr, momentum, epochs, goal_mse, train="net/train.txt", val="net/val.txt", log=""):

    # Plotting data
    accuracy_history = []
    loss_history = []

    # Target: One-hot encoding for digits 0–9
    T = np.eye(output_size)

    # Load training data
    P = np.loadtxt(train, dtype=np.uint8)
    print(f"Loaded training data with shape: {P.shape}")

    # Load test data
    N = np.loadtxt(val, dtype=np.uint8)
    print(f"Loaded test data with shape: {N.shape}")

    # Load labels for training and test data
    plabels = np.array([], dtype=int)
    nlabels = np.array([], dtype=int)

    with open('net/labels.train.txt', 'r') as file:
        plabels = file.read().rstrip().split('\n')
        plabels = np.fromstring(plabels[0], dtype=int, sep=' ')

    with open('net/labels.val.txt', 'r') as file:
        nlabels = file.read().rstrip().split('\n')
        nlabels = np.fromstring(nlabels[0], dtype=int, sep=' ')

    print(f"Loaded {len(plabels)} training labels and {len(nlabels)} validation labels.")
    print('')

    # Weights and biases
    W1 = np.random.randn(hidden_size, input_size) * 0.1
    b1 = np.zeros((hidden_size, 1))
    W2 = np.random.randn(output_size, hidden_size) * 0.1
    b2 = np.zeros((output_size, 1))

    # Momentum terms
    dW1_prev = np.zeros_like(W1)
    db1_prev = np.zeros_like(b1)
    dW2_prev = np.zeros_like(W2)
    db2_prev = np.zeros_like(b2)

    # Training loop
    for epoch in range(epochs):
        total_error = 0

        for i in range(P.shape[1]):  # Loop over each input sample
            x = P[:, [i]]            # Input column vector (20x1)
            t = T[:, [i]]            # Target column vector (10x1)

            # ---- FORWARD PASS ----
            z1 = np.dot(W1, x) + b1          # Hidden layer input
            a1 = f.tansig(z1)                # Hidden layer activation

            z2 = np.dot(W2, a1) + b2         # Output layer input
            a2 = f.purelin(z2)               # Output layer activation (linear)

            # ---- ERROR ----
            e = t - a2                       # Output error
            total_error += np.sum(e**2)      # Accumulate squared error

            # ---- BACKWARD PASS ----
            delta2 = e * f.dpurelin(z2)                     # Output layer delta
            delta1 = np.dot(W2.T, delta2) * f.dtansig(z1)   # Hidden layer delta

            # ---- GRADIENTS ----
            dW2 = lr * np.dot(delta2, a1.T) + momentum * dW2_prev
            db2 = lr * delta2 + momentum * db2_prev
            dW1 = lr * np.dot(delta1, x.T) + momentum * dW1_prev
            db1 = lr * delta1 + momentum * db1_prev

            # ---- UPDATE WEIGHTS ----
            W2 += dW2
            b2 += db2
            W1 += dW1
            b1 += db1

            # Save gradients for momentum
            dW2_prev = dW2
            db2_prev = db2
            dW1_prev = dW1
            db1_prev = db1

        # Calculate average error for the epoch
        avg_error = total_error / P.shape[1]
        loss_history.append(avg_error)

        # Calculate accuracy on the training set
        correct_predictions = 0
        for i in range(N.shape[1]):
            x = N[:, [i]]
            z1 = np.dot(W1, x) + b1
            a1 = f.tansig(z1)
            z2 = np.dot(W2, a1) + b2
            a2 = f.purelin(z2)

            pred = plabels[np.argmax(a2)]
            grd = nlabels[i] if i < len(nlabels) else nlabels[-1] # Use last label if oob

            if pred == grd:
                correct_predictions += 1

        accuracy = correct_predictions / N.shape[1]
        accuracy_history.append(accuracy)

        # Print error every 100 epochs
        if epoch % (epochs / 10) == 0:
            print(f"Epoch {epoch}, MSE: {avg_error}")

        # Check if goal MSE is reached
        if avg_error < goal_mse:
            print(f"Goal MSE reached at epoch {epoch}. Stopped!")
            break

    # Save network parameters
    params_path = "net/params.txt"
    np.savetxt(params_path, [input_size, hidden_size, output_size], fmt='%d')

    # Save trained weights and biases as .txt
    weights_path = "net/best.txt"
    np.savetxt(weights_path, np.hstack((W1.flatten(), b1.flatten(), W2.flatten(), b2.flatten())), fmt='%.6f')

    print('')
    print(f"Trained weights and biases saved to {weights_path}")

    helper.plot_against_epochs(
        (loss_history, accuracy_history),
        name=log
    )

def run(data_path, data_labels_path, params_path="net/params.txt", weights_path="net/best.txt", log=""):

    # Load network parameters
    params = np.loadtxt(params_path, dtype=int)
    input_size, hidden_size, output_size = params

    # Load labels
    plabels = np.array([], dtype=int)
    nlabels = np.array([], dtype=int)

    with open('net/labels.train.txt', 'r') as file:
        plabels = file.read().rstrip().split('\n')
        plabels = np.fromstring(plabels[0], dtype=int, sep=' ')

    with open(data_labels_path, 'r') as file:
        nlabels = file.read().rstrip().split('\n')
        nlabels = np.fromstring(nlabels[0], dtype=int, sep=' ')

    # Load trained weights and biases
    W1 = np.loadtxt(weights_path, max_rows=hidden_size * input_size).reshape(hidden_size, input_size)
    b1 = np.loadtxt(weights_path, skiprows=hidden_size * input_size, max_rows=hidden_size).reshape(hidden_size, 1)
    W2 = np.loadtxt(weights_path, skiprows=hidden_size * input_size + hidden_size, max_rows=output_size * hidden_size).reshape(output_size, hidden_size)
    b2 = np.loadtxt(weights_path, skiprows=hidden_size * input_size + hidden_size + output_size * hidden_size, max_rows=output_size).reshape(output_size, 1)

    # Load input data
    P = np.loadtxt(data_path, dtype=np.uint8)

    # Ensure P is a 2D array
    if P.ndim == 1:
        P = P.reshape(-1, 1)

    results = []

    # Test the network with the training data
    print("\nTesting network:")
    for i in range(P.shape[1]):
        x = P[:, [i]]
        z1 = np.dot(W1, x) + b1
        a1 = f.tansig(z1)
        z2 = np.dot(W2, a1) + b2
        a2 = f.purelin(z2)

        pred = plabels[np.argmax(a2)]
        grd = nlabels[i] if i < len(nlabels) else nlabels[-1]  # Use last label if out of bounds

        print(f"Input = {grd}, Network Output = {pred}")

        results.append((grd, pred))

    np.savetxt(f"net/results_{log}.txt", results, fmt='%d', header='Ground Truth, Prediction', comments='')