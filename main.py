import encode
import net
import os

if not os.path.exists("net"):
    os.makedirs("net")

encode.prepare("train", "net/digitos.txt")
encode.prepare("test", "net/test.txt")
encode.prepare("val", "net/val.txt")

encode.write([1, 2, 3, 4, 5, 6, 7, 8, 9, 0], "net/labels.train.txt")
encode.write([1], "net/labels.test.txt")
encode.write([2], "net/labels.val.txt")

networks = {"A": 15, "B": 25, "C": 35}
trainings = {"1": (0.1, 0.0), "2": (0.4, 0.0), "3": (0.9, 0.0), "4": (0.1, 0.4), "5": (0.9, 0.4)}
pairs = [(n, t) for n in networks for t in trainings]

for network, training in pairs:

    net.train(
        input_size=20,
        hidden_size=networks[network],
        output_size=10,
        lr=trainings[training][0],
        momentum=trainings[training][1],
        epochs=10000,
        goal_mse=0.0005,
        train="net/digitos.txt",
        log=str(network[0]) + str(training[0])
    )

    net.run(
        data_path="net/test.txt",
        data_labels_path="net/labels.test.txt",
        log=str(network[0]) + str(training[0]),
    )