import matplotlib.pyplot as plt

if __name__ == "__main__":
    plt.plot([1, 2, 3, 4])
    plt.xlabel("Epoch")
    plt.ylabel("Classification Test Error")
    plt.show()
    # plt.savefig("test.png")