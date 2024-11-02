import matplotlib.pyplot as plt

def visualize(anchor, positive, negative):
    def show(ax, image):
        ax.imshow(image)
        ax.axis('off')

    fig, axs = plt.subplots(3, 3, figsize=(9, 9))
    for i in range(3):
        show(axs[i, 0], anchor[i])
        show(axs[i, 1], positive[i])
        show(axs[i, 2], negative[i])
    plt.show()
