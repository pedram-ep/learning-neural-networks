# Helper functions for the mini-project of comparing different regularization methods
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def plot_loss_acc(history):
    """
    Input: History of loss and accuracy over epochs of training, for both Training and Evaluation Dataset

    Draws the plot of these values using matplotlib
    """
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    train_acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    
    epochs = range(1, len(train_loss) + 1)
    
    fig, axs = plt.subplots(2, figsize=(10, 7))
    fig.suptitle('Training and Validation Metrics')
    
    # Loss plot
    axs[0].plot(epochs, train_loss, label='Training loss', c='red')
    axs[0].plot(epochs, val_loss, label='Validation loss', c='blue')
    axs[0].set_title('Loss')
    axs[0].set_xlabel('Epochs')
    axs[0].set_ylabel('Loss')
    axs[0].legend()
    axs[0].set_ylim([0, 1.5])
    
    # Accuracy plot
    axs[1].plot(epochs, train_acc, label='Training accuracy', c='red')
    axs[1].plot(epochs, val_acc, label='Validation accuracy', c='blue')
    axs[1].set_title('Accuracy')
    axs[1].set_xlabel('Epochs')
    axs[1].set_ylabel('Accuracy')
    axs[1].legend()
    
    plt.subplots_adjust(hspace=0.5)
    plt.show()

def get_decision_boundaries(model, xmin, xmax, ymin, ymax, steps):
    x_span = np.linspace(xmin, xmax, steps)
    y_span = np.linspace(ymin, ymax, steps)
    xx, yy = np.meshgrid(x_span, y_span)
    points = (np.stack([xx.ravel(), yy.ravel()], axis=1).astype(np.float32))

    z = (model.predict(np.c_[xx.ravel(), yy.ravel()])>0.5).reshape(xx.shape)

    return xx, yy, z

def plot_decision_boundaries(model, x_min, x_max, y_min, y_max, steps):
    plt.figure(figsize=(6, 4))
    xx, yy, z = get_decision_boundaries(model, x_min, x_max, y_min, y_max, steps)
    plt.contourf(xx, yy, z, alpha=0.2, cmap=matplotlib.colors.ListedColormap(["C1", "C0"]));

def plot_loss_acc_on_axes(history, ax_loss, ax_acc, title=""):
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    train_acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    
    epochs = range(1, len(train_loss) + 1)
    
    # Loss plot
    ax_loss.plot(epochs, train_loss, label='Training loss', c='red')
    ax_loss.plot(epochs, val_loss, label='Validation loss', c='blue')
    ax_loss.set_title(f'{title} - Loss')
    ax_loss.set_xlabel('Epochs')
    ax_loss.set_ylabel('Loss')
    ax_loss.legend()
    ax_loss.set_ylim([0, 1.5])
    
    # Accuracy plot
    ax_acc.plot(epochs, train_acc, label='Training accuracy', c='red')
    ax_acc.plot(epochs, val_acc, label='Validation accuracy', c='blue')
    ax_acc.set_title(f'{title} - Accuracy')
    ax_acc.set_xlabel('Epochs')
    ax_acc.set_ylabel('Accuracy')
    ax_acc.legend()

def plot_decision_boundaries_on_axes(model, ax, x_min, x_max, y_min, y_max, steps,
                                    data_x, data_y, title=""):
    xx, yy, z = get_decision_boundaries(model, x_min, x_max, y_min, y_max, steps)
    ax.contourf(xx, yy, z, alpha=0.2, cmap=matplotlib.colors.ListedColormap(["C1", "C0"]));
    colors = ['orange' if label == 0 else 'purple' for label in data_y.ravel()]
    ax.scatter(data_x[:, 0], data_x[:, 1], c=colors, s=4, edgecolors='none')
    ax.set_title(title)
    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_min, y_max])