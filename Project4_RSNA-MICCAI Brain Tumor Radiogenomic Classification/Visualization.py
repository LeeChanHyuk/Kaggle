import matplotlib.pyplot as plt
epochs = [i+1 for i in range(10)]
train_loss = [0.51410, 0.51410, 0.51410, 0.51410, 0.51410, 0.51410, 0.51410, 0.51410, 0.51410, 0.51410]
valid_loss = [0.54310, 0.54310, 0.54310, 0.54310,   0.54310, 0.54310, 0.54310, 0.49138, 0.49138, 0.49138]
plt.plot(epochs, train_loss,'b')
plt.plot(epochs, valid_loss,'r')
plt.xlabel('Epoch')
plt.ylabel('Score')
plt.legend(['train','valid'])
plt.show()