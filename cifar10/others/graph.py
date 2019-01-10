import matplotlib.pyplot as plt

data = []#insert data here
epochs = range(1,51)
val = data[:, 0]#first column
train =data[:, 1]#second column
plt.title("Training and Validation accuracy over 50 Epochs")
plt.plot(epochs,val,c="r",label="Validation accuracy")
plt.plot(epochs,train,c="b",label="Training accuracy")
plt.xticks((epochs))
plt.legend()
plt.xlabel('Epoch #')
plt.ylabel('accuracy \n (1.00 = 100%)')
plt.show()