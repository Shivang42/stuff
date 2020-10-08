import matplotlib.pyplot as plt
import pickle

nn = pickle.load(open('a_nn.pkl','rb'))
plt.plot(range(nn.epochs),nn.eval_['cost'])
plt.ylabel('Cost')
plt.xlabel('Epochs')
plt.show()
plt.plot(range(nn.epochs),nn.eval_['train_acc'],label='training')
plt.plot(range(nn.epochs),nn.eval_['valid_acc'],label='validation',linestyle='--')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend()
plt.show()
print(nn.eval_['valid_acc'])
