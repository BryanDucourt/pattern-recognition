from matplotlib import pyplot as plt
import numpy as np

result = np.load('mlp.npy', allow_pickle=True)
dict_res = result.item()
print('i')

layer_size = dict_res['mean_test_score'][
    np.intersect1d(np.where(dict_res['param_solver'] == 'lbfgs'), np.where(dict_res['param_activation'] == 'tanh'))]
l_layer_size = dict_res['param_hidden_layer_sizes'][
    np.intersect1d(np.where(dict_res['param_solver'] == 'lbfgs'), np.where(dict_res['param_activation'] == 'tanh'))]
plt.plot(layer_size)
plt.show()
