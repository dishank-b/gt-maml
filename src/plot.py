import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy.stats import norm
import seaborn as sns
# %matplotlib inline
import os

sns.set_style("white")

# shading = False

# map_dict = {
#     "maml": "MAML",
#     "first_order_maml": "FOMAML",
#     "pareto_maml": "Pareto MAML",
#     "pareto_maml_first_order": " FO Pareto MAML"
# }

# method_plot = ["maml",  "first_order_maml", "pareto_maml", "pareto_maml_first_order"]

map_dict = {
    "maml_sinusoid": "MAML",
    "maml_sinusoid_first_order": "FOMAML",
    "pareto_sinusoid": "Pareto MAML",
    "pareto_sinusoid_first_order": " FO Pareto MAML"
}

method_plot = ["maml_sinusoid",  "maml_sinusoid_first_order", "pareto_sinusoid", "pareto_sinusoid_first_order"]

# fig, ax1 = plt.subplots()

for i in range(5):
    fig, ax1 = plt.subplots()
    for method in method_plot:
        # arr = np.load(os.path.join("../logs", method, "mean_outer_loss_val.npy"))
        # arr = np.load(os.path.join("../logs", method, "accuracies_before.npy"))
        arr = np.load(os.path.join("../logs", method, "inner_losses.npy"))

        ax1.plot(arr[:,:,i].reshape(-1), label = map_dict[method])
        # ax1.plot(arr, label = map_dict[method])

    plt.ylabel("Accuracy")
    plt.xlabel("Epochs")
    # plt.xlim(0, 50)
    plt.legend(prop={'size': 10})
    plt.savefig("{}_inner_sinusoid.png".format(i))
    plt.clf()
