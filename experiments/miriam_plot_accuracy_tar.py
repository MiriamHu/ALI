import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from blocks.serialization import load
import datetime
import json
import os.path

__author__ = 'mhuijser'

def load_log(tar_file):
    print("Opening .tar file")
    with open(tar_file, 'rb') as src:
        main_loop_loaded = load(src)
    log = main_loop_loaded.log
    print("Finished opening .tar file")
    return log

def get_accuracies(log, epoch_ends, accuracies_to_plot):
    print("Getting accuracies from .log file")
    all_accuracies = dict()
    for accuracy in accuracies_to_plot:
        all_accuracies[accuracy] = []

    for ep_end in epoch_ends:
        for accuracy in accuracies_to_plot:
            all_accuracies[accuracy].append(float(log[ep_end][accuracy]))
    print("Finished getting accuracies from .log file")
    return all_accuracies

def plot_accuracies(plot_file, title, epochs, accuracies):
    print("Start plotting")
    plt.plot(epochs, accuracies["train_ali_get_predictions_data_accuracy"], 'r.', label='Data accuracy train')
    plt. plot(epochs, accuracies["valid_ali_get_predictions_data_accuracy"], 'rs', label='Data accuracy validation', markersize=3)
    plt.plot(epochs, accuracies["train_ali_get_predictions_sample_accuracy"], 'b.', label='Sample accuracy train')
    plt. plot(epochs, accuracies["valid_ali_get_predictions_sample_accuracy"], 'bs', label='Sample accuracy validation', markersize=3)
    plt.suptitle(title, fontsize=20)
    plt.xlabel('Epochs', fontsize=18)
    plt.ylabel('Accuracy', fontsize=16)
    #plt.ylim(min([min(accuracies[key]) for key in accuracies.keys()]), max([max(accuracies[key]) for key in accuracies.keys()]))
    #plt.ylim(min([min(accuracies[key]) for key in accuracies.keys()]), min(6,max([max(accuracies[key]) for key in accuracies.keys()])))
    plt.ylim(0,1)
    plt.legend(loc="upper center", bbox_to_anchor=(0.5,1.05))
    plt.savefig(plot_file, bbox_inches="tight")
    print("Finished plotting and saving figure")

title = "ALI SVHN 0, 8"
#tar_file = '/var/scratch/aiir-mh/ali_svhn108/ali_svhn_108_30.tar'
tar_file = "/var/scratch/aiir-mh/ali_svhn108_04dropout_try2/ali_svhn108_04dropout_try2_100.tar"
spl_file = tar_file.split(os.extsep)
epoch = tar_file.split("_")[-1].split(os.extsep)[0]
json_file = os.path.join(os.path.dirname(tar_file), "accuracies_" + epoch + ".json")
#plot_file = "PLOT_%s.pdf" % datetime.datetime.now().strftime("%Y-%m-%d_%H%M")
plot_file = "plot_accuracies_%s.pdf" % os.path.basename(spl_file[0])
if os.path.isfile(json_file):
    with open(json_file) as data_file:
        data = json.load(data_file)
    plot_accuracies(plot_file, title, data["epochs"], data["accuracies"])
else:
    accuracies_to_plot = ["train_ali_get_predictions_data_accuracy",
                    "valid_ali_get_predictions_data_accuracy",
                    "train_ali_get_predictions_sample_accuracy",
                    "valid_ali_get_predictions_sample_accuracy"
                   ]
    log = load_log(tar_file)
    epoch_ends = log.status["_epoch_ends"]
    accuracies = get_accuracies(log, epoch_ends, accuracies_to_plot)
    epochs = range(log.status["epochs_done"])
    plot_accuracies(plot_file, title, epochs, accuracies)
    with open(json_file, 'w') as outfile:
        json.dump({"epochs":epochs, "accuracies":accuracies}, outfile)

