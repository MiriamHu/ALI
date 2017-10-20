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

def get_losses(log, epoch_ends, losses_to_plot):
    print("Getting losses from .log file")
    all_losses = dict()
    for loss in losses_to_plot:
        all_losses[loss] = []

    for ep_end in epoch_ends:
        for loss in losses_to_plot:
            all_losses[loss].append(float(log[ep_end][loss]))
    print("Finished getting losses from .log file")
    return all_losses

def plot_losses(plot_file, title, epochs, losses):
    print("Start plotting")
    plt.plot(epochs, losses["train_ali_compute_losses_generator_loss"], 'r.', label='Generator train loss')
    plt. plot(epochs, losses["valid_ali_compute_losses_generator_loss"], 'rs', label='Generator validation loss', markersize=3)
    plt.plot(epochs, losses["train_ali_compute_losses_discriminator_loss"], 'b.', label='Discriminator train loss')
    plt. plot(epochs, losses["valid_ali_compute_losses_discriminator_loss"], 'bs', label='Discriminator validation loss', markersize=3)
    plt.suptitle(title, fontsize=20)
    plt.xlabel('Epochs', fontsize=18)
    plt.ylabel('Loss', fontsize=16)
    # plt.ylim(min([min(losses[key]) for key in losses.keys()]), max([max(losses[key]) for key in losses.keys()]))
    plt.ylim(min([min(losses[key]) for key in losses.keys()]), min(6,max([max(losses[key]) for key in losses.keys()])))
    plt.legend(loc="upper center", bbox_to_anchor=(0.5,1.05))
    plt.savefig(plot_file, bbox_inches="tight")
    print("Finished plotting and saving figure")

# title = "ALI SVHN 0, 8"
title = "ALI Handbags vs Shoes"
tar_file = "/var/scratch/aiir-mh/ali_handbags_shoes_celeba_04dropout_100lat/" \
           "ali_handbags_shoes_celeba_04dropout_100lat_100.tar"
# tar_file = "/var/scratch/aiir-mh/ali_handbags_shoes_celeba_04dropout_256lat/" \
#            "ali_handbags_shoes_celeba_04dropout_256lat_100.tar"
spl_file = tar_file.split(os.extsep)
epoch = tar_file.split("_")[-1].split(os.extsep)[0]
json_file = os.path.join(os.path.dirname(tar_file), "losses_" + epoch + ".json")
plot_file = "plot_loss_%s.pdf" % os.path.basename(spl_file[0])
#plot_file = "PLOT_%s.pdf" % datetime.datetime.now().strftime("%Y-%m-%d_%H%M")
if os.path.isfile(json_file):
    with open(json_file) as data_file:
        data = json.load(data_file)
    plot_losses(plot_file, title, data["epochs"], data["losses"])
else:
    losses_to_plot = ["valid_ali_compute_losses_generator_loss",
                   "train_ali_compute_losses_generator_loss",
                   "valid_ali_compute_losses_discriminator_loss",
                   "train_ali_compute_losses_discriminator_loss"]
    log = load_log(tar_file)
    epoch_ends = log.status["_epoch_ends"]
    losses = get_losses(log, epoch_ends, losses_to_plot)
    epochs = range(log.status["epochs_done"])
    plot_losses(plot_file, title, epochs, losses)
    with open(json_file, 'w') as outfile:
        json.dump({"epochs":epochs, "losses":losses}, outfile)
