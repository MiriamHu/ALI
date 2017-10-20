import matplotlib
matplotlib.use('Agg')
from blocks.extensions import SimpleExtension
import os.path
import logging
import shutil
import matplotlib.pyplot as plt
import json
logger = logging.getLogger(__name__)

LOADED_FROM = "loaded_from"
SAVED_TO = "saved_to"
BACKED_UP_TO = "backed_up_to"

__author__ = 'mhuijser'

class BackupModel(SimpleExtension):
    def __init__(self, backup_path, main_model_path, **kwargs):
        kwargs.setdefault("after_training", True)
        super(BackupModel, self).__init__(**kwargs)
        self.backup_path = backup_path
        self.main_model_path = main_model_path

    def do(self, callback_name, *args):
        """Copy the pickled main loop to a specific location with epic number in file name.

        If `*args` contain an argument from user, it is treated as
        saving path to be used instead of the one given at the
        construction stage.

        """
        epic_number = self.main_loop.log.status["epochs_done"]
        logger.info("Backing up the model has started")
        try:
            if not os.path.isdir(self.backup_path):
                os.mkdir(self.backup_path)
            backup_file = os.path.join(self.backup_path, os.path.basename(self.main_model_path))
            backup_file = backup_file.split(os.extsep)
            backup_file = backup_file[0] + "_" + str(epic_number) + "." + backup_file[1]
            shutil.copyfile(self.main_model_path, backup_file)
        except Exception:
            raise
        finally:
            logger.info("Backing up the model has finished")

class PlotLoss(SimpleExtension):
    def __init__(self, save_path, title_plot, **kwargs):
        kwargs.setdefault("after_training", True)
        super(PlotLoss, self).__init__(**kwargs)
        self.save_path = save_path
        self.losses_to_plot = ["valid_ali_compute_losses_generator_loss",
                               "train_ali_compute_losses_generator_loss",
                               "valid_ali_compute_losses_discriminator_loss",
                               "train_ali_compute_losses_discriminator_loss"]
        self.title_plot = title_plot

    def do(self, callback_name, *args):
        logger.info("Plotting the loss has started")

        epoch_number = self.main_loop.log.status["epochs_done"]
        epoch_ends = self.main_loop.log.status["_epoch_ends"]
        epochs = range(epoch_number)
        losses = self.get_losses(epoch_ends, self.losses_to_plot)

        json_file = os.path.join(self.save_path, "losses_"+str(epoch_number)+".json")
        self.save_to_json(json_file, epochs, losses)

        plot_file = os.path.join(self.save_path, "plot_"+str(epoch_number)+".pdf")
        self.plot_losses(plot_file, epochs, losses)

        logger.info("Plotting the loss has finished")

    def get_losses(self, epoch_ends, losses_to_plot):
        all_losses = dict()
        for loss in losses_to_plot:
            all_losses[loss] = []

        for ep_end in epoch_ends:
            for loss in losses_to_plot:
                all_losses[loss].append(float(self.main_loop.log[ep_end][loss]))
        return all_losses

    def plot_losses(self, plot_file, epochs, losses):
        plt.figure()
        logger.info("Start plotting")
        plt.plot(epochs, losses["train_ali_compute_losses_generator_loss"], 'r.', label='Generator train loss')
        plt. plot(epochs, losses["valid_ali_compute_losses_generator_loss"], 'rs', label='Generator validation loss', markersize=3)
        plt.plot(epochs, losses["train_ali_compute_losses_discriminator_loss"], 'b.', label='Discriminator train loss')
        plt. plot(epochs, losses["valid_ali_compute_losses_discriminator_loss"], 'bs', label='Discriminator validation loss', markersize=3)
        plt.suptitle(self.title_plot, fontsize=20)
        plt.xlabel('Epochs', fontsize=18)
        plt.ylabel('Loss', fontsize=16)
        plt.ylim(min([min(losses[key]) for key in losses.keys()]), max([max(losses[key]) for key in losses.keys()]))
        plt.legend(loc="upper center", bbox_to_anchor=(0.5,1.05))
        plt.savefig(plot_file, bbox_inches="tight")
        logger.info("Finished plotting and saving figure")

    def save_to_json(self, json_file, epochs, losses):
        logger.info("Backing up losses to json")
        with open(json_file, 'w') as outfile:
            json.dump({"epochs":epochs, "losses":losses}, outfile)
        logger.info("Backing up losses to json finished")


class PlotAccuracy(SimpleExtension):
    def __init__(self, save_path, title_plot, **kwargs):
        kwargs.setdefault("after_training", True)
        super(PlotAccuracy, self).__init__(**kwargs)
        self.save_path = save_path
        self.accuracies_to_plot = ["train_ali_get_predictions_data_accuracy",
                    "valid_ali_get_predictions_data_accuracy",
                    "train_ali_get_predictions_sample_accuracy",
                    "valid_ali_get_predictions_sample_accuracy"
                   ]
        self.title_plot = title_plot

    def do(self, callback_name, *args):
        logger.info("Plotting the accuracy has started")

        epoch_number = self.main_loop.log.status["epochs_done"]
        epoch_ends = self.main_loop.log.status["_epoch_ends"]
        epochs = range(epoch_number)
        accuracies = self.get_accuracies(epoch_ends, self.accuracies_to_plot)

        json_file = os.path.join(self.save_path, "accuracies_"+str(epoch_number)+".json")
        self.save_to_json(json_file, epochs, accuracies)

        plot_file = os.path.join(self.save_path, "plot_accuracies_"+str(epoch_number)+".pdf")
        self.plot_accuracies(plot_file, self.title_plot, epochs, accuracies)

        logger.info("Plotting the loss has finished")

    def get_accuracies(self, epoch_ends, accuracies_to_plot):
        print("Getting accuracies from .log file")
        all_accuracies = dict()
        for accuracy in accuracies_to_plot:
            all_accuracies[accuracy] = []

        for ep_end in epoch_ends:
            for accuracy in accuracies_to_plot:
                all_accuracies[accuracy].append(float(self.main_loop.log[ep_end][accuracy]))
        print("Finished getting accuracies from .log file")
        return all_accuracies

    def plot_accuracies(self, plot_file, title, epochs, accuracies):
        print("Start plotting")
        plt.figure()
        plt.plot(epochs, accuracies["train_ali_get_predictions_data_accuracy"], 'r.', label='Data accuracy train')
        plt.plot(epochs, accuracies["valid_ali_get_predictions_data_accuracy"], 'rs', label='Data accuracy validation', markersize=3)
        plt.plot(epochs, accuracies["train_ali_get_predictions_sample_accuracy"], 'b.', label='Sample accuracy train')
        plt.plot(epochs, accuracies["valid_ali_get_predictions_sample_accuracy"], 'bs', label='Sample accuracy validation', markersize=3)
        plt.suptitle(title, fontsize=20)
        plt.xlabel('Epochs', fontsize=18)
        plt.ylabel('Accuracy', fontsize=16)
        #plt.ylim(min([min(accuracies[key]) for key in accuracies.keys()]), max([max(accuracies[key]) for key in accuracies.keys()]))
        #plt.ylim(min([min(accuracies[key]) for key in accuracies.keys()]), min(6,max([max(accuracies[key]) for key in accuracies.keys()])))
        plt.ylim(0,1)
        plt.legend(loc="upper center", bbox_to_anchor=(0.5,1.05))
        plt.savefig(plot_file, bbox_inches="tight")
        print("Finished plotting and saving figure")

    def save_to_json(self, json_file, epochs, accuracies):
        logger.info("Backing up accuracies to json")
        with open(json_file, 'w') as outfile:
            json.dump({"epochs":epochs, "accuracies":accuracies}, outfile)
        logger.info("Backing up accuracies to json finished")