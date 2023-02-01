import os
import time
import detector
import random
from skimage import io
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as func
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter



class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(40, 200)
        self.fc2 = nn.Linear(200, 800)
        self.fc3 = nn.Linear(800, 2400)
        self.fc4 = nn.Linear(2400, 4800)
        self.fc5 = nn.Linear(4800, 2400)
        self.fc6 = nn.Linear(2400, 800)
        self.fc7 = nn.Linear(800, 200)
        self.fc8 = nn.Linear(200, 40)

    def forward(self, x):
        x = func.relu(self.fc1(x))
        x = func.relu(self.fc2(x))
        x = func.relu(self.fc3(x))
        x = func.relu(self.fc4(x))
        x = func.relu(self.fc5(x))
        x = func.relu(self.fc6(x))
        x = func.relu(self.fc7(x))
        x = func.relu(self.fc8(x))
        return x


class NetworkManager:
    @staticmethod
    def format_input(input_):
        input_ = input_.astype(np.float32)
        return input_.flatten()

    @staticmethod
    def format_label(input_, label_):
        label_ = NetworkManager.format_input(label_)
        return label_ - input_

    @staticmethod
    def format_output(input_, output_):
        output_ = output_.detach().numpy()[0]
        input_ = input_.detach().numpy()[0]
        output_ = input_ + output_

        temp_output = np.zeros(shape=(20, 2), dtype=np.float32)
        for i in range(20):
            temp_output[i] = [output_[i * 2], output_[i * 2 + 1]]
        output_ = temp_output

        return output_.astype(np.int32)

    def __init__(self):
        self.net = None


class NetworkTrainer(NetworkManager):
    @staticmethod
    def create_samples(file_name):
        def to_str(num):
            if num < 10:
                num = "0" + str(num)
            else:
                num = str(num)
            return num

        all_i = range(1, 50)
        all_j = [1, 14]
        i_size = len(all_i)
        j_size = len(all_j)
        n_samples = i_size * j_size
        inputs = np.zeros(shape=(n_samples, 40), dtype=np.float32)
        labels = np.zeros(shape=(n_samples, 40), dtype=np.float32)

        print("reading all images", end='', flush=True)
        for i in all_i:
            for j in all_j:
                neutral_image = io.imread('%s/../data/img/M-0%s-%s.bmp' % (os.getcwd(), to_str(i), to_str(j)))
                smile_image = io.imread('%s/../data/img/M-0%s-%s.bmp' % (os.getcwd(), to_str(i), to_str(j + 1)))
                _, _, neutral_coords = detector.get_face_infos(neutral_image)
                _, _, smile_coords = detector.get_face_infos(smile_image)

                neutral_coords = neutral_coords[48:]
                smile_coords = smile_coords[48:]
                index = (i - 1) * j_size + (0 if j == 1 else 1)
                inputs[index] = NetworkManager.format_input(neutral_coords)
                labels[index] = NetworkManager.format_label(inputs[index], smile_coords)

                print(".", end='', flush=True)

        print(" : done")
        print("shape=" + str(inputs.shape) + "  " + str(labels.shape))
        np.savez('%s/../data/samples/%s' % (os.getcwd(), file_name), inputs=inputs, labels=labels)

        return inputs, labels

    @staticmethod
    def load_samples(file_name):
        data = np.load('%s/../data/samples/%s.npz' % (os.getcwd(), file_name))
        return data['inputs'], data['labels']

    def __init__(self, samples_name="default", training_portion=0.8):
        assert isinstance(training_portion, float) and 0 < training_portion <= 1
        super().__init__()

        try:
            self.inputs, self.labels = self.load_samples(samples_name)
        except (FileNotFoundError, KeyError):
            self.inputs, self.labels = self.create_samples(samples_name)
        assert len(self.inputs) == len(self.labels)

        n = int(training_portion * len(self.inputs))
        self.training_inputs = self.inputs[:n]
        self.training_labels = self.labels[:n]
        self.test_inputs = self.inputs[n:]
        self.test_labels = self.labels[n:]

    def get_training_batch(self, batch_portion):
        n = int(batch_portion * len(self.training_inputs))
        batch_inputs = random.choices(self.training_inputs, k=n)
        batch_labels = random.choices(self.training_labels, k=n)
        return torch.from_numpy(np.array(batch_inputs)), torch.from_numpy(np.array(batch_labels))

    def get_testing_batch(self):
        return torch.from_numpy(np.array(self.test_inputs)), torch.from_numpy(np.array(self.test_labels))

    def test(self, board, index):
        assert self.net is not None

        criterion = nn.SmoothL1Loss()
        batch_inputs, batch_labels = self.get_testing_batch()
        outputs = self.net(batch_inputs)
        batch_loss = criterion(outputs, batch_labels).item()
        board.add_scalar("Loss/Testing", batch_loss, index)

    def train(self, name="default", n_batch=2000, batch_portion=1., is_save=True, is_backups=False,
              is_verbose=False):
        assert isinstance(batch_portion, float) and 0 < batch_portion <= 1

        self.net = Net()
        criterion = nn.SmoothL1Loss()
        optimizer = optim.SGD(self.net.parameters(), lr=1e-2, momentum=0.9)

        running_loss = 0.0
        batch_loss = 0.0
        board = None
        if is_verbose:
            board = SummaryWriter(log_dir="runs/%s" % name)

        old_time = None
        i_backup = None
        backup_frequency = None

        if is_backups:
            old_time = time.process_time()
            i_backup = 0
            backup_frequency = 300

        for i in range(n_batch):
            batch_inputs, batch_labels = self.get_training_batch(batch_portion)
            optimizer.zero_grad()
            outputs = self.net(batch_inputs)

            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()

            batch_loss = loss.item()
            running_loss += batch_loss

            if is_backups:
                current_time = time.process_time()
                if current_time > old_time + backup_frequency:
                    torch.save(self.net, '%s/../data/nn/backups/nn-%s-v%i' % (os.getcwd(), name, i_backup))
                    i_backup += 1
                    old_time = current_time

            if is_verbose:
                board.add_scalar("Loss/Training", batch_loss, i)
                self.test(board, i)

        if is_verbose:
            print("Total loss average : %.3f" % (running_loss / n_batch))
        if is_save:
            torch.save(self.net, '%s/../data/nn/%s' % (os.getcwd(), name))


class NetworkPredictor(NetworkManager):
    def __init__(self, net=None):
        super().__init__()
        self.net = net

    def to_input(self, image):
        input_ = np.zeros(shape=(1, 40), dtype=np.float32)

        _, _, coords = detector.get_face_infos(image)
        coords = coords[48:]
        input_[0] = self.format_input(coords)
        return torch.from_numpy(input_)

    def predict(self, image, name="default"):
        if self.net is None:
            try:
                self.net = torch.load('%s/../data/nn/%s' % (os.getcwd(), name))
            except FileNotFoundError:
                trainer = NetworkTrainer()
                trainer.train(name)
                self.net = trainer.net

        input_ = self.to_input(image)
        return self.format_output(input_, self.net(input_))

