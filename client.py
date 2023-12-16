import torch
import numpy as np

from torch import nn
from utils import get_seg_len, make_seq_batch
from losses import DCCLoss

ALPHA = 100


class Client:
    def __init__(self, client_train, client_train_idx, modality, config):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.train = client_train
        self.modality = modality  # "A", "B", or "AB"
        self.train_idx = client_train_idx
        self.seg_len = get_seg_len(len(client_train["A"]), config)
        self.lr = float(config["CLIENT"]["lr"])
        self.n_epochs = int(config["CLIENT"]["num_epochs"])
        self.optimizer = config["CLIENT"]["optimizer"]
        self.criterion = config["CLIENT"]["criterion"]
        self.model_ae = config["SIMULATION"]["model_ae"]
        self.rep_size = int(config["FL"]["rep_size"])
        self.DCCAE_lamda = float(config["FL"]["DCCAE_lamda"])
        if config["SIMULATION"]["data"] == "ur_fall":
            self.batch_min = 16
            self.batch_max = 32
        else:
            self.batch_min = 128
            self.batch_max = 256

    def freeze(self, sub_model):
        """Freeze the parameters of a model"""
        for param in sub_model.parameters():
            param.requires_grad = False

    def unfreeze(self, sub_model):
        """Unfreeze the parameters of a model"""
        for param in sub_model.parameters():
            param.requires_grad = True

    def train_split_ae(self, model, optimizer, criterion, A_train, B_train, idx_start, idx_end):
        """Trains split bi-modal autoencoder"""
        sub_epoch_losses = []
        if self.modality == "A" or self.modality == "AB":
            x_A = A_train[:, idx_start:idx_end, :]
            seq_A = torch.from_numpy(x_A).double().to(self.device)
            inv_idx = torch.arange(seq_A.shape[1]-1, -1, -1).long()

        if self.modality == "B" or self.modality == "AB":
            x_B = B_train[:, idx_start:idx_end, :]
            seq_B = torch.from_numpy(x_B).double().to(self.device)
            inv_idx = torch.arange(seq_B.shape[1]-1, -1, -1).long()

        optimizer.zero_grad()
        if self.modality == "A":
            self.freeze(model.encoder_B)
            self.freeze(model.decoder_B)

            output, _ = model(seq_A, "A")
            loss = criterion(output, seq_A[:, inv_idx, :])
            sub_epoch_losses.append(loss.item())
            loss.backward()
            optimizer.step()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            self.unfreeze(model.encoder_B)
            self.unfreeze(model.decoder_B)
        elif self.modality == "B":
            self.freeze(model.encoder_A)
            self.freeze(model.decoder_A)

            _, output = model(seq_B, "B")
            loss = criterion(output, seq_B[:, inv_idx, :])
            sub_epoch_losses.append(loss.item())
            loss.backward()
            optimizer.step()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            self.unfreeze(model.encoder_A)
            self.unfreeze(model.decoder_A)
        elif self.modality == "AB":
            # Train with input of modality A and output of modalities A&B
            self.freeze(model.encoder_B)
            output_A, output_B = model(seq_A, "A")
            loss_A = criterion(output_A, seq_A[:, inv_idx, :])
            loss_B = criterion(output_B, seq_B[:, inv_idx, :])
            loss = loss_A + loss_B
            sub_epoch_losses.append(loss.item())
            loss.backward()
            optimizer.step()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self.unfreeze(model.encoder_B)

            # Train with input of modality B and output of modalities A&B
            self.freeze(model.encoder_A)
            output_A, output_B = model(seq_B, "B")
            loss_A = criterion(output_A, seq_A[:, inv_idx, :])
            loss_B = criterion(output_B, seq_B[:, inv_idx, :])
            loss = loss_A + loss_B
            sub_epoch_losses.append(loss.item())
            loss.backward()
            optimizer.step()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self.unfreeze(model.encoder_A)

        return sub_epoch_losses

    def train_dcc_ae(self, model, optimizer, dcc_criterion, A_train, B_train, idx_start, idx_end):
        """Trains deep canonically correlated autoencoder (DCCAE)"""
        sub_epoch_losses = []
        if self.modality == "A" or self.modality == "AB":
            x_A = A_train[:, idx_start:idx_end, :]
            seq_A = torch.from_numpy(x_A).double().to(self.device)
            inv_idx = torch.arange(seq_A.shape[1]-1, -1, -1).long()

        if self.modality == "B" or self.modality == "AB":
            x_B = B_train[:, idx_start:idx_end, :]
            seq_B = torch.from_numpy(x_B).double().to(self.device)
            inv_idx = torch.arange(seq_B.shape[1]-1, -1, -1).long()

        mse_criterion = nn.MSELoss().to(self.device)
        optimizer.zero_grad()
        if self.modality == "A":
            self.freeze(model.encoder_B)
            self.freeze(model.decoder_B)

            _, _, output, _ = model(x_A=seq_A)
            loss = mse_criterion(output, seq_A[:, inv_idx, :])
            sub_epoch_losses.append(loss.item())
            loss.backward()
            optimizer.step()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            self.unfreeze(model.encoder_B)
            self.unfreeze(model.decoder_B)
        elif self.modality == "B":
            self.freeze(model.encoder_A)
            self.freeze(model.decoder_A)

            _, _, _, output = model(x_B=seq_B)
            loss = mse_criterion(output, seq_B[:, inv_idx, :])
            sub_epoch_losses.append(loss.item())
            loss.backward()
            optimizer.step()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            self.unfreeze(model.encoder_A)
            self.unfreeze(model.decoder_A)
        elif self.modality == "AB":
            # Train with input of modalities A&B and output of modalities A&B
            rep_A, rep_B, output_A, output_B = model(x_A=seq_A, x_B=seq_B)
            loss_A = mse_criterion(output_A, seq_A[:, inv_idx, :])
            loss_B = mse_criterion(output_B, seq_B[:, inv_idx, :])
            loss_dcc = dcc_criterion.loss(rep_A, rep_B)
            loss = loss_dcc + self.DCCAE_lamda*(loss_A + loss_B)
            sub_epoch_losses.append(loss.item())
            loss.backward()
            optimizer.step()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return sub_epoch_losses

    def get_weight(self):
        """Gets the training weight of the client"""
        # Since all clients have same amount of local data, it's same as local
        # data size=1 for all. So the weight for multimodal clients is ALPHA and
        # the weight for unimodal clients is 1.
        if self.modality == "AB":
            return ALPHA
        else:
            return 1

    def update(self, model):
        """Updates the global model using local data.

        Args:
            model: a deep copy of the global AE model of the server

        Returns:
            A tuple containing the updated local model, its weight, and local training loss
        """

        if self.criterion == "MSELoss":
            criterion = nn.MSELoss().to(self.device)
        elif self.criterion == "DCCAELoss":
            criterion = DCCLoss(self.rep_size, self.device)
        if self.optimizer == "Adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)

        model.train()
        round_loss = []
        for epoch in range(self.n_epochs):
            epoch_losses = []

            batch_size = np.random.randint(
                low=self.batch_min, high=self.batch_max)
            A_train, B_train, _ = make_seq_batch(
                self.train, self.train_idx, self.seg_len, batch_size)
            # A_train and B_train both are in the shape of (batch_size, seq_len, input_size), i.e., batch first
            seq_len = A_train.shape[1]

            idx_start = 0
            idx_end = 0
            while idx_end < seq_len:
                win_len = np.random.randint(low=16, high=32)
                idx_start = idx_end
                idx_end += win_len
                idx_end = min(idx_end, seq_len)
                if self.model_ae == "split_LSTM":
                    sub_epoch_losses = self.train_split_ae(
                        model, optimizer, criterion, A_train, B_train, idx_start, idx_end)
                elif self.model_ae == "DCCAE_LSTM":
                    sub_epoch_losses = self.train_dcc_ae(
                        model, optimizer, criterion, A_train, B_train, idx_start, idx_end)
                epoch_losses.extend(sub_epoch_losses)
            round_loss.append(np.mean(epoch_losses))

        return model, self.get_weight(), np.mean(round_loss)
