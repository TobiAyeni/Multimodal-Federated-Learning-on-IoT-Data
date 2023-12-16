import os
import torch
import torch.nn.functional as F
import multiprocessing

from torch import nn
from torchvision.models import resnet18
from torchvision import transforms
from PIL import Image


class LSTMEncoder(nn.Module):
    def __init__(self, input_size, representation_size, num_layers=1, batch_first=True):
        super(LSTMEncoder, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=representation_size,
                            num_layers=num_layers, batch_first=batch_first)
        nn.init.orthogonal_(self.lstm.weight_ih_l0)
        nn.init.orthogonal_(self.lstm.weight_hh_l0)

    def forward(self, x):
        out, _ = self.lstm(x)
        return out


class LSTMDecoder(nn.Module):
    def __init__(self, representation_size, output_size, num_layers=1, batch_first=True):
        super(LSTMDecoder, self).__init__()
        self.lstm = nn.LSTM(input_size=representation_size, hidden_size=output_size,
                            num_layers=num_layers, batch_first=batch_first)
        nn.init.orthogonal_(self.lstm.weight_ih_l0)
        nn.init.orthogonal_(self.lstm.weight_hh_l0)

    def forward(self, x):
        out, _ = self.lstm(x)
        return out


class LSTMAutoEncoder(nn.Module):
    def __init__(self, input_size, representation_size, num_layers=1, batch_first=True):
        super(LSTMAutoEncoder, self).__init__()
        self.batch_first = batch_first
        self.encoder = LSTMEncoder(
            input_size=input_size, representation_size=representation_size, num_layers=num_layers, batch_first=batch_first)
        self.decoder = LSTMDecoder(representation_size=representation_size,
                                   output_size=input_size, num_layers=num_layers, batch_first=batch_first)

    def forward(self, x):
        seq_len = x.shape[1] if self.batch_first else x.shape[0]
        out = self.encoder(x)
        representation = out[:, -1,
                             :].unsqueeze(1) if self.batch_first else out[-1, :, :].unsqueeze(0)
        representation_seq = representation.expand(-1, seq_len, -1)
        x_prime = self.decoder(representation_seq)
        return x_prime

    def encode(self, x):
        x = self.encoder(x)
        return x


class DCCLSTMAutoEncoder(nn.Module):
    def __init__(self, input_size_A, input_size_B, representation_size, num_layers=1, batch_first=True):
        super(DCCLSTMAutoEncoder, self).__init__()
        self.batch_first = batch_first
        self.encoder_A = LSTMEncoder(
            input_size=input_size_A, representation_size=representation_size, num_layers=num_layers, batch_first=batch_first)
        self.decoder_A = LSTMDecoder(representation_size=representation_size,
                                     output_size=input_size_A, num_layers=num_layers, batch_first=batch_first)
        self.encoder_B = LSTMEncoder(
            input_size=input_size_B, representation_size=representation_size, num_layers=num_layers, batch_first=batch_first)
        self.decoder_B = LSTMDecoder(representation_size=representation_size,
                                     output_size=input_size_B, num_layers=num_layers, batch_first=batch_first)

    def forward(self, x_A=None, x_B=None):
        """Takes the input from two modalities and forwards.

        Args:
            x_A: input tensor of modality A
            x_B: input tensor of modality B

        Returns:
            A tuple containing the rep_A, rep_B, x_prime_A, and x_prime_B
        """
        if x_A != None:
            # Forward in the modality A pipe line
            seq_len_A = x_A.shape[1]
            out_A = self.encoder_A(x_A)
            rep_A = out_A[:, -1,
                          :].unsqueeze(1) if self.batch_first else out_A[-1, :, :].unsqueeze(0)
            rep_seq_A = rep_A.expand(-1, seq_len_A, -1)
            x_prime_A = self.decoder_A(rep_seq_A)

            if x_B == None:
                return(rep_A.squeeze(), None, x_prime_A, None)

        if x_B != None:
            # Forward in the modality B pipe line
            seq_len_B = x_B.shape[1]
            out_B = self.encoder_B(x_B)
            rep_B = out_B[:, -1,
                          :].unsqueeze(1) if self.batch_first else out_B[-1, :, :].unsqueeze(0)
            rep_seq_B = rep_B.expand(-1, seq_len_B, -1)
            x_prime_B = self.decoder_B(rep_seq_B)
            if x_A == None:
                return(None, rep_B.squeeze(), None, x_prime_B)

        return (rep_A.squeeze(), rep_B.squeeze(), x_prime_A, x_prime_B)

    def encode(self, x, modality):
        assert (modality == "A" or modality ==
                "B"), "Modality is neither A nor B"
        out = self.encoder_A(x) if modality == "A" else self.encoder_B(x)
        return out


class SplitLSTMAutoEncoder(nn.Module):
    def __init__(self, input_size_A, input_size_B, representation_size, num_layers=1, batch_first=True):
        super(SplitLSTMAutoEncoder, self).__init__()
        self.batch_first = batch_first
        self.encoder_A = LSTMEncoder(
            input_size=input_size_A, representation_size=representation_size, num_layers=num_layers, batch_first=batch_first)
        self.decoder_A = LSTMDecoder(representation_size=representation_size,
                                     output_size=input_size_A, num_layers=num_layers, batch_first=batch_first)
        self.encoder_B = LSTMEncoder(
            input_size=input_size_B, representation_size=representation_size, num_layers=num_layers, batch_first=batch_first)
        self.decoder_B = LSTMDecoder(representation_size=representation_size,
                                     output_size=input_size_B, num_layers=num_layers, batch_first=batch_first)

    def forward(self, x, modality):
        assert (modality == "A" or modality ==
                "B"), "Modality is neither A nor B"

        seq_len = x.shape[1] if self.batch_first else x.shape[0]
        out = self.encoder_A(x) if modality == "A" else self.encoder_B(x)
        representation = out[:, -1, :].unsqueeze(
            1) if self.batch_first else out[-1, :, :].unsqueeze(0)
        representation_seq = representation.expand(-1, seq_len, -1)
        x_prime_A = self.decoder_A(representation_seq)
        x_prime_B = self.decoder_B(representation_seq)
        return (x_prime_A, x_prime_B)

    def encode(self, x, modality):
        assert (modality == "A" or modality ==
                "B"), "Modality is neither A nor B"
        out = self.encoder_A(x) if modality == "A" else self.encoder_B(x)
        return out


class MLP(nn.Module):
    def __init__(self, input_size, n_classes, dropout=0.0):
        super(MLP, self).__init__()
        self.n_classes = n_classes
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(input_size, n_classes)

    def forward(self, x):
        out = self.fc(self.dropout(x))
        out = out.contiguous().view(-1, self.n_classes)
        return F.log_softmax(out, dim=1)


class ResNetMapper(nn.Module):
    resnet = resnet18(pretrained=True).double()
    resnet_mapper = nn.Sequential(*list(resnet.children())[:-1])

    @classmethod
    def map(cls, idxs):
        imgs = ur_fall_idxs_to_imgs(idxs)
        cls.resnet_mapper.eval()

        with torch.no_grad():
            x = cls.resnet_mapper(imgs)
            x = x.view(x.size(0), -1)

        return x


def process_one(one_file):
    idx_frame, f_img = one_file
    img = Image.open(f_img)
    return (idx_frame, img)


def ur_fall_idxs_to_imgs(idxs):
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), ])

    t_imgs = torch.empty(
        (idxs.shape[0], 3, 224, 224), dtype=torch.float64)
    f_list = []
    for idx_frame, frame in enumerate(idxs):
        is_fall = "adl" if frame[0] == 0 else "fall"
        run = int(frame[1])
        frame_num = int(frame[2])
        f_img = os.path.join(f"data/ur_fall", is_fall, f"cam0-rgb", f"{is_fall}-{str(run).zfill(2)}-cam0-rgb",
                             f"{is_fall}-{str(run).zfill(2)}-cam0-rgb-{str(frame_num).zfill(3)}.png")
        f_list.append((idx_frame, f_img))
    with multiprocessing.Pool(8) as p:
        results = p.map(process_one, f_list)
    for r in results:
        t_imgs[r[0]] = preprocess(r[1]).double()
    return t_imgs
