import torch.nn as nn

def get_encoder(input_dim, encoder_hidden, nonlinearity):
    modules = []

    for h in encoder_hidden:
        modules.append(nn.Linear(input_dim, h))
        modules.append(nonlinearity())
        input_dim = h

    return nn.Sequential(*modules)

def get_decoder(input_dim, output_dim, decoder_hidden, nonlinearity):
    modules = []

    for h in decoder_hidden:
        modules.append(nn.Linear(input_dim, h))
        modules.append(nonlinearity())
        input_dim = h

    modules.append(nn.Linear(decoder_hidden[-1], output_dim))
    return nn.Sequential(*modules)
