import torch
import torch.nn as nn
import utils

device = utils.get_default_device()

class Encoder(nn.Module):
    def __init__(self, input_size, latent_size):
        super().__init__()
        self.input_size = input_size
        self.latent_size = latent_size
        self._define_layer(self.input_size, self.latent_size)

    def _define_layer(self, input_size, latent_size):
        self.linear1 = nn.Linear(input_size, input_size//2)
        self.linear2 = nn.Linear(input_size//2, input_size//4)
        self.linear3 = nn.Linear(input_size//4, latent_size)
        self.relu = nn.ReLU(True)

    def forward(self, w):
        out = self.linear1(w)
        out = self.relu(out)
        out = self.linear2(out)
        out = self.relu(out)
        out = self.linear3(out)
        z = self.relu(out)

        return z
    
class Decoder(nn.Module):
    def __init__(self, latent_size, out_size):
        super().__init__()
        self.latent_size = latent_size
        self.out_size = out_size
        self._define_layers(self.latent_size, self.out_size)

    def _define_layers(self, latent_size, out_size):
        self.linear1 = nn.Linear(latent_size, out_size//4)
        self.linear2 = nn.Linear(out_size//4, out_size//2)
        self.linear3 = nn.Linear(out_size//2, out_size)
        self.relu = nn.ReLU(True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, z):
        out = self.linear1(z)
        out = self.relu(out)
        out = self.linear2(out)
        out = self.relu(out)
        out = self.linear3(out)
        w = self.sigmoid(out)

        return w
    

class UsadModel(nn.Module):
    def __init__(self, w_size, z_size):
        super().__init__()
        self.encoder = Encoder(w_size, z_size)
        self.decoder1 = Decoder(z_size, w_size)
        self.decoder2 = Decoder(z_size, w_size)

    def train_step(self, batch, n):
        z = self.encoder(batch)
        w1 = self.decoder1(z)
        w2 = self.decoder2(z)
        w3 = self.decoder2(self.encoder(w1))

        # MSE

        loss1 = 1/n*torch.mean((batch-w1)**2)+(1-1/n)*torch.mean((batch-w3)**2)
        loss2 = 1/n*torch.mean((batch-w2)**2)-(1-1/n)*torch.mean((batch-w3)**2)
        

        return loss1, loss2
    
    def validation_step(self, batch, n):
        with torch.no_grad():
            z = self.encoder(batch)
            w1 = self.decoder1(z)
            w2 = self.decoder2(z)
            w3 = self.decoder2(self.encoder(w1))
            # MSE
            loss1 = 1/n*torch.mean((batch-w1)**2)+(1-1/n)*torch.mean((batch-w3)**2)
            loss2 = 1/n*torch.mean((batch-w2)**2)-(1-1/n)*torch.mean((batch-w3)**2)
            

        return {'val_loss1':loss1, 'val_loss2':loss2}
    
    def validation_epoch_end(self, outpus):
        batch_losses1 = [x['val_loss1'] for x in outpus]
        epoch_loss1 = torch.stack(batch_losses1).mean()
        batch_losses2 = [x['val_loss2'] for x in outpus]
        epoch_loss2 = torch.stack(batch_losses2).mean()

        return {'val_loss1': epoch_loss1.item(), 'val_loss2': epoch_loss2.item()}
    
    def epoch_end(self, epoch, result):
        print("Epoch [{}], val_loss1: {:.4f}, val_loss2: {:.4f}".format(epoch, result['val_loss1'], result['val_loss2']))

def evaluate(model, val_loader, n):
    outputs = [model.validation_step(utils.to_device(batch, device), n) for [batch] in val_loader]
    return model.validation_epoch_end(outputs)
    
def training(epochs, model, train_loader, val_loader, opt_func=torch.optim.Adam):
    history = []
    optimizer1 = opt_func(list(model.encoder.parameters())+ list(model.decoder1.parameters()), lr=0.1)
    optimizer2 = opt_func(list(model.encoder.parameters())+ list(model.decoder2.parameters()), lr=0.1)

    for epoch in range(epochs):
        for [batch] in train_loader:
            batch=utils.to_device(batch, device)

            # train AE1
            loss1, loss2 = model.train_step(batch, epoch+1)
            loss1.backward()
            optimizer1.step()
            optimizer1.zero_grad()

            # train AE2
            loss1, loss2 = model.train_step(batch, epoch+1)
            loss2.backward()
            optimizer2.step()
            optimizer2.zero_grad()

        result = evaluate(model, val_loader, epoch+1)
        model.epoch_end(epoch+1, result)
        history.append(result)

    return history

def testing(model, test_loader, alpha=.5, beta=.5):
    results = []
    with torch.no_grad():
        for [batch] in test_loader:
            batch = utils.to_device(batch, device)
            w1 = model.decoder1(model.encoder(batch))
            w2 = model.decoder2(model.encoder(w1))
            results.append(alpha*torch.mean((batch-w1)**2, axis=1)+beta*torch.mean((batch-w2)**2, axis=1))

    return results