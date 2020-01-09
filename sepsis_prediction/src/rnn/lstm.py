import logging
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

class MyLSTM(nn.Module):
    def __init__(self, input_size,  batch_size, hidden_size=12, num_layers=1):
        super(MyLSTM, self).__init__()
        self.batch_size = batch_size
        self.hidden_size = hidden_size

        self.model = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True,
                             bidirectional=True)

        self.fc = nn.Sequential(
            nn.Linear(in_features=2*hidden_size, out_features=int(hidden_size/2)),
            nn.Tanh(),
            nn.Dropout(),
            nn.Linear(in_features=int(hidden_size/2), out_features=2)
        )

    def forward(self, input_tuple):
        seqs, lengths = input_tuple
        logger.debug("seqs shape: {}".format(seqs.shape))
        out, (ht, _) = self.model(seqs)
        logger.debug("out shape: {}".format(out.shape))
        logger.debug("ht shape: {}".format(ht.shape))
        out = out[:,-1,:]
        logger.debug("out shape: {}".format(out.shape))
        y_pred = self.fc(out)
        sm = F.log_softmax(y_pred)
        logger.debug(sm)
        return sm
