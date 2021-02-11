import torch
import torch.nn as nn

# ######################################################################################################################
#                                                   NTXent
# ######################################################################################################################
class NTXent(nn.Module):
    def __init__(self, batchSize, temperature):

        super().__init__()

        self.batchSize   = batchSize
        self.temperature = temperature
        self.__mask      = self.__mask_correlated_samples()

        self.crossEntropy     = nn.CrossEntropyLoss(reduction="sum")
        self.cosineSimilarity = nn.CosineSimilarity(dim=2)

    # ================================================== MASK CORR SAMPLES =============================================
    def __mask_correlated_samples(self):

        b = self.batchSize
        N = 2 * b
        mask = torch.ones((N, N), dtype=torch.bool).fill_diagonal_(0)
        for i in range(self.batchSize):
            mask[b + i,     i] = 0
            mask[    i, b + i] = 0
        return mask

    # ================================================== FORWARD =======================================================
    def forward(self, zA, zB):

        N = 2 * self.batchSize

        z   = torch.cat((zA, zB), dim=0)
        sim = self.cosineSimilarity(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature

        simAB = torch.diag(sim,  self.batchSize)
        simBA = torch.diag(sim, -self.batchSize)

        # We have 2N samples, but with Distributed training every GPU gets N examples too, resulting in: 2xNxN
        posSamples = torch.cat((simAB, simBA), dim=0).reshape(N, 1)
        nevSamples = sim[self.__mask].reshape(N, -1)

        labels = torch.zeros(N).to(posSamples.device).long()
        logits = torch.cat((posSamples, nevSamples), dim=1)
        loss   = self.crossEntropy(logits, labels) / N

        return loss