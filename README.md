step for the paper "Spectral domain strategies for hyperspectral super-resolution: Transfer learning and channel enhance network"

Step 1: Toggle the pseudo dataset in "data_loader.py"

Step 2: Train weights with prior knowledge on the pseudo dataset

Step 3: Switch real dataset in "data_loader.py" again

Step 4: Load pre-trained weights using "torch.load".

Step 5: Train on the real dataset to fully exploit the potential of the network.
