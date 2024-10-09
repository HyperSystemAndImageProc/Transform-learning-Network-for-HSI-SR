step for the paper "Spectral domain strategies for hyperspectral super-resolution: Transfer learning and channel enhance network"

Step 1: Toggle the pseudo dataset in "data_loader.py"

Step 2: Train weights with prior knowledge on the pseudo dataset

Step 3: Switch real dataset in "data_loader.py" again

Step 4: Load pre-trained weights using "torch.load".

Step 5: Train on the real dataset to fully exploit the potential of the network.

The steps for the preparation of the pseudo dataset are in section 3.1 of the article. (https://doi.org/10.1016/j.jag.2024.104180)
