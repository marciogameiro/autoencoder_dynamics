from autoencoder_model import *
import numpy as np

# Read training data from file
XY = np.loadtxt('train_data.csv', delimiter=',')

# Split each row into x and y vectors
XY_data = np.array([[np.array(xy[:N]), np.array(xy[N:])] for xy in XY])

# Move the training dataset to the selected device
XY_data = torch.tensor(np.array(XY_data), dtype=torch.float32).to(device)

# Training loop
print('Training autoencoder ...')

# Put model in training mode
model.train()

for epoch in range(num_epochs):
    optimizer.zero_grad()
    # Get the initial x states
    x_current = XY_data[:, 0, :]
    # Get the x states after one step
    x_forward = XY_data[:, 1, :]
    # Encode the initial states
    z_current = model.encoder(x_current)
    # Step forward in latent space
    z_forward = model.latent_map(z_current)

    # Reconstruction Loss
    x_current_dec = model.decoder(z_current)
    loss = criterion(x_current_dec, x_current)

    # Decode z forward to full space
    x_forward_dec = model.decoder(z_forward)
    # Add the prediction error to the total loss
    loss += criterion(x_forward_dec, x_forward)

    loss.backward()
    optimizer.step()

    # Tell the scheduler to check the loss at the end of every epoch
    scheduler.step(loss)

    if epoch % 100 == 0:
        print(f'Epoch {epoch}/{num_epochs} | Total Loss: {loss.item():0.6f}')

# Save model weights
torch.save(model.state_dict(), 'ci_model_weights.pth')
