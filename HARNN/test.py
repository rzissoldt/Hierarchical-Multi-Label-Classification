import numpy as np
from scipy.stats import chi2

class LambdaUpdater:
    def __init__(self, num_layers, initial_k, final_k, num_epochs):
        self.num_layers = num_layers
        self.initial_k = initial_k
        self.final_k = final_k
        self.current_epoch = 0
        self.k = initial_k
        self.num_epochs = num_epochs
        self.step_size = (self.final_k - self.initial_k) / num_epochs

    def update_lambdas(self):
        self.k = self.initial_k + self.step_size * self.current_epoch

    def get_lambda_values(self):
        x_values = np.linspace(10**-16, self.final_k, self.num_layers)  # Generate x values
        density_values = chi2.pdf(x_values, self.k)  # Evaluate density function at x values
        lambda_values = density_values / np.sum(density_values)  # Normalize density values
        return lambda_values

    def next_epoch(self):
        self.current_epoch += 1

# Beispielverwendung
num_epochs = 40
num_layers = 7
initial_k = 1  # Initialer Parameter k für Chi-Quadrat-Verteilung
final_k = 10

lambda_updater = LambdaUpdater(num_layers=num_layers, initial_k=initial_k, final_k=final_k, num_epochs=num_epochs)

for epoch in range(num_epochs):
    lambda_updater.next_epoch()
    lambda_updater.update_lambdas()

    lambda_values = lambda_updater.get_lambda_values()
    print(f"Epoch {epoch + 1}: Lambda Values - {lambda_values}")
