import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

class EigenvaluePlotter:
    def __init__(self):
        # Load data
        self.alphas = np.load('lst_data/alphas.npy')
        self.modes = np.load('lst_data/modes.npy')
        self.y = np.load('lst_data/y.npy')

        # Create figure and subplots
        self.fig, (self.ax_spectrum, self.ax_eigenfunction) = plt.subplots(1, 2, figsize=(12, 5))
        self.fig.canvas.manager.set_window_title("Eigenvalue and Eigenfunction Viewer")

        # Plot initial spectrum
        self.plot_spectrum()

        # Initialize eigenfunction plot
        self.eigenfunction_lines = None

        # Connect the click event
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)

        # Add a reset button
        self.ax_button = plt.axes([0.45, 0.01, 0.1, 0.05])
        self.button = Button(self.ax_button, 'Reset View')
        self.button.on_clicked(self.reset_view)

    def plot_spectrum(self):
        self.ax_spectrum.clear()
        self.scatter = self.ax_spectrum.scatter(np.real(self.alphas), np.imag(self.alphas), 
                                                c=np.real(self.alphas), cmap='viridis')
        self.ax_spectrum.set_xlabel(r'$\Re(\alpha)$')
        self.ax_spectrum.set_ylabel(r'$\Im(\alpha)$')
        self.ax_spectrum.set_title('Eigenvalue Spectrum')
        self.fig.colorbar(self.scatter, ax=self.ax_spectrum, label=r'$\Re(\alpha)$')

    def plot_eigenfunction(self, index):
        self.ax_eigenfunction.clear()
        mode = self.modes[:, index]
        self.ax_eigenfunction.plot(np.real(mode), self.y, label='Real part')
        self.ax_eigenfunction.plot(np.imag(mode), self.y, label='Imaginary part')
        self.ax_eigenfunction.plot(np.abs(mode), self.y, label='Magnitude', linestyle='--')
        self.ax_eigenfunction.set_xlabel(r"$u'$")
        self.ax_eigenfunction.set_ylabel(r'$\eta$')
        self.ax_eigenfunction.set_title(f'Eigenfunction for α = {self.alphas[index]:.4f}')
        self.ax_eigenfunction.legend()
        self.fig.canvas.draw_idle()

    def on_click(self, event):
        if event.inaxes == self.ax_spectrum:
            click_x, click_y = event.xdata, event.ydata
            distances = np.abs(self.alphas - (click_x + 1j*click_y))
            index = np.argmin(distances)
            self.plot_eigenfunction(index)

            # Highlight the selected eigenvalue
            if hasattr(self, 'highlight'):
                self.highlight.remove()
            self.highlight = self.ax_spectrum.plot(np.real(self.alphas[index]), 
                                                   np.imag(self.alphas[index]), 
                                                   'ro', markersize=10, markerfacecolor='none')[0]
            self.fig.canvas.draw_idle()

    def reset_view(self, event):
        self.plot_spectrum()
        self.ax_eigenfunction.clear()
        self.ax_eigenfunction.set_title('Click on an eigenvalue to view its eigenfunction')
        self.fig.canvas.draw_idle()

    def show(self):
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    plotter = EigenvaluePlotter()
    plotter.show()
