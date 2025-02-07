import io

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

mpl.use('Agg')
from PIL import Image


def plt_fig_to_np(fig):
    buf = io.BytesIO()  # Create an in-memory binary stream
    fig.savefig(buf, format='png')  # Save the figure to the stream
    buf.seek(0)  # Reset the stream position to the beginning
    img = Image.open(buf)  # Read the image from the stream
    img = np.array(img)
    plt.close()
    return img


def plot_episode_data(episode_latent_means, episode_latent_logvars, episode_events, episode_num, process_num, ensemble_size=None):
    # Extract and reshape data for the given trial and process
    means = np.array([episode_latent_means[episode_num][i][process_num] for i in range(len(episode_latent_means[episode_num]))])
    logvars = np.array([episode_latent_logvars[episode_num][i][process_num] for i in range(len(episode_latent_logvars[episode_num]))])

    # Check if means and logvars are 2D arrays (time x latent_dim)
    if means.ndim != 2 or logvars.ndim != 2:
        raise ValueError("Data for means and logvars must be 2D (time x latent_dim).")

    # Assuming means and logvars are now lists of arrays, one per timestep
    latent_dim = means[0].shape[0]
    events = episode_events[episode_num][process_num]

    # Create a figure with two subplots
    fig, axs = plt.subplots(2, 1, figsize=(10, 8))

    # Time steps
    timesteps = np.arange(len(means))

    # Determine line styles for ensemble
    line_styles = ['-','--',':','-.'] if ensemble_size else ['-']
    ensemble_group_size = max(1, latent_dim // ensemble_size) if ensemble_size else latent_dim

    # Plot means and logvars, storing the line objects
    latent_lines = []
    for dim in range(latent_dim):
        line_style = line_styles[dim // ensemble_group_size % len(line_styles)]
        line, = axs[0].plot(timesteps, [mean[dim] for mean in means], label=f'LD {dim+1}', linestyle=line_style)
        axs[1].plot(timesteps, [logvar[dim] for logvar in logvars], linestyle=line_style)
        latent_lines.append(line)

    # Plotting vertical lines for events and creating legend entries
    event_lines = []  # To store the lines for the events
    for timestep, event in events.items():
        line = axs[0].axvline(x=timestep, linestyle='-', color='k', label=f'{event} at {timestep}')
        axs[1].axvline(x=timestep, linestyle='-', color='k')
        event_lines.append((line, event))

    axs[0].set_title(f'Trial {episode_num+1} Mean Values')
    axs[1].set_title(f'Trial {episode_num+1} Logvar Values')
    axs[0].set_xlabel('Timesteps')
    axs[1].set_xlabel('Timesteps')
    axs[0].set_ylabel('Mean Value')
    axs[1].set_ylabel('Logvar Value')

    # Creating separate legends
    axs[0].legend(handles=latent_lines, loc='upper right')
    axs[1].legend([line for line, label in event_lines], [label for line, label in event_lines], loc='upper right')
    
    # Adjust layout
    plt.tight_layout()

    # Save the figure
    # plt.savefig('test.png')

    # Convert the plot to an image array and return it
    fig.canvas.draw()
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    plt.close()

    return image

