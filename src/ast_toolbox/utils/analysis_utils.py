import os
import pickle

import matplotlib.animation
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np


def render_paths_heatmap_gif(filepath, gif_file=None, frames=10):
    ped_x_max = 3.0
    ped_y_max = 10.0
    ped_x_min = -10.0
    ped_y_min = -6.0
    LOGMIN = 0.01
    fidelity = 100

    fig = plt.figure()
    ax = plt.axes()

    fidelity = 100
    ped_position_discrete = np.zeros((fidelity, fidelity))

    # line, = ax.plot([], [], lw=2)
    filepath + '/paths_itr_0.p'
    bounds = [0.1, 1, 10**1, 10**2, 10**3, 10**4, 10**5]
    im = ax.imshow(ped_position_discrete,
                   # norm=colors.LogNorm(vmin=max(ped_position_discrete.min(), LOGMIN)),
                   norm=colors.BoundaryNorm(boundaries=bounds, ncolors=256),
                   extent=[ped_x_min, ped_x_max, ped_y_min, ped_y_max]
                   )
    ((-1.5 - ped_y_min)) / (ped_y_max - ped_y_min)
    ((4.5 - ped_y_min)) / (ped_y_max - ped_y_min)
    ax.axhline(y=-1.5, xmin=0, xmax=1)
    ax.axhline(y=4.5, xmin=0, xmax=1)
    vmin = max(ped_position_discrete.min(), LOGMIN)
    vmax = max(ped_position_discrete.max(), vmin + LOGMIN)
    im.set_clim(vmin, vmax)
    fig.colorbar(im, ax=ax)  # , extend='max')
    # initialization function: plot the background of each frame

    def get_count(itr):
        # pdb.set_trace()
        filename = filepath + '/paths_itr_' + str(itr) + '.p'

        if (os.path.exists(filename)):
            with open(filename, 'rb') as f:
                paths = pickle.load(f)

            for path in range(paths['env_infos']['state'].shape[0]):
                # car = paths['env_infos']['state'][path, :, 3:7]
                # peds = paths['env_infos']['state'][path, :, 9:13].reshape((1, 4))
                for stp in range(paths['env_infos']['state'].shape[1]):
                    pedx = paths['env_infos']['state'][path, stp, 11]
                    pedy = paths['env_infos']['state'][path, stp, 12]

                    if pedx < ped_x_max and pedx > ped_x_min and pedy < ped_y_max and pedy > ped_y_min:
                        ped_x_idx = int(((pedx - ped_x_min) * fidelity) // (ped_x_max - ped_x_min))
                        ped_y_idx = int(((pedy - ped_y_min) * fidelity) // (ped_y_max - ped_y_min))

                        ped_position_discrete[-ped_y_idx, ped_x_idx] += 1

    def init():
        get_count(0)
        vmin = max(ped_position_discrete.min(), LOGMIN)
        vmax = ped_position_discrete.max()
        im.set_data(ped_position_discrete)
        im.set_clim(vmin, vmax)
        # im.set_norm(colors.LogNorm(vmin=max(ped_position_discrete.min(), LOGMIN)))
        return [im]

    # animation function.  This is called sequentially
    def animate(i):
        get_count(i)
        vmin = max(ped_position_discrete.min(), LOGMIN)
        vmax = ped_position_discrete.max()
        # print(i, ped_position_discrete.max())
        im.set_data(ped_position_discrete)
        im.set_clim(vmin, vmax)
        # im.set_norm(colors.LogNorm(vmin=max(ped_position_discrete.min(), LOGMIN)))
        return [im]

    anim = matplotlib.animation.FuncAnimation(fig, animate, init_func=init, frames=frames, interval=200, blit=True)
    anim.save(filepath + '/' + gif_file, writer='imagemagick')


def render_paths(filepath, gif_file=None):
    fidelity = 100
    ped_position_discrete = np.zeros((fidelity, fidelity))

    itr = 0
    filename = filepath + '/paths_itr_' + str(itr) + '.p'

    ani = matplotlib.animation.FuncAnimation(fig, animate, init_func=init_animation, frames=50)
    ani.save('/tmp/animation.gif', writer='imagemagick', fps=30)

    while(os.path.exists(filename)):
        with open(filename, 'rb') as f:
            paths = pickle.load(f)

            fig, ax = plt.subplots()
            render_itr_heatmap(paths, ped_position_discrete, fig=fig, ax=ax)

        # pdb.set_trace()

        if gif_file is None:
            plt.show()

        itr += 1
        filename = filepath + '/paths_itr_' + str(itr) + '.p'


def render_itr_heatmap(samples_data, visit_counts, fig=None, ax=None):
    ped_x_max = 3.0
    ped_y_max = 10.0
    ped_x_min = -10.0
    ped_y_min = -6.0
    LOGMIN = 0.01
    fidelity = 100
    if ax is None or fig is None:
        fig, ax = plt.subplots()

    for path in range(samples_data['env_infos']['state'].shape[0]):
        # car = paths['env_infos']['state'][path, :, 3:7]
        # peds = paths['env_infos']['state'][path, :, 9:13].reshape((1, 4))
        for stp in range(samples_data['env_infos']['state'].shape[1]):
            pedx = samples_data['env_infos']['state'][path, stp, 11]
            pedy = samples_data['env_infos']['state'][path, stp, 12]
            if pedx < ped_x_max and pedx > ped_x_min and pedy < ped_y_max and pedy > ped_y_min:
                ped_x_idx = int(((pedx - ped_x_min) * fidelity) // (ped_x_max - ped_x_min))
                ped_y_idx = int(((pedy - ped_y_min) * fidelity) // (ped_y_max - ped_y_min))

                visit_counts[ped_x_idx, ped_y_idx] += 1

    fig, ax = plt.subplots()
    pcm = ax.imshow(visit_counts,
                    norm=colors.LogNorm(vmin=max(visit_counts.min(), LOGMIN), vmax=visit_counts.max()),
                    extent=[ped_x_min, ped_x_max, ped_y_min, ped_y_max]
                    )
    ((-1.5 - ped_y_min)) / (ped_y_max - ped_y_min)
    ((4.5 - ped_y_min)) / (ped_y_max - ped_y_min)
    ax.axhline(y=-1.5, xmin=0, xmax=1)
    ax.axhline(y=4.5, xmin=0, xmax=1)
    fig.colorbar(pcm, ax=ax, extend='max')
