import gym_minigrid.window as window
import matplotlib.pyplot as plt
# import gym_minigrid.rendering as rendering
# import gym_minigrid.minigrid as minigrid


class Window(window.Window):
    """
    Customized minigrid Window object.
    
    A number of features are based on HAL implementation, but unlike HAL, we
    are inheriting functionality from original class.
    """

    def __init__(self, title, args=None, env=None):
        self.no_image_shown = True  # In updated version of gym

        self.fig = None

        self.imshow_obj = None

        # Create the figure and axes
        self.fig, self.ax = plt.subplots()

        # Show the env name in the window title
        # NOTE: added 'manager':
        # https://github.com/danielhrisca/asammdf/issues/829
        # https://github.com/Udayraj123/OMRChecker/pull/56
        self.fig.canvas.manager.set_window_title(title)

        # Turn off x/y axis numbering/ticks
        self.ax.set_xticks([], [])
        self.ax.set_yticks([], [])

        # Flag indicating the window was closed
        self.closed = False

        def close_handler(evt):
            self.closed = True

        self.fig.canvas.mpl_connect('close_event', close_handler)

        # TEMPORARY
        self.record = False
        self.args = args
        self.env = env

    def redraw(self, img):
        ''' based on eponymous function in manual_control.py '''
        if not self.args.agent_view:
            img = self.env.render('rgb_array', tile_size=self.args.tile_size,
                                  highlight=False)

        self.show_img(img)

    def reset(self, done=False):
        ''' based on eponymous function in manual_control.py
            done is passed as True for the purpose of only saving successful
            trajectories upon reset
        '''
        if self.args.increment_seed:
            self.args.seed += 1

        if self.args.seed != -1:
            self.env.seed(self.args.seed)

        # if len(self.trajectory) > 0 and done and self.record:
        #     self.save_trajectory()
        # self.trajectory = []

        obs = self.env.reset()
        self.prev_obs = obs

        if hasattr(self.env, 'mission'):
            print('Mission: %s' % self.env.mission)
            self.set_caption(self.env.mission)
        # self.update_buttons()
        print('Resetting environment!')
        self.redraw(obs)


    def step(self, action):
        ''' based on eponymous function in manual_control.py '''
        # Perform step

        prev_obs = self.prev_obs
        obs, reward, done, info = self.env.step(action)
        print('step=%s, reward=%.2f' % (self.env.step_count, reward))

        # Store trajectory
        # if self.trajectory is not None:
        #     # store newly generated portion of trajectory
        #     self.trajectory.append({
        #         'reward': reward,
        #         'done': done,
        #         'obs': prev_obs,
        #         'action': action,
        #         'next_obs': obs,
        #         'demo': True,
        #     })
        self.prev_obs = obs

        # Update matplotlib button inventory figures
        # self.update_buttons()

        # Finish or redraw
        if done:
            print('done!')
            self.reset(done=True)
        else:
            self.redraw(obs)
            # print("inventory:", obs['inventory_readable'])