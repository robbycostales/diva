# Modified and documented from https://github.com/ShangtongZhang/DeepRL

class ConstantSchedule:
    def __init__(self, val):
        self.val = val

    def __call__(self):
        return self.val


class LinearSchedule:
    def __init__(self, start, end=None, steps=None, steps_per_iter=1):
        """ Linear interpolation between start and end over steps.

        Args:
            start (float): initial value.
            end (float): final value.
            steps (int or tuple): number of steps to interpolate over.
                If tuple, interpret as (start, end) step numbers.
            steps_per_iter (int): number of steps per call/iteration.
        """
        if end is None:
            # Constant schedule case if end is None
            end = start
            steps = 1

        if type(steps) is tuple:
            # If steps is a tuple, interpret it as (start, end) step numbers
            self.step_start = steps[0]
            n_steps = steps[1] - steps[0]
        elif type(steps) is int:
            # If steps is an int, interpret it as the end step number
            self.step_start = 0
            n_steps = steps
        else:
            raise ValueError('steps must be int or tuple')

        if float(n_steps) == 0:
            self.inc = 0
        else:
            self.inc = (end - start) / float(n_steps)

        self.current = start
        self.end = end
        self.curr_step = 0
        if end > start:
            self.bound = min
        else:
            self.bound = max
        self.steps_per_iter = steps_per_iter  # steps per iter

    def __call__(self, steps=None):
        """ Return the next value in the schedule. 
        
        Args:
            steps (int): number of steps to increment by. If None, use
                steps_per_iter.
        
        Returns:
            float: next value in the schedule.
        """
        if steps is None:
            steps = self.steps_per_iter
        val = self.current
        self.curr_step += steps
        if self.curr_step >= self.step_start:
            self.current = self.bound(self.current + self.inc * steps, self.end)
        return val