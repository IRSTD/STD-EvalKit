import time

from .utils import _TYPES


def time_cost_deco(func):

    def wrapper(self, *args, **kwargs):
        if func.__name__ == 'update':
            start_time = time.time()
            res = func(self, *args, **kwargs)
            vars(self)['time_cost'].append(time.time() - start_time)
            if vars(self)['debug']:
                print(
                    f'{self.__class__.__name__}.update() took {time.time()-start_time:.2f}s.'
                )
        else:
            start_time = time.time()
            print(
                f'{self.__class__.__name__}.update() took '
                f'{sum(vars(self)["time_cost"])/len(vars(self)["time_cost"]):.2f}s each time.'
            )
            res = func(self, *args, **kwargs)
            if vars(self)['debug']:
                print(
                    f'{self.__class__.__name__}.update() and get() took a total of '
                    f'{sum(vars(self)["time_cost"]) + time.time()-start_time:.2f}s.'
                )
            if hasattr(self, '__repr__'):
                print(self.__repr__())
        return res

    return wrapper


class BaseMetric:

    def __init__(self, debug: bool = False, print_table: bool = True):
        """Base class for all metrics.

        Args:
            debug (bool, optional): Whether to print more detailed information, \
                such as European distances, will disable multi-threading.\
                Defaults to False.
            print_table (bool, optional): Whether to output the form on the command line. Defaults to True.
        """
        self.debug = debug
        self.print_table = print_table
        self.time_cost = []

    @time_cost_deco
    def update(self, preds: _TYPES, labels: _TYPES):
        """Support CHW, BCHW, HWC,BHWC, Image Path, or in their list form (except BHWC/BCHW),
            like [CHW, CHW, ...], [HWC, HWC, ...], [Image Path, Image Path, ...].

            Although support Image Path, but not recommend.
            Note :
                All preds are probabilities image from 0 to 1 in default.
                If images, Preds must be probability image from 0 to 1 in default.
                If path, Preds must be probabilities image from 0-1 in default, if 0-255,
                    we are force /255 to 0-1 to probability image.
        Args:
            labels (_TYPES): Ground Truth images or image paths in list or single.
            preds (_TYPES): Preds images or image paths in list or single.
        """
        raise NotImplementedError

    @time_cost_deco
    def get(self):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    @property
    def table(self):
        raise NotImplementedError

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}'
