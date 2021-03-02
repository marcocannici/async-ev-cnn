import numpy as np

from src.layers.layer import Layer


class IntegrationLayer(Layer):
    """
    Integrates one event at a time into a leaky surface. It generates a new event whenever a value in the surface
    changes sign.
    """

    def __init__(self, leak, h_surface, w_surface):
        """
        :param leak: The leak to be applied by the layer
        :param h_surface: The height of the input canvas
        :param w_surface: The width of the input canvas
        """

        self._leak = leak
        self._h_surface = h_surface
        self._w_surface = w_surface

        self._prev_ts = 0
        self._out_shape = [1, h_surface, w_surface]
        self._init_surface = np.zeros([1, h_surface, w_surface], dtype=np.float32)
        self._surface = self._init_surface.copy()

        self._cached_actfn = None

    def surface(self):
        return self._surface

    def layer_actfn(self):

        if self._cached_actfn is None:
            self._cached_actfn = (self._surface > 0).astype(np.float32)
        return self._cached_actfn

    def conv_actfn(self):

        if self._cached_actfn is None:
            self._cached_actfn = (self._surface > 0).astype(np.float32)
        return self._cached_actfn

    def out_shape(self):
        return self._out_shape

    def reset(self):
        self._prev_ts = 0
        self._surface = self._init_surface.copy()
        self._cached_actfn = None

    def compute(self, events, _):

        # Retrieves the coordinates of the event
        y, x, ts = events.T
        last_event_ts = np.max(ts)

        # Computes the leak to be applied to each value in the surface
        delta_leak = (last_event_ts - self._prev_ts) * self._leak

        # Saves which values were positive before the update
        before_positives = self._surface > 0
        # Updates the surface by applying the leak
        self._surface = self._surface - delta_leak
        after_leak_negatives = self._surface <= 0
        # Sets to zero the coordinates with negative values
        self._surface[after_leak_negatives] = 0

        # Updates the coordinate of the event
        self._surface[:, y, x] += 1.0 - (last_event_ts - ts) * self._leak
        after_events_negatives = self._surface <= 0
        # Sets to zero the coordinates with negative values
        self._surface[after_events_negatives] = 0

        # Retrieves the coordinates of the pixels that changed from positive to negative during the update
        # Adds to the mask of changed coordinates the input event, it must be always forwarded
        new_event_mask = np.logical_and(before_positives,
                                        np.logical_or(after_leak_negatives, after_events_negatives))
        new_event_mask[:, y, x] = True

        # Discard the first coordinate (depth)
        new_events = np.where(new_event_mask)[1:]

        # Updates the prev_ts so that it can be used for the next run
        self._prev_ts = last_event_ts

        # Resets cached values
        self._cached_actfn = None

        return new_events, delta_leak

    def compute_all(self, events, delta_leak=None):

        return self.compute(events, None)