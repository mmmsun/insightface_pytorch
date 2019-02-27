from __future__ import division

import torch

from ignite.metrics.metric import Metric
from ignite.exceptions import NotComputableError
from ignite.engine import Events
from ignite.metrics import CategoricalAccuracy as Accuracy


class AccuracyIter(Accuracy):
    """
    Calculates the accuracy.

    - `update` must receive output of the form `(y_pred, y)`.
    - `y_pred` must be in the following shape (batch_size, num_categories, ...) or (batch_size, ...)
    - `y` must be in the following shape (batch_size, ...)
    """
    def __init__(self, output_transform=lambda x: x):
        super(AccuracyIter, self).__init__()
        self._output_transform = output_transform
        self.reset()

    @torch.no_grad()
    def iteration_completed(self, engine, name):
        output = self._output_transform(engine.state.output)
        self.update(output)
        engine.state.metrics[name] = self.compute()

    def attach(self, engine, name):
        # if not engine.has_event_handler(self.started, Events.ITERATION_STARTED):
        engine.add_event_handler(Events.ITERATION_STARTED, self.started)
        # if not engine.has_event_handler(self.iteration_completed, Events.ITERATION_COMPLETED):
        engine.add_event_handler(Events.ITERATION_COMPLETED, self.iteration_completed, name)
        engine.add_event_handler(Events.EPOCH_COMPLETED, self.completed, name)