import torch


class _BaseAggregator(object):
    """Base class of aggregators.
    Args:
        dist_communicator (object): A link object which can broadcast / gather, etc.
    """

    def __call__(self, inputs):
        """Aggregate the inputs and update in-place.
        Args:
            inputs (list): A list of tensors to be aggregated.
        Raises:
            NotImplementedError:
        """
        raise NotImplementedError


class _BaseAsyncAggregator(object):
    """AsyncAggregator base object"""


    def __call__(self, inputs):
        """Aggregate the inputs and update in-place.
        Args:
            inputs (list): A list of tensors to be aggregated.
        Raises:
            NotImplementedError:
        """
        raise NotImplementedError


class Mean(_BaseAggregator):
    def __call__(self, inputs):
        values = torch.stack(inputs, dim=0).mean(dim=0)
        return values



class AsyncMean(_BaseAsyncAggregator):
    def __call__(self, inputs):
        filtered = list(filter(lambda x: x is not None, inputs))
        values = torch.stack(filtered, dim=0).sum(dim=0) / len(inputs)
        return values



class DecentralizedAggregator(_BaseAggregator):
    """
    This Aggregators is applied to all nodes. It has access to the node information and a row of mixing matrix.
    """

    def __init__(self, node, weights):
        super().__init__()
        assert len(weights.shape) == 1
        self.node = node
        self.weights = weights

    def __call__(self, inputs):
        """
        The `inputs` is a list of tensors. The first element is the weight of itself, the second to the last elements are the
        gradient of its neighbors.
        """
        assert len(inputs) == 1 + len(self.node.edges)
        s = self.weights[self.node.index] * inputs[0]
        for e, inp in zip(self.node.edges, inputs[1:]):
            theothernode = e.theother(self.node)
            s += self.weights[theothernode.index] * inp
        return s
