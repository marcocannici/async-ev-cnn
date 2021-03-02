
class Layer:
    """
    A layer of the event-based ConvNet. An object from this class is stateful, meaning that it stores its internal
    state across different executions of the graph. You can build the network by connecting different layers using the
    constructor and then performing the actual computation by using the compute() method. The output feature map of each
    layer can be obtained with the featuremap() method, while surface() provides the surface of the layer before the
    application of the layer's activation function. The layer_actfn() provides the activation function applied by the
    current layer as a matrix that multiplied by the surface will produce the feature map. conv_actfn() instead is used
    by the layer to keep track of the transformation applied by the network up to the current layer.

    The graph can be defined as follows:

    # Definition of the stateful objects
    intgr = IntegrationLayer(leak, frame_height, frame_width)
    conv1 = Conv2DLayer(intgr, k1, np.array([b1]), 1, alpha)
    pool1 = MaxPoolLayer(conv1, [2, 2], 2)
    ...

    # Definition of the graph as a closure
    def graph(input):
        intgr_ev, delta_leak = intgr.compute(event, ts)
        conv1_ev = conv1.compute(intgr_ev, delta_leak)
        pool1_ev = pool1.compute(conv1_ev)
        ...

        return last_layer.featuremap()

    The network will maintain its stare across subsequent calls of graph().
    """

    def reset(self):
        """
        Resets the layer's state
        """
        raise NotImplementedError('Subclasses must override reset()')

    def compute(self, events, delta_leak):
        """
        Performs the layer's computation.

        :return: the events produced by the computation as a pair (events_y, events_x) and the leak
        """
        raise NotImplementedError('Subclasses must override compute()')

    def compute_all(self, events, delta_leak=None):
        """
        Performs the computations from the input layer to the current layer.
        :return: the events produced by the computation as a pair (events_y, events_x)
        """
        raise NotImplementedError('Subclasses must override compute_all()')

    def surface(self):
        """
        Returns the surface of the current layer, the feature map before the activation functions's application
        """
        raise NotImplementedError('Subclasses must override surface()')

    def layer_actfn(self):
        """
        The activation function applied by the current layer
        """
        raise NotImplementedError('Subclasses must override layer_actfn()')

    def conv_actfn(self):
        """
        The transformation applied by the network un to the current layer
        """
        raise NotImplementedError('Subclasses must override conv_actfn()')

    def out_shape(self):
        """
        The shape of the featuremap and surface
        """
        raise NotImplementedError('Subclasses must override out_shape()')

    def featuremap(self):
        """
        The featuremap of the current layer
        """
        return self.surface() * self.layer_actfn()