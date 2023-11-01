

class FunctionCtx:

    def save_for_forward(self, *tensors):
        self._saved_forward_tensors = tensors

    def save_for_backward(self, *tensors):
        self._saved_backward_tensors = tensors


class BackwardFunction(FunctionCtx):

    def apply(self, *args):
        backward = self._forward_cls.backward
        return backward(*args)


class FunctionMeta(type):

    def __init__(cls, name, bases, attrs) -> None:
        backward_fn = type(name, (BackwardFunction, ), {
                           "_forward_cls": cls})
        cls._backward_fn = backward_fn
        super().__init__(name, bases, attrs)


class SingletonFunction(FunctionCtx, metaclass=FunctionMeta):
    _instances = {}

    def __new__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__new__(cls)
        return cls._instances[cls]

    @staticmethod
    def forward(ctx, tensors):
        raise NotImplementedError

    @staticmethod
    def backward(ctx, grad):
        raise NotImplementedError


class Function(SingletonFunction):

    def apply(self, *args):
        super().apply(*args)
