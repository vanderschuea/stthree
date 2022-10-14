import torch
import torch.nn as nn


def check_inputs(inputs):
    """ Checks that inputs isn't composed of multiple tensors """
    iitype = type(inputs)
    if iitype == tuple:
        if len(inputs)>1:
            print(f"Input longer than 1 are not supported")
            raise NotImplementedError()
        inputs = inputs[0]
    elif iitype != torch.Tensor:
        raise NotImplementedError()
    return inputs


def count_conv2d(m:nn.Conv2d, inputs:torch.Tensor, outputs:torch.Tensor):
    """ Counts the #params and #ops in a conv2d layer """
    inputs = check_inputs(inputs)
    cin  = m.in_channels
    cout = m.out_channels
    x_stride, y_stride = m.stride
    x_kernel, y_kernel = m.kernel_size

    total_ops = cin*cout*x_kernel*y_kernel * (inputs.size(-2)/x_stride) * (inputs.size(-1)/y_stride) / m.groups
    m.total_ops = torch.Tensor([int(total_ops)])


def count_bn(m:nn.BatchNorm2d, inputs:torch.Tensor, outputs:torch.Tensor):
    """ Counts the #params and #ops in a bn2d layer """
    inputs = check_inputs(inputs)
    cin  = m.num_features
    total_ops = cin * inputs.size(-2) * inputs.size(-1) * 4
    m.total_ops = torch.Tensor([int(total_ops)])


def count_linear(m:nn.Linear, inputs:torch.Tensor, outputs:torch.Tensor):
    """ Counts the #params and #ops in a linear layer """
    cin  = m.in_features
    cout = m.out_features
    total_ops = cin*cout
    m.total_ops = torch.Tensor([int(total_ops)])


def add_hooks(m, handler_collection):
    if len(list(m.children())) > 0:
        return

    if hasattr(m, "total_ops") or hasattr(m, "total_params"):
        raise Warning("Either .total_ops or .total_params is already defined in %s.\n"
                        "Be careful, it might change your code's behavior." % str(m))

    m.register_buffer('total_ops', torch.zeros(1))
    m.register_buffer('total_params', torch.zeros(1))

    for p in m.parameters():
        m.total_params += torch.Tensor([p.numel()])

    fn = None
    if isinstance(m, nn.Conv2d):
        fn = count_conv2d
    elif isinstance(m, nn.BatchNorm2d):
        fn = count_bn
    elif isinstance(m, nn.Linear):
        fn = count_linear

    if fn is not None:
        handler = m.register_forward_hook(fn)
        handler_collection.append(handler)


@torch.no_grad()
def count(model, inputs):
    """ Counts the #params and #ops in a torch model

    Params
    ------
    cfg : dict
        The config used to determine the input size for computation
    """
    model.eval()
    handler_collection = []
    model.apply(lambda m: add_hooks(m, handler_collection=handler_collection))
    model(inputs)

    total_ops = 0
    total_params = 0
    mparams, gops = {}, {}
    for mname, m in model.named_modules():
        if not isinstance(m, nn.Conv2d) and not isinstance(m, nn.Linear):
            if hasattr(m, "total_ops"):
                del m.total_ops
                del m.total_params
            continue
        total_ops += m.total_ops
        total_params += m.total_params
        mparams[mname] = m.total_params.item()
        gops[mname] = m.total_ops.item()
        del m.total_ops
        del m.total_params

    total_ops = total_ops.item()
    total_params = total_params.item()

    # reset model to original status
    for handler in handler_collection:
        handler.remove()

    n_params = total_params
    n_ops = total_ops
    return n_params, n_ops, mparams, gops
