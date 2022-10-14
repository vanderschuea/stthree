import qtoml as toml
from pathlib import Path
import sys
import argparse
import inspect
import torch

def get_class(obj_params, packages, *args, list_to_tensor=False, **optional):
    """ Extract an object corresponding to @obj_params["type"] in one @packages
        and initializes it with @obj_params (minus 'type')
        An @optional argument can be passed and will be added to @obj_params
        if it fits the the function's signature

    """

    # Find the correct package name and extract class or function
    obj_params = obj_params.copy()
    fx_type = obj_params["type"]
    del obj_params["type"]

    if type(packages) != list:
        packages = [packages]

    for package in packages:
        try:
            my_obj = getattr(package, fx_type)
        except:
            continue

        # Match arguments and check their validity
        my_obj_args = inspect.signature(my_obj).parameters
        for key, arg in optional.items():
            if key in my_obj_args or "kwargs" in my_obj_args:
                obj_params[key] = arg
            else:
                print(f"Couldn't find corresponding key for {key}: {arg} pair")

        if list_to_tensor:
            # Make all lists into torch tensors
            for key, item in obj_params.items():
                if type(item) == str or type(item) == dict:
                    continue
                if type(item) != list:
                    item = [item]
                obj_params[key] = torch.Tensor(item)

        return my_obj(*args, **obj_params)
    raise ValueError(f"Couldn't find {fx_type} inside one of {packages}")


def over_write_cfg(cfg, overwrites, namespace_trail=""):
    changed = set()
    if type(cfg) == list:
        for idx, item in enumerate(cfg):
            changed = changed.union(
                over_write_cfg(item, overwrites, namespace_trail=namespace_trail+"."+str(idx))
            )
    elif type(cfg) == dict:
        for key_part in cfg:
            key = namespace_trail+"."+key_part
            if key in overwrites:
                new_val = overwrites[key]
                orig_type = type(cfg[key_part])
                if orig_type == int:
                    new_val = int(new_val)
                elif orig_type == float:
                    new_val = float(new_val)
                elif orig_type == bool:
                    new_val = {
                        "False": False, "false": False,
                        "True": True, "true": True,
                    }.get(new_val, None)
                    if new_val is None:
                        raise ValueError(f"{overwrites[key]} is not of type bool")
                elif orig_type != str:
                    raise argparse.ArgumentTypeError(
                        f"Option {key[1:]} is trying to overwrite\n {cfg[key_part]},\n which is not a primitive type"
                    )
                cfg[key_part] = new_val
                changed.add(key)
            else:
                changed = changed.union(
                    over_write_cfg(cfg[key_part], overwrites, namespace_trail=key)
                )
    return changed


def get_config(infile=None):
    """ Load the config file and parses the input parameters

    Parameters:
    -----------
    infile : str
        If None will parse the input arguments sent via the command line. Every option will
        overwrite a value present in the config. Types will be matched whenever possible
        If set, will simply load the file without overwrites

    Returns:
    --------
    cfg : dict
        The config in the form of a dictionnary
    """
    if infile is not None:
        infile = Path(infile)
        with open(infile, 'r') as fp:
            return toml.loads(fp.read())
    else:
        assert len(sys.argv) > 1, "Expected: ./launch <config_file> [--overwrite.args]"

        parser = argparse.ArgumentParser()
        parser.add_argument('config_file', help="Path to config file containing the training parameters")
        args, unknown = parser.parse_known_args()

        assert len(unknown) % 2 == 0, "Every option should have a value assigned"
        overwrites = {}
        for i in range(len(unknown) // 2):
            opt = unknown[2*i]
            assert opt[:2] == "--", "Only full name optional parameters are supported"
            opt = "." + opt[2:]
            val = unknown[2*i+1]
            overwrites[opt] = val

        with open(args.config_file, 'r') as fp:
            cfg = toml.loads(fp.read())

        changed = over_write_cfg(cfg, overwrites)
        for opt in overwrites:
            if opt not in changed:
                raise parser.error(f"Did not find a match for {opt[1:]} in config file")
        return cfg

