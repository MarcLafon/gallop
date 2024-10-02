from typing import Callable
import argparse


def float_range(min_value: float, max_value: float) -> Callable[[str], float]:
    """Return function handle of an argument type function for
       ArgumentParser checking a float range: mini <= arg <= maxi
         mini - minimum acceptable argument
         maxi - maximum acceptable argument"""

    # Define the function with default arguments
    def float_range_checker(arg: str) -> float:
        """New Type function for argparse - a float within predefined range."""

        try:
            f = float(arg)
        except ValueError:
            raise argparse.ArgumentTypeError("must be a floating point number")
        if f < min_value or f > max_value:
            raise argparse.ArgumentTypeError("must be in range [" + str(min_value) + " .. " + str(max_value) + "]")
        return f

    # Return function handle to checking function
    return float_range_checker
