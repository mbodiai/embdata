from functools import partial
import traceback
from typing import Literal
import logging
import numpy as np
from rich.highlighter import ReprHighlighter
from rich.pretty import pretty_repr
from rich.text import Text
from sys import argv

def prettify(obj, include: Literal["all", "private", "null"] | None = None) -> str:
    if "expand_all" in argv:
        expand_all = True
        max_depth = None
        max_length = None
        max_string = None
    elif "expand" in argv:
        expand_all = False
        max_depth = None
        max_length = None
        max_string = 100
    else:
        expand_all = False
        max_depth = 10
        max_length = 10
        max_string = 30
    
    make_prety = partial(pretty_repr, max_length=max_length, max_depth=max_depth, expand_all=expand_all, max_string=max_string)
    try:
        if isinstance(obj, np.ndarray):
            pretty_str = make_prety(np.round(obj, 2))
        elif hasattr(obj, "__dict__"):
            pretty_str = make_prety({
                            k: np.round(v, 2) if isinstance(v, np.ndarray | np.number) else "b'...'" if isinstance(v, bytes | bytearray) else v\
                                 for k, v in obj.__dict__.items() if\
                                v is not None or include in ["all", "null"]\
                                    and (include in ["private", "all"] or not k.startswith("_"))\
                             

                        })
        else:
            pretty_str = make_prety(obj)

        pretty_text = Text.from_ansi(
                    pretty_str,
                    justify="left",
                    overflow="crop",
                    no_wrap=False,
                    style="pretty",
                )
        pretty_text = (
            ReprHighlighter()(pretty_text)
            if pretty_text
            else Text(
                f"{type(obj)}.__repr__ returned empty string",
                style="dim italic",
            )
        )
        pretty_text = pretty_text.with_indent_guides(
            style="repr.indent",
        )
        return "".join(list(pretty_text.plain))
    except Exception as e:
        logging.error(f"Error in prettify: traceback: {traceback.format_exc()}")
        return obj.__repr__()