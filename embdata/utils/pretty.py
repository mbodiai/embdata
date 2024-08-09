import logging
import traceback
from functools import partial
from sys import argv
from typing import Literal

import numpy as np
from rich.highlighter import ReprHighlighter
from rich.pretty import pretty_repr
from rich.text import Text

CYAN = "\033[36m"
YELLOW = "\033[33m"
MAGENTA = "\033[35m"
RESET = "\033[0m"
import re


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
        max_length = 1
        max_string = 30

    make_pretty = partial(pretty_repr, max_length=max_length, max_depth=max_depth, expand_all=expand_all, max_string=max_string)
    try:
        # if isinstance(obj, np.ndarray):
        #     pretty_str = make_pretty(np.round(obj, 2))
        if hasattr(obj, "__dict__"):
            pretty_str = make_pretty({
                            k: np.round(v, 2) if isinstance(v, np.ndarray | np.number) else "b'...'" if isinstance(v, bytes | bytearray) else v\
                                 for k, v in obj.__dict__.items() if\
                                v is not None or include in ["all", "null"]\
                                    and (include in ["private", "all"] or not k.startswith("_"))\
                        })
        else:
            pretty_str = make_pretty(obj)

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
        pretty_str = "".join(list(pretty_text.plain))

        # Apply color codes
        colored_str = (
            pretty_str.replace("'", f"{CYAN}'")  # Strings
            .replace('"', f'{CYAN}"')
            .replace("True", f"{MAGENTA}True{RESET}")  # Booleans
            .replace("False", f"{MAGENTA}False{RESET}")
            .replace("None", f"{MAGENTA}None{RESET}")
        )
        # Color numbers
        colored_str = re.sub(r"\b(\d+(\.\d+)?)\b", f"{CYAN}\\1{RESET}", colored_str)

        # Color keys
        return re.sub(r"(\w+)(?=\s*[=:])", f"{YELLOW}\\1{RESET}", colored_str)
    except Exception:
        logging.exception(f"Error in prettify: traceback: {traceback.format_exc()}")  # noqa: G004
        return obj.__repr__()
