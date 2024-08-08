from typing import Literal
import logging
import numpy as np
from rich.highlighter import ReprHighlighter
from rich.pretty import pretty_repr
from rich.text import Text


def prettify(obj, include: Literal["all", "private", "null"] | None = None) -> str:
    try:
        pretty_str = pretty_repr({
                        k: np.round(v, 2) if isinstance(v, np.ndarray | np.number) else v for k, v in obj.__dict__.items() if\
                             v is not None or include in ["all", "null"]\
                                and (include in ["private", "all"] or k.startswith("_"))
                    })
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