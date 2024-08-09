from embdata.sample import Sample


class State(Sample):
    """A class for storing the state of an environment."""
    done: bool = False
    """Whether the episode is done as determined by the environment."""
    is_first: bool = False
    is_terminal: bool = False
    """Whether the state is terminal for any reason; error, or otherwise."""
