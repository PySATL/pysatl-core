import re

from pysatl_core import __version__


def test_version_pep440() -> None:
    assert re.match(
        r"^\d+!\d+(\.\d+)*([abc]|rc)?\d*(\.post\d+)?(\.dev\d+)?$|^\d+(\.\d+)*([abc]|rc)?\d*(\.post\d+)?(\.dev\d+)?$",
        __version__,
    )
