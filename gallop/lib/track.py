from typing import Type, TypeVar, Union, Sequence, Iterable, Optional, Callable, List, Dict, Any
import os
from numbers import Number
from collections import OrderedDict

from rich.text import Text
from rich.style import StyleType
from rich.table import Column
from rich.console import Console, JustifyMethod
from rich.highlighter import Highlighter
from rich.progress import Task, BarColumn, TextColumn, TaskProgressColumn, TimeRemainingColumn, Progress, ProgressColumn, MofNCompleteColumn, SpinnerColumn

from gallop.lib.boolean_flags import boolean_flags

NoneType = Type[None]
ProgressType = TypeVar("ProgressType")
KwargsType = Dict[str, Any]


class PostFixColumn(ProgressColumn):

    def __init__(
        self,
        style: StyleType = "none",
        justify: JustifyMethod = "left",
        markup: bool = True,
        highlighter: Optional[Highlighter] = None,
        table_column: Optional[Column] = None,
    ) -> None:
        self.justify: JustifyMethod = justify
        self.style = style
        self.markup = markup
        self.highlighter = highlighter
        super().__init__(table_column=table_column or Column(no_wrap=True))

    @staticmethod
    def format_num(n: Number) -> str:
        # from:
        # https://github.com/tqdm/tqdm/blob/master/tqdm/std.py
        """
        Intelligent scientific notation (.3g).
        Parameters
        ----------
        n  : int or float or Numeric
            A Number.
        Returns
        -------
        out  : str
            Formatted number.
        """
        f = '{0:.3g}'.format(n).replace('+0', '+').replace('-0', '-')
        n = str(n)
        return f if len(f) < len(n) else n

    def render(self, task: "Task") -> Text:
        # adapted from:
        # https://github.com/tqdm/tqdm/blob/master/tqdm/std.py
        _postfix = task.fields['postfix']

        if _postfix:
            postfix = OrderedDict([])
            for key in sorted(_postfix.keys()):
                postfix[key] = _postfix[key]
            # Preprocess stats according to datatype
            for key in postfix.keys():
                # Number: limit the length of the string
                if isinstance(postfix[key], Number):
                    postfix[key] = self.format_num(postfix[key])
                # Else for any other type, try to get the string conversion
                elif not isinstance(postfix[key], str):
                    postfix[key] = str(postfix[key])
                # Else if it's a string, don't need to preprocess anything
            # Stitch together to get the final postfix
            _text = "• " + ', '.join(key + '=' + postfix[key].strip() for key in postfix.keys()) + " ]"
        else:
            _text = "]"

        if self.markup:
            text = Text.from_markup(_text, style=self.style, justify=self.justify)
        else:
            text = Text(_text, style=self.style, justify=self.justify)
        if self.highlighter:
            self.highlighter.highlight(text)
        return text


class TimeElapsedColumn(ProgressColumn):
    """Renders time elapsed."""

    def __init__(self, compact: bool = False, **kwargs: KwargsType) -> None:
        self.compact = compact
        super().__init__(**kwargs)

    def render(self, task: "Task") -> Text:
        """Show time elapsed."""
        elapsed = task.finished_time if task.finished else task.elapsed
        if elapsed is None:
            return Text("-:--:--", style="progress.elapsed")

        minutes, seconds = divmod(int(elapsed), 60)
        hours, minutes = divmod(minutes, 60)

        if self.compact and not hours:
            formatted = f"{minutes:02d}:{seconds:02d}"
        else:
            formatted = f"{hours:02d}:{minutes:02d}:{seconds:02d}"

        return Text(formatted, style="progress.elapsed")


class SpeedColumn(TextColumn):

    def render(self, task: "Task") -> Text:
        speed = task.speed
        if speed is None:
            return Text("", style="progress.percentage")
        return Text(f"{speed:.1f} it/s", style="progress.percentage")


def get_progress(
    description: str = "Working...",
    auto_refresh: bool = True,
    console: Optional[Console] = None,
    transient: bool = False,
    get_time: Optional[Callable[[], float]] = None,
    refresh_per_second: float = 10,
    style: StyleType = "bar.back",
    complete_style: StyleType = "bar.complete",
    finished_style: StyleType = "bar.finished",
    pulse_style: StyleType = "bar.pulse",
    disable: bool = False,
    show_speed: bool = True,
) -> Iterable[ProgressType]:
    """Track progress by iterating over a sequence.

    Args:
        sequence (Iterable[ProgressType]): A sequence (must support "len") you wish to iterate over.
        description (str, optional): Description of task show next to progress bar. Defaults to "Working".
        total: (float, optional): Total number of steps. Default is len(sequence).
        auto_refresh (bool, optional): Automatic refresh, disable to force a refresh after each iteration. Default is True.
        transient: (bool, optional): Clear the progress on exit. Defaults to False.
        console (Console, optional): Console to write to. Default creates internal Console instance.
        refresh_per_second (float): Number of times per second to refresh the progress information. Defaults to 10.
        style (StyleType, optional): Style for the bar background. Defaults to "bar.back".
        complete_style (StyleType, optional): Style for the completed bar. Defaults to "bar.complete".
        finished_style (StyleType, optional): Style for a finished bar. Defaults to "bar.finished".
        pulse_style (StyleType, optional): Style for pulsing bars. Defaults to "bar.pulse".
        update_period (float, optional): Minimum time (in seconds) between calls to update(). Defaults to 0.1.
        disable (bool, optional): Disable display of progress.
        show_speed (bool, optional): Show speed if total isn't known. Defaults to True.
    Returns:
        Iterable[ProgressType]: An iterable of the values in the sequence.

    """
    disable = disable or boolean_flags(os.getenv("TQDM_DISABLE", False)) or boolean_flags(os.getenv("RICH_DISABLE", False)) or boolean_flags(os.getenv("DEBUG", False))

    columns: List["ProgressColumn"] = [SpinnerColumn()]

    columns.extend(
        [TextColumn("[progress.description]{task.description}:")] if description else []
    )
    columns.extend(
        (
            TaskProgressColumn(show_speed=show_speed),
            BarColumn(
                style=style,
                complete_style=complete_style,
                finished_style=finished_style,
                pulse_style=pulse_style,
            ),
            MofNCompleteColumn(),
            TextColumn("["),
            TimeElapsedColumn(compact=True),
            TextColumn("<"),
            TimeRemainingColumn(compact=True, elapsed_when_finished=False),
            TextColumn("•", justify="left"),
            SpeedColumn(text_format=""),
            PostFixColumn(justify="left"),  # TODO: this creates a space when no postfix
            # TextColumn("]"),  # This was moved to PostFixColumn; it should be removed from there
        )
    )
    progress = Progress(
        *columns,
        auto_refresh=auto_refresh,
        console=console,
        transient=transient,
        get_time=get_time,
        refresh_per_second=refresh_per_second or 10,
        disable=disable,
    )

    return progress


class track:

    def __init__(
        self,
        sequence: Union[Sequence[ProgressType], Iterable[ProgressType]],
        description: str = None,
        total: Optional[float] = None,
        update_period: float = 0.1,
        **kwargs: KwargsType,
    ) -> NoneType:
        self.sequence = sequence
        self.description = description
        self.total = total
        self.update_period = update_period

        kwargs['description'] = description
        self.progress = get_progress(**kwargs)

        self.progress.add_task(description, total=total, postfix='')

    def set_description(self, description: str) -> NoneType:
        self.progress.update(0, description=description)

    def set_postfix(self, postfix: Dict[str, float]) -> NoneType:
        self.progress.update(0, postfix=postfix)

    def __iter__(self) -> Iterable[ProgressType]:
        with self.progress:
            yield from self.progress.track(
                self.sequence, total=self.total, description=self.description, update_period=self.update_period,
                task_id=0)


if __name__ == '__main__':
    import time

    for x in track(range(100)):
        time.sleep(0.1)

    for x in track(range(100)):
        time.sleep(0.1)
