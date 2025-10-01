from rich.console import Console
from rich.theme import Theme

_theme = Theme({
    "info": "bold cyan",
    "warning": "bold yellow",
    "error": "bold red",
    "success": "bold green",
    "title": "bold magenta",
})

Console = Console(theme=_theme)