"""Pick-from-a-list prompts: arrow keys in a real terminal, typed answers otherwise."""

from __future__ import annotations

import sys

import click
import typer


def can_use_menu() -> bool:
    return sys.stdin.isatty() and sys.stdout.isatty()


def choose(title: str, options: list[tuple[str, str]], default: str | None = None) -> str:
    """Return the value of the chosen option.

    `options` is a list of (value, label). With a terminal attached the user moves with the
    arrow keys and presses Enter; when input is piped (scripts, tests) they type the option's
    number or its value, so nothing here needs a terminal.
    """
    values = [value for value, _ in options]
    if default not in values:
        default = values[0]
    if can_use_menu():
        try:
            import questionary
        except ImportError:  # fall through to the typed prompt
            questionary = None
        if questionary is not None:
            picked = questionary.select(
                title,
                choices=[questionary.Choice(label, value=value) for value, label in options],
                default=default,
                instruction="(arrow keys, Enter to select)",
            ).ask()
            if picked is None:  # Ctrl+C
                raise typer.Abort()
            return picked

    click.echo(title)
    for number, (_value, label) in enumerate(options, 1):
        click.echo(f"  {number}) {label}")
    accepted = [*values, *(str(n) for n in range(1, len(values) + 1))]
    answer = typer.prompt("Choice", type=click.Choice(accepted), default=default, show_choices=False)
    return values[int(answer) - 1] if answer.isdigit() else answer
