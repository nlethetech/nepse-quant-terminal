"""The terminal's stylesheet must parse: a broken one stops the app from starting,
and no other unit test loads it."""

from pathlib import Path

from textual.css.stylesheet import Stylesheet

TCSS = Path(__file__).resolve().parents[2] / "apps" / "tui" / "dashboard_tui.tcss"


def test_dashboard_stylesheet_parses():
    sheet = Stylesheet()
    sheet.read(TCSS)
    sheet.parse()  # raises StylesheetParseError on any syntax error
    assert sheet.rules
