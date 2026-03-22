# Fonts Directory

Place `.ttf` or `.otf` font files in this directory to use custom fonts
with the typographic and visual injection attacks.

## Usage

Pass the font path when initialising an attack:

```python
from attacks import TypographicAttack

attack = TypographicAttack(font_path="utils/fonts/MyFont.ttf")
```

## Recommended Fonts

Any TrueType or OpenType font works. Some good options for adversarial
research:

- **Roboto** -- clean, widely available
- **Courier New** -- monospaced, high OCR readability
- **Arial** -- default fallback used by this toolkit

Fonts are not included in this repository due to licensing.
