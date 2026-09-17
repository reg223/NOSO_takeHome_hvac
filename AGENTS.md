# Project contribution rules

## Commit messages

Every commit message must begin with a lowercase tag of 3–5 characters,
followed immediately by a colon and a space, then a concise description of the
work:

```text
<tag>: <description>
```

Use a tag that names the kind of work, for example:

- `misc:` documentation, requirements, configuration, or maintenance
- `data:` dataset parsing, validation, or audit work
- `feat:` new product or policy behavior
- `fix:` a bug correction
- `test:` tests and test infrastructure
- `train:` model training, fitting, or training artifacts
- `eval:` evaluation, scoring, or analysis

Tags must be lowercase and 3–5 characters long. Keep the description specific
and written in the imperative mood where practical.

When a new category of work is introduced, add its tag and meaning to this
list before using it in a commit. Commit categories are metadata, so they do
not correspond to `.gitignore` patterns; add a path to `.gitignore` only when
the category also introduces generated or local-only files.
