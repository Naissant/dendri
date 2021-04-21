# Contributor Guide

## Development Install

```bash
git clone https://github.com/GDITHealth/dendri
pip install -e .[dev]
```

## Updating PyPi

[Versioneer](https://github.com/python-versioneer/python-versioneer) is used to automatically update the project version based on tags.

A pre-configured Github Action is triggered when a tagged commit is pushed.

```bash
git commit -m "message"
git push
# Wait for tests to complete before tagging commit and pushing tag.
git tag 1.2.3
git push --tags
```

Useful: [https://stackoverflow.com/questions/3745135/push-git-commits-tags-simultaneously](https://stackoverflow.com/questions/3745135/push-git-commits-tags-simultaneously)
