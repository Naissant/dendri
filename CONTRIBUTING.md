# Contributor Guide

## Updating PyPi

[Versioneer](https://github.com/python-versioneer/python-versioneer) is used to automatically update the project version based on tags.

A pre-configured Github Action is triggered when a tagged commit is pushed.

```bash
git commit -m "message"
git tag v1.2.3
git push --tags
```
