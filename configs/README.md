# Local configuration

The shipped defaults live inside the package (`src/newal/data/default.yaml`) so
that an installed copy always has them, not only a source checkout.

To override anything, write `local.yaml` in this directory. It is gitignored and
is merged over the defaults. A good starting point:

```bash
newal config > configs/local.yaml
```

Then delete everything you are not changing — every key is optional.
