# PeSTo container

This image runs PeSTo i_v4_1 residue-level ligand-interface inference. Model
parameters remain in the configurable mn-ligand reference directory at:

```text
pesto/i_v4_1/model_ckpt.pt
```

Build the image with `docker compose build pesto`. The normalized native output
contains residue probabilities in CSV form and a PDB with those probabilities
stored in the temperature-factor field.
