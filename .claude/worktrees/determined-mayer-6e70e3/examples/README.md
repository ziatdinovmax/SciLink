To analyze the example data, run
```bash
scilink analyze --data data/YBCO_tem.tif --metadata data/YBCO_tem.json
```

Or launch a full session with

```bash
scilink analyze
```

And follow the instructions/hints to launch the analysis.

To generate an example AMBER force field, run
```bash 
scilink prepare-ff --test --solvate --goal "Test solvated system"
```
