# Wave-FDM-Teaching

Teaching wave propagation with the finite difference method.

Companion code for: "Teaching Wave Propagation with the Finite Difference Method: From the CFL Condition to Physical Phenomena" by X. Cao, Ma'anshan University (submitted to European Journal of Physics).

## Contents

- `wave_fdm.py` — Complete simulation code with 4 numerical experiments (standing waves, traveling wave packets, boundary reflections, numerical dispersion)
- `convergence_study.py` — Grid convergence analysis script (L2-norm error vs. grid resolution)
- `requirements.txt` — Python dependencies (numpy, matplotlib)

## Usage

```bash
pip install -r requirements.txt
python wave_fdm.py      # Generate all figures
python convergence_study.py  # Convergence analysis
```

## License

MIT License
