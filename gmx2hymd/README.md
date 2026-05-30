# gmx2diffmd
gmx2diffmd is a CLI utility that converts
[AMBER *itp* & *top*](http://www.cgmartini.nl/index.php/martini) coordinates and topologies
(obtained for example with [GROMACS](https://www.charmm-gui.org/)) to inputs
for [diff-aMD](https://github.com/Cascella-Group-UiO/HyMD) and [∂-MD](https://github.com/Cascella-Group-UiO/Diff-HyMD).

After cloning the repository, the program can be installed with pip.\
Optionally you can also install it in a virtual environment
```terminal
python -m venv <your_venv_dir> --upgrade-deps
source <your_venv_dir>/bin/activate
```
Then
```terminal
cd gmx2hymd
pip install .
```

You should now have access to these commands
```terminal
gmx2diffmd -f <input>.gro -p <topol>.top
diffmd-h5toxyz -f simulation.h5 -o simulation.xyz
diffmd-energy -f energy.log
```

Check the the other available flags with
```terminal
gmx2diffmd --help
diffmd-h5toxyz --help
diffmd-energy --help
```
In the `test_system` directory you can find example input files to use with `gmx2diffmd`.
