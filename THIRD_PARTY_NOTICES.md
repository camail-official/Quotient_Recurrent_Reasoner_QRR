# Third-party notices

## FPRM / TRM / HRM (MIT; HRM Apache-2.0)

The training harness, dataset loader, dataset builders (Sudoku-Extreme, Maze-Hard, ARC),
the ARC evaluator and the shared layer utilities (`pretrain.py`, `puzzle_dataset.py`,
`dataset/`, `evaluators/arc.py`, `models/layers.py`, `models/common.py`,
`models/sparse_embedding.py`, `models/losses.py`, `models/ema.py`) are derived from the FPRM
codebase, distributed under the following license:

```
MIT License

Copyright (c) 2025 Samsung Electronics Co., Ltd. All Rights Reserved.
Copyright (c) 2026 Sajad Movahedi, Vera Milovanovic, Shlomo Libo Feigin, Alexander Theus, Thomas Hofmann, Valentina Boeva, T. Konstantin Rusch, Antonio Orvieto

This project is derived from the Tiny Recursive Model (TRM) codebase
(https://github.com/SamsungSAILMontreal/TinyRecursiveModels, MIT License),
which is itself based on the Hierarchical Reasoning Model (HRM)
(https://github.com/sapientinc/HRM, Apache License 2.0).

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## EqR (Apache-2.0)

`dataset/build_maze_unique_dataset.py` is from EqR (https://github.com/locuslab/EqR,
Apache License 2.0), modified as noted at the top of the file.

## Mini-ARC

The Mini-ARC tasks are from https://github.com/KSB21ST/MINI-ARC; `dataset/miniarc_to_kaggle.py`
converts a clone of that repository.
