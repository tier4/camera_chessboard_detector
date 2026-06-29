# Third-Party Notices and Attribution

`camera_chessboard_detector` implements, and is a derivative work of, the
growth-based automatic checkerboard / chessboard corner detector of
**Andreas Geiger et al.** ("libcbdetect"). Both the original reference
implementation and the C++ reference port listed below are distributed under
the GNU General Public License, with which this project's `GPL-3.0-or-later`
license (see [LICENSE](LICENSE)) is compatible.

## Upstream sources

- **Original reference implementation (MATLAB), "libcbdetect"** — Andreas
  Geiger, Karlsruhe Institute of Technology (KIT).
  <https://www.cvlibs.net/software/libcbdetect/> — GNU General Public License.
- **C++ reference port, "ftdlyc/libcbdetect"** —
  <https://github.com/ftdlyc/libcbdetect> — GPL-3.0.

The upstream sources carry the following copyright and license notice, which
is reproduced here as required by the GPL:

> Copyright 2012. All rights reserved.
> Author: Andreas Geiger
> Institute of Measurement and Control Systems (MRT)
> Karlsruhe Institute of Technology (KIT), Germany
>
> This is free software; you can redistribute it and/or modify it under the
> terms of the GNU General Public License as published by the Free Software
> Foundation; either version 3 of the License, or any later version.
>
> This software is distributed in the hope that it will be useful, but WITHOUT
> ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
> FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

## How this project relates to the upstream

This project re-implements the libcbdetect detection pipeline (corner
likelihood, non-maximum suppression, sub-pixel refinement, scoring and
growth-based board structure recovery) and adds an optional CUDA pipeline. It
is offered under the same copyleft terms as the upstream work.

## License compatibility

The upstream code is licensed "GNU General Public License ... version 3 of the
License, or any later version" (i.e. `GPL-3.0-or-later`). This project is also
distributed under `GPL-3.0-or-later`, which preserves and is compatible with
the upstream license terms.

## Citation

If you use this software in academic work, please cite the original paper:

```bibtex
@inproceedings{Geiger2012ICRA,
  author    = {Andreas Geiger and Frank Moosmann and {\"O}mer Car and Bernhard Schuster},
  title     = {Automatic Camera and Range Sensor Calibration using a Single Shot},
  booktitle = {International Conference on Robotics and Automation (ICRA)},
  year      = {2012}
}
```
