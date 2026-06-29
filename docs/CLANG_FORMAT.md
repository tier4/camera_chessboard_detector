# Code style — clang-format & clang-tidy

Formatting and identifier naming are enforced by the `.clang-format` and
`.clang-tidy` files at the repository root, wired up through
[pre-commit](https://pre-commit.com/) (`.pre-commit-config.yaml`).

For day-to-day use, see the **Contributing** section of the
[README](../README.md). This document records the conventions and the manual
commands behind the hooks.

## Toolchain

The configs target the **LLVM 18** tools. On Debian/Ubuntu:

```bash
sudo apt install clang-format-18 clang-tidy-18
```

`clang-format` itself is version-tolerant: the bracket style uses the legacy
`AlignAfterOpenBracket: BlockIndent` spelling, which is valid from clang-format
14 through 22+ (newer releases still accept it for backward compatibility).

## Formatting — `.clang-format`

Google-based style with Allman braces, a 100-column limit, and pointer /
reference bound to the variable name (`const cv::Mat &img`, `int *p`).

Format every source file in place:

```bash
find src include test \( -name '*.cpp' -o -name '*.hpp' -o -name '*.cu' -o -name '*.cuh' \) \
  -print0 | xargs -0 clang-format -i
```

Check without editing (fails on any diff — useful in CI):

```bash
find src include test \( -name '*.cpp' -o -name '*.hpp' -o -name '*.cu' -o -name '*.cuh' \) \
  -print0 | xargs -0 clang-format --dry-run --Werror
```

## Naming — `.clang-tidy`

`.clang-tidy` enables `readability-identifier-naming` with these conventions:

| Kind | Style | Example |
| --- | --- | --- |
| types / enum types / type aliases | `CamelCase` | `CornerArray`, `RefineType`, `RealT` |
| enum constants | `UPPER_CASE` (SCREAMING_SNAKE) | `CUDA_SEPARABLE`, `REFINE_ALL` |
| methods / functions | `camelBack` | `detectChessboards`, `toMat` |
| variables / parameters / local constants | `lower_case` | `kernel_size`, `rows_left` |
| namespaces | `lower_case` | `camera_chessboard_detector` |
| static / global constants, constexpr | `k` + `CamelCase` | `kPi`, `kMatDepth` |
| private / protected / static members | `lower_case_` (trailing `_`) | `radius_`, `lambda_` |

The trailing `_` is for **non-public** members only — keep public data members
suffix-free, or make internal state `private`.

clang-tidy needs a compile database:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
```

Report violations (C++ translation units + their headers):

```bash
run-clang-tidy-18 -p build \
  -clang-tidy-binary=clang-tidy-18 \
  -header-filter='camera_chessboard_detector/(src|include)/' '\.cpp$'
```

Apply fixes automatically (add `-fix`):

```bash
run-clang-tidy-18 -p build -fix \
  -clang-tidy-binary=clang-tidy-18 \
  -clang-apply-replacements-binary=clang-apply-replacements-18 \
  -header-filter='camera_chessboard_detector/(src|include)/' '\.cpp$'
```

## CUDA (`.cu`) caveat — read before running `-fix`

`.cu` files are compiled by `nvcc`, so clang-tidy cannot analyse them; the lint
deliberately restricts the translation units to `'\.cpp$'`. Identifiers local to
a `.cu` file are therefore **not** enforced and may keep any style.

The catch is *shared symbols* — names declared in a header and defined or used
inside a `.cu` (CUDA class methods, enums, kernel-wrapper functions). A `-fix`
run renames the header / `.cpp` side but **not** the `.cu` side, breaking the
build. After any automated rename:

1. Build CPU **and** CUDA (`cmake --build build`) and update the `.cu` references
   to the renamed symbols by hand until it compiles.
2. Run `ctest`, and ideally diff detector output before/after. Automated renames
   can silently *collide* — e.g. stripping the trailing `_` from a public member
   that shares a name with a parameter turns `member_ = value` into the
   self-assignment `member = value`, which compiles and may still pass coarse
   tests while changing results. Such state belongs in `private` (with the `_`).
