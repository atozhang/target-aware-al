# Data and asset licenses

This file documents the sources and licenses of data and third-party assets included in the repository. The top-level MIT license applies to the repository software and does not override the terms of third-party materials.

| Files | Upstream | License |
|---|---|---|
| `data/BH/BH_HTE_scaled_3.csv` | [doylelab/rxnpredict](https://github.com/doylelab/rxnpredict) (Ahneman et al., *Science* 2018) | Upstream MIT license (reproduced below) |
| `data/SAD/SAD_Database.csv`, `data/SAD/benchmark_splits/*.hdf5` (7 files) | [SEDenmarkLab/SAD](https://github.com/SEDenmarkLab/SAD) (Ocampo et al., *ACS Cent. Sci.* 2025) | Upstream MIT license (reproduced below); authors request citation of the associated paper |
| `figures/fonts/texgyreheros-*.otf` (4 files) | GUST e-foundry (TeX Gyre Heros) | GUST Font License v1.0 (`figures/fonts/GUST-FONT-LICENSE.txt`; LPPL 1.3c or later) |
| `data/Wittig/wittig_literature_f119.csv` | Curated in this work from Vedejs and Peterson (1994) | See the note below |
| `data/Wittig/wittig_cas_f119.csv` | Curated in this work from reactions sourced from the CAS Content Collection™ (CAS is a division of the American Chemical Society) | Redistributed with the permission of CAS; see the note below |

## Wittig literature data

The 734 literature records in `data/Wittig/wittig_literature_f119.csv` were independently curated in this work from experimental results reported by Vedejs and Peterson (1994). The machine-readable reaction representation, standardization, and computed descriptors were created in this work. The source publication is cited in `data/README.md`. No license for the source publication is conveyed by this repository.

## CAS-derived Wittig reactions

The 233 records in `data/Wittig/wittig_cas_f119.csv` are reactions from publications and patents dated 2010–2015 sourced from the CAS Content Collection™, a comprehensive human-curated scientific data repository (CAS is a division of the American Chemical Society). They were curated in this work and are redistributed with the permission of CAS; see also [CAS Reactions](https://www.cas.org/cas-data/cas-reactions). The reaction structures of CAS-derived records in `results/wittig_qualitative_study/` are derived from the same CAS reaction records.

The curated reaction representation, the Z selectivities derived from the recorded product amounts, and the computed descriptors were created in this work. No other license for the CAS records is conveyed by this repository. See `data/Wittig/README.md`.

## Cited, not redistributed

The Rh dataset (Kalikadien et al., *Chem. Sci.* 2024) is available from [4TU.ResearchData](https://doi.org/10.4121/ecbd4b91-c434-4bdf-a0ed-4e9e0fb05e94.v1) under CC BY 4.0 but is not redistributed here. See `data/README.md` for details.

---

## MIT license — Doyle `rxnpredict`

```text
Copyright 2017 Derek T. Ahneman and Jesus G. Estrada and Shishi Lin and Spencer D. Dreher and Abigail G. Doyle

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
```

## MIT license — Denmark `SAD`

```text
MIT License

Copyright (c) 2025-2026 The Board of Trustees of the University of Illinois

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