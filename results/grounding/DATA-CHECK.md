# Grounding v1 dataset preflight

**Full archives approved for private academic research use.** The initial preflight identified three image archives over 2 GB. Transfers are stored under `~/datasets/satquery-grounding-v1/`. The repository carries only frozen annotation pointers and hashes.

| Dataset | Published image archives | Annotation and split files | Licence and research demo |
|---|---|---|---|
| DIOR-RSVG | `JPEGImages.zip` 5,386,728,679 bytes (5.39 GB) | `Annotations.zip` 7,774,359 bytes. `train.txt` 154,012 bytes, `val.txt` 21,925 bytes, `test.txt` 42,873 bytes. | Authors specify research use only and list CC BY-NC 4.0. Suitable for an attributed noncommercial research demo under those terms. Commercial use requires formal permission. |
| VRSBench | `Images_train.zip` 8.36 GB and `Images_val.zip` 3.98 GB | `Annotations_train.zip` 28.6 MB, `Annotations_val.zip` 12.8 MB, `VRSBench_train.json` 64.9 MB, referring evaluation JSON 10.3 MB, caption evaluation JSON 4.81 MB. | Text annotations are CC BY 4.0 with attribution. DOTA-derived imagery is academic-use only, with the underlying Google Earth terms applying where relevant. The authors intend academic research and education. Confirm imagery terms for the intended SIH demo before displaying or redistributing samples. |

- DIOR-RSVG sources. [Authors’ release and research-use terms](https://github.com/ZhanYang-nwpu/RSVG-pytorch#license) · [CC BY-NC 4.0 legal text](https://creativecommons.org/licenses/by-nc/4.0/legalcode) · [Archived official-download byte counts](https://github.com/sohamsssssssssssssssss/sih2026/blob/3be90daa235ab83b7768a16128d2b1619bff84f1/eval/results/grounding/dior-rsvg-dataset-audit.json)
- VRSBench sources. [Authors’ download listing](https://huggingface.co/datasets/xiang709/VRSBench/tree/main) · [Authors’ licence scope](https://github.com/lx709/VRSBench#licensing-information) · [CC BY 4.0 legal text](https://creativecommons.org/licenses/by/4.0/legalcode) · [DOTA upstream usage terms](https://captain-whu.github.io/DOTA/dataset.html#image-source-and-usage-license) · [Academic intended uses](https://papers.nips.cc/paper_files/paper/2024/file/05b7f821234f66b78f99e7803fffa78a-Supplemental-Datasets_and_Benchmarks_Track.pdf)

## Splits, coordinates and resolution

| Dataset | Sizes and splits | Boxes | Documented ground sample distance |
|---|---|---|---|
| DIOR-RSVG | 17,402 RGB images, 38,320 expressions. Released train/validation/test lists have 26,991 / 3,829 / 7,500 expressions and 14,730 / 3,465 / 6,102 distinct images within each split. | Per-image XML contains axis-aligned pixel `xmin,ymin,xmax,ymax` boxes on 800 × 800 images. | Parent DIOR documents a dataset-wide 0.5–30 m range. No per-image GSD was verified for the RSVG release. |
| VRSBench | 29,614 images and captions, 52,472 references. Published train/test contain 20,264 / 9,350 images and captions and 36,313 / 16,159 references. The test image archive is named `Images_val.zip`. | 512 × 512 patches. Source metadata supports oriented corners and horizontal `xyxy`. The authors distinguish 0–1 coordinates in the raw preparation pipeline from 0–100 in the Hugging Face train/eval JSONs. | Mixed sources. DOTA provides source-image GSD where known. VRSBench’s preparation code maps it to qualitative resolution for DOTA and leaves the field empty for DIOR. No uniform VRSBench GSD or complete per-patch numeric GSD was verified. |

- DIOR-RSVG geometry and split sources. [Authors’ paper](https://arxiv.org/html/2210.12634) · [Released split counts](https://github.com/ZhanYang-nwpu/RSVG-pytorch#introduction) · [Parent DIOR paper, section 3](https://arxiv.org/pdf/1909.00133) · [XML reader](../../eval/suites/grounding_dior_rsvg.py)
- VRSBench geometry and split sources. [Paper, table 2](https://papers.nips.cc/paper_files/paper/2024/file/05b7f821234f66b78f99e7803fffa78a-Paper-Datasets_and_Benchmarks_Track.pdf) · [Coordinate note](https://github.com/lx709/VRSBench#-updates) · [Preparation code](https://github.com/lx709/VRSBench/blob/main/extract_patch_json.py) · [DOTA metadata](https://captain-whu.github.io/DOTA/dataset.html#meta-data)

The earlier DIOR-RSVG audit found 4,206 train/test, 2,449 train/validation, and 1,115 validation/test image IDs in common. The v1 manifest therefore needs image-level assignment checks. VRSBench includes DIOR crops, so the two datasets also need a shared parent-image identity check where provenance permits.

## Archive and source checksums

VRSBench uses publisher revision `6cee2968fd752a6d51c6cb2d18dded2bc0baa218`. The DIOR mirror at `eb2f30e` publishes the same image and annotation archive SHA-256 values as the earlier official-download audit. For the two small VRSBench evaluation JSONs, the publisher Git blob IDs verified the downloads and the table records the locally measured SHA-256. The source files stay private under the dataset directory. [VRSBench publisher metadata](https://huggingface.co/api/datasets/xiang709/VRSBench/tree/6cee2968fd752a6d51c6cb2d18dded2bc0baa218) · [DIOR mirror metadata](https://huggingface.co/datasets/LittleCollections/DIOR-RSVG/tree/eb2f30e)

| Dataset / source file | Bytes | SHA-256 | Local verification |
|---|---:|---|---|
| DIOR-RSVG `Annotations.zip` | 7,774,359 | `cecbf38add026dcfef9e5505c6594090cb11912c4801e6aa98514aa469024e4c` | Matches official-download audit |
| DIOR-RSVG `JPEGImages.zip` | 5,386,728,679 | `1c0a52b10c9471cb02c801590d87ab23035fb82fe776ed825682a0e3270e690e` | Publisher/mirror hash recorded, full local verification pending |
| DIOR-RSVG `train.txt` | 154,012 | `d55036d5dd9b234e74439cda910c8c0119894fbfcf72386412aef8be208c436b` | Matches official-download audit |
| DIOR-RSVG `val.txt` | 21,925 | `6324ad50498948e791e714cc03cb2ac7af4872e333087a48912f714141e979f9` | Matches official-download audit |
| DIOR-RSVG `test.txt` | 42,873 | `d1d0b200fc2be1c5c1ca776601278adb3566d0efc6ea7a690ba927ce1877e5f8` | Matches official-download audit |
| VRSBench `Images_train.zip` | 8,359,313,269 | `ece04d7e402621b924e5be9865fd0c725afb59cc96fbf0392409196a50c60481` | Publisher hash recorded, full local verification pending |
| VRSBench `Images_val.zip` | 3,976,656,690 | `67f99c1dd437fdab94fea1f54e049ad56d198c3c873237c79d9b944200d248b1` | Publisher hash recorded, full local verification pending |
| VRSBench `Annotations_train.zip` | 28,571,512 | `836ea7b2c4b19be27edea393331f3d4ab4d5abb0ce4715fc8b4bdb4c9bdc74b9` | Publisher hash recorded, full local verification pending |
| VRSBench `Annotations_val.zip` | 12,789,474 | `c63f7d5c2bef15572a14794bec7d5773ad45c8e8447df5c601a9e036bba04ffb` | Publisher hash recorded, full local verification pending |
| VRSBench `VRSBench_train.json` | 64,923,606 | `8a126b8342eeeb364e5673c166be09f92ebaae81a60dd43885d64dc084be7e73` | Matches publisher SHA-256 |
| VRSBench `VRSBench_EVAL_referring.json` | 10,277,680 | `fd63f7c6b77a158f4cc933a1ead88fa63aa23ebcf30f2ec9be111f3567ff1b44` | Matches publisher Git blob ID |
| VRSBench `VRSBench_EVAL_Cap.json` | 4,809,521 | `d45fea7288bd7b243c968add4f9d1d2fae473826a4dccb32df0663e209826be2` | Matches publisher Git blob ID |

The manifest was frozen from the verified annotation sources and the ZIP central directories read from the pinned remote archives. The ZIP indexes establish member names, sizes and CRCs. Full image archive hashes must match the manifest before evaluation.

## Frozen seed-17 samples

The six [manifest files](MANIFEST.json) identify rows and ZIP members by stable source references. The builder sorts by SHA-256 keys derived from seed 17 and first assigns image parents, including matching DIOR parents across both datasets and filename aliases with matching archive member size and CRC-32. Publisher test parents are excluded from selection and training. The two evaluation manifests use distinct source parents.

For VRSBench grounding, the sampler requires a single nonempty expression and one valid positive-area xyxy gold box in 0–100. The released training JSON had 4,298 refer rows without a single box in the published format, 6,451 with out-of-range or nonpositive-area boxes, and two with no tagged expression. The evaluation JSON had 13 unusable grounding rows. These rows were excluded before selecting any evaluation IDs. No evaluation ran.

The selected training data total 10,000 items on 7,500 immediate image files. VRSBench caption rows reuse its selected training images. The selected training members account for 2,591,173,640 compressed bytes across the image ZIPs, excluding archive metadata, and can be extracted by the listed IDs for a later approved run.
