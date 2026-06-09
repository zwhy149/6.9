# Data locations and protocol

## Local data locations

The raw data were not copied into this GitHub package. Local source locations used during the experiments:

- 100Ah external-short-circuit and copied/difficult normal data: `D:\工作文件\大电池短路数据`
- 5Ah labeled external-short-circuit data: `D:\Battery_ESC\labeled_fault_data`
- Public short-circuit data: `D:\工作文件\公共短路数据`

## Why raw data are not included

This package contains code, final results, conclusions, and paper figures. Raw experimental spreadsheets are intentionally not uploaded here because the user request only asked for code/conclusions/images and because the data may be large or not intended for public redistribution.

## Split protocol

- 100Ah files are split by duplicate-aware groups.
- Copied/difficult normal files are kept within one partition and are not allowed to leak between train, validation, and test.
- Validation selects thresholds and ensemble weights.
- Test data are used only for final reporting.
- Thirty balanced admissible random seeds are averaged.
- Inputs are voltage-only.

## Reporting rule

The valid reviewer-facing 100Ah binary result is the strict 30-seed grouped result, not a single split and not the public fault-only locked check.
