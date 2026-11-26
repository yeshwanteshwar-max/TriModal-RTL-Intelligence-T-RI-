# ML Model and EDA Integration Audit

**Repository root:** `/workspaces/TriModal-RTL-Intelligence-T-RI-`  
**Audit date:** 2026-11-26

## Existing architecture

The project is a Python/PyTorch RTL analysis workflow organized around:

- `scripts/python/parsers/` for RTL, VCD, and Yosys JSON parsing.
- `scripts/python/simulation/` and `scripts/python/run_flow.py` for Icarus Verilog/VVP simulation and VCD generation.
- `scripts/python/synthesis/` for Pyosys synthesis and JSON/RTLIL/Verilog/Graphviz artifact generation.
- `backend/ML/` for modality feature extraction, encoders, fusion, inference, training, and experimental optimization code.
- `backend/API/` for FastAPI routes and request schemas, including design, analysis, simulation, multimodal, reporting, and optimization surfaces.
- `tests/` for parser, simulation, synthesis, and ML smoke tests.

The API and CLI paths already accept VCD-related input (`analysis_schema.vcd_path`, simulation routes, and `run_flow.py --vcd`). The current multimodal route reports whether modality strings are present; the new feature and encoder modules remain Python-callable and are not exposed as new prediction endpoints. No TypeScript/JavaScript frontend source appeared in the source inventory, and no frontend contract or existing API was changed. The existing multimodal model remains separate and unchanged.

## Reusable modules and current capabilities

| Area | Existing implementation | Audit finding |
|---|---|---|
| RTL parsing | `scripts/python/parsers/rtl_parser.py` | Lightweight module/port parsing, tokenization, and structural feature counts. |
| VCD parsing | `scripts/python/parsers/vcd_parser.py` | Existing custom parser already handled scopes, declarations, scalar and binary values, and timestamps, but did not retain aliases or signal type metadata, parse real values, or report capped input. |
| VCD analysis | `backend/ML/features/vcd_features.py`, `scripts/python/analysis/signal_activity.py` | The ML extractor only counted lines and scalar-looking transitions; the legacy activity helper counts text lines and is not a waveform feature pipeline. |
| VCD model | `backend/ML/encoders/vcd_encoder.py` | A GRU existed, but its fixed feature dimension/defaults and final-hidden-state pooling did not match a complete extracted feature sequence or masked pooling. |
| Netlist parsing | `scripts/python/parsers/netlist_parser.py` | Validates Yosys JSON and constructs a NetworkX graph. Cell-pin directions were not reflected in edges, and top-module selection was unavailable. |
| Netlist features/model | `backend/ML/features/netlist_features.py`, `backend/ML/encoders/netlist_encoder.py` | Static Verilog text cell counting existed. The encoder used simple sum aggregation rather than a GraphSAGE mean-neighbor update; there was no JSON-to-node-feature conversion. |
| Training/data | `backend/ML/training/`, `scripts/python/data_generation/` | Synthetic metadata and multimodal smoke training exist. The generated records have no VCD or netlist paths and do not provide labeled waveform/graph training examples. |
| Fusion/inference | `backend/ML/core/multimodal_model.py`, `backend/ML/inference/` | A multimodal fusion scaffold and prediction helpers already exist. These encoders are deliberately not connected to it in this change. |
| Simulation/synthesis | `scripts/python/simulation/`, `scripts/python/synthesis/`, `scripts/python/run_flow.py` | Existing Icarus/VVP and Pyosys artifact workflows can generate the VCD and Yosys JSON inputs used below. |

The repository includes `waveform/vcd/raw/counter.vcd`, `netlist/json/counter.json`, and `netlist/graphs/counter.dot`. The checked-in counter waveform and JSON were used for end-to-end feature/embedding smoke checks.

## Environment and dependencies

Checked in this environment:

- Python 3, PyTorch, NumPy, pytest, and Pyosys are installed.
- `iverilog` and `vvp` are available.
- The standalone `yosys` executable is not on `PATH`; the existing Pyosys Python binding is available.
- `pyvcd` is declared in `requirements.txt` but is not installed here. The project already had a custom parser, so the implementation extends it and does not add a parser dependency.
- PyTorch Geometric is not required: GraphSAGE aggregation uses native PyTorch tensor operations.
- Existing training data is synthetic and has no VCD/JSON modality paths; no trained model checkpoint or representative labeled waveform dataset was found in the repository.

## Implemented pipeline

### VCD waveform encoder

`parse_vcd` remains a dictionary keyed by identifier for compatibility. `VCDSignal` retains its existing fields and now also exposes type, initial value, aliases, and whether event storage hit its cap. Parsing remains line-streamed and stores at most the configured number of signals and value events.

`extract_vcd_features` computes per-signal width, event/transition/toggle and edge counts, toggle/value-change rates, time-weighted high/low/X/Z occupancy, numeric-value statistics, and activity ratio. Circuit-level output includes signal/event/timestamp counts, duration in raw VCD ticks, the declared timescale string, aggregate and bus activity, activity distribution, and explicitly heuristic clock/reset candidates.

`extract_vcd_sequence` creates a fixed-width, 13-channel sequence with normalized time, event activity, edge/toggle counts, bit-state ratios, numeric bus activity, and inferred clock/reset activity. Transitions are accumulated into time bins rather than discarded during downsampling. `sampling_period`, sequence length, `none`/`minmax`/`zscore` normalization, and a sliding-window helper are configurable.

`VCDEncoder` is an independent bidirectional GRU with input projection, multilayer dropout, masked mean/max temporal pooling, variable sequence lengths, optional embedding normalization, and a 128-dimensional default embedding. `VCDPredictionHeads` provides optional behavior/activity/anomaly/regression heads. Activity thresholds are configurable; statistical anomaly findings are candidates, not verified functional defects. No head is trained without suitable labels.

### Netlist graph encoder

`netlist_to_graph` retains the NetworkX graph interface while adding optional module selection and direction-aware port/pin edges. `extract_netlist_graph` converts Yosys JSON to 16-dimensional typed node features and an edge index, plus graph-level counts. `NetlistEncoder` performs GraphSAGE concatenation of each node with its mean-neighbor representation, then mean-pools node embeddings by graph. It handles batched graphs and empty graphs and returns a 128-dimensional default embedding.

Both encoders are independent callable modules, each returning `[batch, 128]` embeddings suitable for later fusion. The pre-existing multimodal model, API contracts, simulation flow, and synthesis flow are not rewired.

## Compatibility and limitations

- Existing `parse_vcd(path, max_signals, max_values)` and `VCDSignal` fields remain available. Empty files return an empty signal map; malformed non-empty files raise `ValueError`.
- Existing VCD feature keys `lines` and `transitions` remain, with `transitions` now counting parsed value-change events across supported value forms.
- The default VCD feature width changes from the old placeholder dimension to 13. Existing serialized GRU checkpoints, if any, are not architecture-compatible and must be retrained.
- Existing 16-feature netlist encoder input shape and 128-dimensional output are preserved, but its architecture changes from sum aggregation to GraphSAGE; old netlist encoder weights are not guaranteed compatible.
- Graph edges are made bidirectional inside GraphSAGE aggregation so messages can propagate across connectivity regardless of Yosys pin direction, while the parser retains direction metadata.
- Clock/reset names and reset polarity are heuristics, not design/constraint facts. Reported simulation duration uses raw VCD timestamp ticks; the declared timescale string is exposed but not converted to physical units.
- The existing synthetic dataset does not train these modality encoders. Embeddings can be computed, but classification, regression, and learned anomaly predictions are not validated until representative labeled datasets and model training are added.
- The existing text-based activity helper and legacy summary helper remain available; the new feature module is the richer model-facing analysis path.

## Verification

- Focused waveform/GraphSAGE tests and the existing ML tests pass.
- The complete repository test suite passes.
- The existing counter VCD successfully parses into a `(256, 13)` sequence and a 128-D GRU embedding.
- The existing counter Yosys JSON successfully converts to a 51-node/92-edge graph with 16-D node features and a 128-D GraphSAGE embedding.
