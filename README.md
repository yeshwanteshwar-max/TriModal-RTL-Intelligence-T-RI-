# TriModal RTL Intelligence

TriModal RTL Intelligence is an end-to-end experiment platform for understanding and improving digital hardware designs. It connects three complementary sources of evidence:

1. Verilog RTL source code
2. VCD simulation waveforms
3. Synthesized Yosys/Pyosys netlists

The project combines hardware parsing, EDA automation, graph analysis, temporal analysis, machine learning, and synthetic data generation in one reproducible workflow.

## 1. What problem does it solve?

Hardware teams often evaluate a design using separate tools and datasets: RTL engineers inspect source code, verification engineers analyze simulations, and synthesis teams measure the resulting circuit structure. These views describe the same design but are rarely combined automatically.

This project provides a practical foundation for answering questions such as:

- Which kind of hardware structure does this RTL implement?
- How active are its signals during simulation?
- How many cells are used, and how are they connected?
- What optimization strategy is appropriate for the observed structure?
- How can RTL, timing behavior, and circuit topology be analyzed together?
- Can a missing modality be handled without breaking the analysis pipeline?

The current implementation includes real parsing and synthesis, a self-checking counter testbench, deterministic synthetic RTL generation, and a multimodal PyTorch model with modality masking. ML predictions are experimental and should not be presented as verified PPA measurements until the model is trained on representative, validated data.

## 2. Project architecture

```text
RTL_AI_STUDIO/
├── backend/
│   ├── API/                 # FastAPI application and request schemas
│   └── ML/                  # Encoders, fusion model, training, and inference
├── constraints/             # Clock, timing, I/O, and synthesis configuration
├── design/
│   ├── filelist/            # Verilog compilation filelists
│   ├── rtl/                 # Synthesizable RTL
│   ├── tb/                  # Icarus Verilog testbenches
│   └── generated/           # Generated RTL candidates
├── netlist/
│   ├── raw/                 # RTLIL netlists
│   ├── json/                # Yosys JSON netlists
│   ├── verilog/             # Synthesized Verilog
│   ├── graphs/              # Graphviz DOT graphs
│   ├── img/                 # Numbered netlist PNG images
│   └── reports/             # Human-readable synthesis reports
├── scripts/python/
│   ├── parsers/             # RTL, VCD, and netlist parsers
│   ├── synthesis/           # Pyosys synthesis and export
│   ├── simulation/          # Icarus Verilog and VVP helpers
│   ├── analysis/            # RTL and circuit analysis
│   ├── data_generation/     # Deterministic synthetic RTL generation
│   └── run_flow.py          # Main VCD/VVP/netlist workflow
├── waveform/
│   ├── vcd/raw/             # Raw simulation waveforms
│   └── vvp/                 # Compiled Icarus Verilog executables
├── data/
│   ├── raw/                 # Original RTL, VCD, and netlist data
│   ├── processed/           # Parsed features and graphs
│   └── training/            # Synthetic samples, labels, and metadata
├── workspace/               # Design candidates and experiments
├── tests/                   # Parser, simulation, synthesis, and ML tests
└── requirements.txt
```

The repository retains the original project folders and script names where compatibility matters. New canonical folders follow a clear `backend`, `data`, `scripts`, `tests`, and `design` layout.

## 3. Why this approach?

RTL source, simulation behavior, and synthesized structure provide different kinds of evidence:

| Modality | What it observes | Example features |
|---|---|---|
| RTL | Intent and implementation | Modules, ports, operators, assignments, sequential blocks |
| VCD | Runtime behavior | Signal values, timestamps, transitions, unknowns, duty cycle |
| Netlist | Synthesized structure | Ports, wires, cells, connectivity, fan-in, fan-out |

A multimodal approach avoids relying on only one weak representation. It can operate even when one modality is unavailable, because embeddings are projected, normalized, masked, and fused separately. Incomplete data is reported as missing; the system does not invent it.

PyTorch is used for the model so that encoders, fusion, and prediction heads can evolve independently. The netlist encoder uses lightweight native PyTorch message passing, avoiding a hard dependency on PyTorch Geometric.

## 4. End-to-end workflow

```text
RTL source
   ↓
Icarus Verilog simulation
   ↓                         Pyosys/Yosys synthesis
VCD waveform ───────────→    Netlist JSON
   ↓                              ↓
VCD parser                  Graph builder
   ↓                              ↓
Temporal features           Native GNN embedding
                                  ↓
RTL tokenizer → RTL features → 128-D embedding ─┐
VCD encoder       → 128-D embedding ───────────┤
Netlist encoder   → 128-D embedding ───────────┤
                                                ↓
                                      Gated 256-D fusion
                                                ↓
                     ┌──────────────────────────┼────────────────┐
                     ↓                          ↓                ↓
              Design class                 PPA proxy       Optimization
```

1. `iverilog` compiles the RTL and testbench into a VVP executable.
2. `vvp` executes the testbench and writes a VCD waveform.
3. Pyosys reads and synthesizes the RTL.
4. Yosys exports RTLIL, JSON, and synthesized Verilog.
5. Python converts the JSON netlist into graph and report artifacts.
6. Graphviz renders a clear PNG with a generation number.
7. The RTL, VCD, and netlist parsers prepare model inputs.
8. The gated fusion model produces classification, regression, and recommendation outputs.

## 5. Technology stack

- Python 3
- Verilog RTL and SystemVerilog-compatible source inputs
- Icarus Verilog and VVP
- Pyosys 0.71.post40 and Yosys
- PyTorch
- NumPy
- NetworkX
- Graphviz
- FastAPI and Uvicorn
- Pytest
- Pandas, scikit-learn, Transformers, and Datasets

Graphviz and Icarus Verilog are system tools. Python packages are listed in `requirements.txt`.

## 6. How it works

### RTL encoder

The lightweight RTL parser recognizes modules, ports, parameters, operators, sequential and combinational constructs. The tokenizer preserves hardware keywords and common Verilog operators and emits structural feature counts plus tokens.

### VCD encoder

The bounded VCD parser maintains hierarchical scopes, signal aliases, widths, types, initial values, and timestamped scalar, vector, binary, real, `X`, and `Z` events. Feature extraction produces per-signal and circuit activity statistics plus a normalized 13-channel temporal sequence. The independent bidirectional GRU accepts padded variable-length batches and produces a normalized 128-dimensional waveform embedding. Optional behavior, activity, anomaly, and regression heads are separate from the encoder; they require labeled training data before predictions are meaningful.

Raw toggle counts are interpreted as switching activity, not as real power consumption.

### Netlist graph encoder

The Yosys JSON parser validates module and graph structure and creates direction-aware connections between ports, cells, and nets. Typed 16-dimensional node features feed a native PyTorch GraphSAGE mean-neighbor encoder, which returns a 128-dimensional graph embedding without requiring PyTorch Geometric. Both encoders can be called independently and are not yet wired into the existing multimodal model.

### Multimodal fusion

Each modality is projected from its native representation to a trainable 256-dimensional latent space. A gate learns how much of each available modality to use. Masked modalities receive no influence. The fused embedding feeds:

- A design-class classifier
- Separate PPA-proxy regression outputs
- A confidence-scored optimization recommender

Low-confidence recommendations are not asserted as verified optimization decisions.

## 7. Dataset and training

The synthetic generator creates parameterized half-adder, full-adder, multiplexer, decoder, counter, and comparator designs. Records include a design identity, RTL path, label, provenance, and dataset version.

Dataset splits are reproducible and grouped by original design family to reduce variant leakage.

```bash
python3 scripts/python/train_pipeline.py --epochs 2
```

The generated smoke checkpoint is written to `backend/ML/checkpoints/multimodal_smoke.pt`.

## 8. Example netlist outputs

One complete workflow run generates:

- `netlist/raw/counter.rtlil`
- `netlist/json/counter.json`
- `netlist/verilog/counter_netlist.v`
- `netlist/graphs/counter.dot`
- `netlist/img/counter_1.png`
- `netlist/reports/counter_report.txt`
- `waveform/vvp/counter.vvp`
- `waveform/vcd/raw/counter.vcd`

Every subsequent generation increments the PNG number.

## 9. Quick start

```bash
python3 -m pip install -r requirements.txt
python3 scripts/python/run_flow.py
python3 -m pytest -q
```

For complete setup and testing commands, see the private local command guide described by `.run_workflow.md`.

## 10. Current maturity

The repository is a working research and automation scaffold, not a substitute for industrial timing, power, formal equivalence, or sign-off tools. PPA models require realistic labeled datasets and calibrated validation. Optimization candidates must be simulation-tested and formally checked before deployment.
