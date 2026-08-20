# WSDM Data Boundary

The WSDM root repository tracks data documentation only. Raw and processed data remain local and are intentionally excluded from the root Git history.

Tracked metadata consists of the release cards under `data/raw/bcp_search_agent_trajectory/` and `data/raw/LRAT-Train/`. The independent `search_agent` repository is now mounted at `../search_agent/` and is excluded as a separate Git working copy.

Local-only payloads include the six BrowseComp-Plus trajectory JSONL files, LRAT training pairs and trajectory archives, processed split JSONL files, and any downloaded benchmark or corpus assets.

Public source for the diagnostic trajectories: https://huggingface.co/datasets/liuqi6777/bcp_search_agent_trajectory

Public source for the LRAT training data: https://huggingface.co/datasets/Yuqi-Zhou/LRAT-Train

The exact local split and source identities are recorded in `data/processed/early_stop_v1/manifest.json`; that manifest remains local-only with the data payload.
