"""Freeze and acquire a model-blind, recent experimental PDB candidate pool."""

import concurrent.futures
import json
import urllib.parse
import urllib.request

import yaml

from pdbthink.acquisition.cache import StructureCache
from pdbthink.util import REPO_ROOT, stable_hash, write_json


def query_candidates(root, config):
    def terminal(attribute, operator, value):
        return {
            "type": "terminal",
            "service": "text",
            "parameters": {"attribute": attribute, "operator": operator, "value": value},
        }

    selection = config["selection"]
    query = {
        "query": {
            "type": "group",
            "logical_operator": "and",
            "nodes": [
                terminal("entity_poly.rcsb_entity_polymer_type", "exact_match", "Protein"),
                terminal(
                    "entity_poly.rcsb_sample_sequence_length",
                    "range",
                    {
                        "from": min(s["min"] for s in config["length_strata"]),
                        "to": max(s["max"] for s in config["length_strata"]),
                    },
                ),
                terminal("rcsb_entry_info.deposited_polymer_entity_instance_count", "equals", 1),
                terminal("rcsb_entry_info.resolution_combined", "less_or_equal", selection["max_resolution"]),
                terminal("exptl.method", "exact_match", selection["method"]),
                terminal(
                    "rcsb_accession_info.initial_release_date",
                    "range",
                    {
                        "from": selection["release_from"],
                        "to": selection["release_to"],
                    },
                ),
            ],
        },
        "return_type": "polymer_entity",
        "request_options": {
            "paginate": {"start": 0, "rows": 10000},
            "group_by": {"aggregation_method": "sequence_identity", "similarity_cutoff": 30},
            "group_by_return_type": "representatives",
        },
    }
    root.mkdir(parents=True, exist_ok=True)
    query_path = root / "query.json"
    if query_path.exists() and json.loads(query_path.read_text()) != query:
        raise ValueError("RCSB query changed; create a new source snapshot")
    write_json(query_path, query)
    if not (root / "response.json").exists():
        url = "https://search.rcsb.org/rcsbsearch/v2/query?json=" + urllib.parse.quote(json.dumps(query))
        with urllib.request.urlopen(url, timeout=120) as response:
            (root / "response.json").write_bytes(response.read())
    return json.loads((root / "response.json").read_text())


def main():
    root = REPO_ROOT / "data/prediction_sources/pdb_v1"
    config = yaml.safe_load((REPO_ROOT / "configs/prediction_pdb_v1.yaml").read_text())
    response = query_candidates(root, config)
    excluded = set(json.loads((REPO_ROOT / "data/provenance/coordinate_exclusions.json").read_text()))
    entities = [
        r["identifier"] for r in response["result_set"] if r["identifier"].split("_")[0] not in excluded
    ]
    entities = sorted(entities, key=lambda x: stable_hash(config["seed"], x))[
        : config["selection"]["candidate_limit"]
    ]
    cache = StructureCache()
    failures = []
    hashes = {}

    def acquire(entity):
        try:
            record = cache.get_pdb(entity.split("_")[0])
            return entity, record, None
        except Exception as exc:
            return entity, None, str(exc)

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        for n, (entity, record, error) in enumerate(pool.map(acquire, entities), 1):
            if record:
                hashes[record.entry] = record.sha256
            else:
                failures.append({"entity": entity, "reason": error})
            if n % 50 == 0:
                print("Acquired", n, "/", len(entities), flush=True)
    write_json(
        root / "sources.json",
        {
            "entities": [e for e in entities if e.split("_")[0] in hashes],
            "source_hashes": hashes,
            "excluded_existing_entries": len(excluded),
            "failures": failures,
        },
    )


if __name__ == "__main__":
    main()
