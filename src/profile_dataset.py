import os
import csv
import json
from collections import defaultdict

def profile_dataset(base_dir):
    report = {
        "files": {
            "S1": [],
            "S2": [],
            "S3": [],
            "ground_truth": []
        },
        "datasets": {
            "train": [],
            "test": []
        },
        "sources": {},
        "ground_truth_structure": {},
        "mapping_logic": ""
    }
    
    # Track files
    for split in ["train", "test"]:
        split_dir = os.path.join(base_dir, split)
        if not os.path.exists(split_dir):
            continue
            
        for fname in os.listdir(split_dir):
            if not fname.endswith(".tsv"):
                continue
            
            fpath = os.path.join(split_dir, fname)
            report["datasets"][split].append(fname)
            
            if "source1" in fname: report["files"]["S1"].append(fname)
            elif "source2" in fname: report["files"]["S2"].append(fname)
            elif "source3" in fname: report["files"]["S3"].append(fname)
            elif "ground_truth" in fname: report["files"]["ground_truth"].append(fname)

            # Process Ground Truth
            if "ground_truth" in fname:
                with open(fpath, "r", encoding="utf-8") as f:
                    reader = csv.reader(f, delimiter="\t")
                    headers = next(reader)
                    
                    row_count = 0
                    sample_row = None
                    for row in reader:
                        if row_count == 0:
                            sample_row = row
                        row_count += 1
                        
                    report["ground_truth_structure"] = {
                        "columns": headers,
                        "row_count": row_count,
                        "sample": dict(zip(headers, sample_row)) if sample_row else {}
                    }
                    
                    report["mapping_logic"] = "S1 entity IDs map to a comma-separated list of S2 and S3 entity IDs in the 'matched_entity_ids' column."
                continue

            # Process Sources (S1, S2, S3)
            with open(fpath, "r", encoding="utf-8") as f:
                reader = csv.reader(f, delimiter="\t")
                try:
                    headers = next(reader)
                except StopIteration:
                    continue
                
                stats = {
                    "columns": headers,
                    "row_count": 0,
                    "missing_values": {"business_name": 0, "business_address": 0, "country": 0},
                    "duplicate_entity_ids": 0,
                    "country_distribution": defaultdict(int),
                    "lengths": {
                        "business_name": {"min": float('inf'), "max": 0, "sum": 0, "valid_count": 0},
                        "business_address": {"min": float('inf'), "max": 0, "sum": 0, "valid_count": 0}
                    }
                }
                
                seen_ids = set()
                
                name_idx = headers.index("business_name") if "business_name" in headers else -1
                addr_idx = headers.index("business_address") if "business_address" in headers else -1
                country_idx = headers.index("country") if "country" in headers else -1
                id_idx = headers.index("entity_id") if "entity_id" in headers else -1

                # use large field size limit
                csv.field_size_limit(10**7)

                for row in reader:
                    stats["row_count"] += 1
                    
                    # ID
                    if id_idx != -1 and id_idx < len(row):
                        eid = row[id_idx]
                        if eid in seen_ids:
                            stats["duplicate_entity_ids"] += 1
                        seen_ids.add(eid)
                        
                    # Missing values & Lengths
                    if name_idx != -1 and name_idx < len(row):
                        val = row[name_idx].strip()
                        if not val or val.lower() in ["nan", "null", "na", "none", "\\n"]:
                            stats["missing_values"]["business_name"] += 1
                        else:
                            length = len(val)
                            stats["lengths"]["business_name"]["min"] = min(stats["lengths"]["business_name"]["min"], length)
                            stats["lengths"]["business_name"]["max"] = max(stats["lengths"]["business_name"]["max"], length)
                            stats["lengths"]["business_name"]["sum"] += length
                            stats["lengths"]["business_name"]["valid_count"] += 1
                    else:
                        stats["missing_values"]["business_name"] += 1
                        
                    if addr_idx != -1 and addr_idx < len(row):
                        val = row[addr_idx].strip()
                        if not val or val.lower() in ["nan", "null", "na", "none", "\\n"]:
                            stats["missing_values"]["business_address"] += 1
                        else:
                            length = len(val)
                            stats["lengths"]["business_address"]["min"] = min(stats["lengths"]["business_address"]["min"], length)
                            stats["lengths"]["business_address"]["max"] = max(stats["lengths"]["business_address"]["max"], length)
                            stats["lengths"]["business_address"]["sum"] += length
                            stats["lengths"]["business_address"]["valid_count"] += 1
                    else:
                        stats["missing_values"]["business_address"] += 1
                        
                    if country_idx != -1 and country_idx < len(row):
                        val = row[country_idx].strip()
                        if not val or val.lower() in ["nan", "null", "na", "none", "\\n"]:
                            stats["missing_values"]["country"] += 1
                        else:
                            stats["country_distribution"][val] += 1
                    else:
                        stats["missing_values"]["country"] += 1

                # Finalize length stats (calc mean, remove sum)
                for field in ["business_name", "business_address"]:
                    if stats["lengths"][field]["valid_count"] > 0:
                        stats["lengths"][field]["mean"] = stats["lengths"][field]["sum"] / stats["lengths"][field]["valid_count"]
                    else:
                        stats["lengths"][field]["mean"] = 0
                        stats["lengths"][field]["min"] = 0
                    del stats["lengths"][field]["sum"]
                    del stats["lengths"][field]["valid_count"]
                
                # Sort country distribution by count descending, keep top 10 for brevity
                stats["country_distribution"] = dict(sorted(stats["country_distribution"].items(), key=lambda item: item[1], reverse=True)[:10])
                
                report["sources"][fname] = stats

    with open("outputs/dataset_profile.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)
        
    print("Dataset profiling complete. Report saved to outputs/dataset_profile.json")

if __name__ == "__main__":
    profile_dataset("student_resource/dataset")
