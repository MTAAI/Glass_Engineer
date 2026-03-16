"""
Glass Expert AI — Generate Training Q&A Directly from SciGlass CSV
===================================================================
Reads glass_data.csv directly (no DB needed) and generates Q&A training
pairs for fine-tuning. Skips rows already covered by DB ingestion.

Usage:
    python scripts/generate_sciglass_training.py
    python scripts/generate_sciglass_training.py --skip-rows 10680
"""
import os
import sys
import json
import re
import random
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger

logger.remove()
logger.add(sys.stdout, format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}", level="INFO")

CSV_PATH = Path("data/csv/glass_data.csv")
OUTPUT_DIR = Path("models/qwen14b-glass-expert/data")

SYSTEM_PROMPT = (
    "You are Glass Expert AI, a highly specialized assistant trained on thousands "
    "of glass science research papers, textbooks, and material databases. You provide "
    "accurate, detailed, and technically precise answers about glass composition, "
    "properties, manufacturing processes, defects, characterization, and applications. "
    "Always cite relevant glass science principles in your answers."
)

SCIGLASS_Q_TEMPLATES = [
    "What is the composition of glass #{glass_id}?",
    "Describe the properties of glass #{glass_id}.",
    "What are the components and properties of glass #{glass_id}?",
    "Tell me about the composition and characteristics of glass #{glass_id}.",
    "What glass system does glass #{glass_id} belong to and what are its properties?",
    "Provide the detailed composition and measured properties for glass #{glass_id}.",
]

COMPOSITION_Q_TEMPLATES = [
    "What glass compositions contain {component} at around {value}%?",
    "Describe a glass with {component} content of approximately {value}%.",
    "What are the properties of glasses with high {component} content?",
    "Tell me about glasses in the {system} system.",
]

PROPERTY_Q_TEMPLATES = [
    "What are the thermal properties of glass #{glass_id}?",
    "What is the density and refractive index of glass #{glass_id}?",
    "Describe the physical properties measured for glass #{glass_id}.",
]


def load_property_names():
    """Load LISTPROP.csv to map property codes to readable names."""
    prop_path = Path("data/csv/LISTPROP.csv")
    if prop_path.exists():
        df = pd.read_csv(str(prop_path), encoding="utf-8", on_bad_lines="skip")
        mapping = {}
        for _, row in df.iterrows():
            fld = str(row.get("FLDNAM", "")).strip().lower()
            name = str(row.get("NAME", "")).strip()
            unit = str(row.get("Unit", "")).strip()
            if fld and name:
                mapping[fld] = f"{name} ({unit})" if unit and unit != "nan" else name
        return mapping
    return {}


def row_to_text(row, prop_names):
    """Convert a sparse glass data row to natural language."""
    glass_id = row.get("glass_id", "unknown")
    comps = []
    props = []

    for col, val in row.items():
        if pd.isna(val) or str(val).strip() == "" or col == "glass_id":
            continue
        val_str = str(val).strip()

        if col.startswith("comp_"):
            component = col[5:].upper().replace("_", "")
            try:
                fval = float(val_str)
                if fval > 0:
                    comps.append((component, fval))
            except ValueError:
                pass

        elif col.startswith("prop_") or col.startswith("t_") or col.startswith("d_"):
            readable = prop_names.get(col.lower(), col)
            try:
                fval = float(val_str)
                props.append((readable, fval))
            except ValueError:
                props.append((readable, val_str))

    return glass_id, comps, props


def generate_qa_for_row(glass_id, comps, props):
    """Generate 1-2 Q&A pairs from a single glass row."""
    if not comps and not props:
        return []

    pairs = []

    # Build text description
    comp_str = ", ".join(f"{c}: {v:.2f}%" for c, v in comps[:20])
    prop_str = ", ".join(f"{n}={v:.4g}" if isinstance(v, float) else f"{n}={v}" for n, v in props[:15])

    parts = [f"Glass #{glass_id}"]
    if comp_str:
        parts.append(f"Composition: {comp_str}")
    if prop_str:
        parts.append(f"Properties: {prop_str}")
    full_text = " -- ".join(parts)

    # Q&A pair 1: General composition/properties
    q = random.choice(SCIGLASS_Q_TEMPLATES).format(glass_id=glass_id)
    a = f"From the SciGlass database:\n\n{full_text}"
    pairs.append({
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": q},
            {"role": "assistant", "content": a},
        ]
    })

    # Q&A pair 2 (50% chance): Composition-focused or property-focused
    if random.random() < 0.5 and comps:
        # Pick a major component
        major = max(comps, key=lambda x: x[1])
        system_components = [c for c, v in comps if v > 5]
        system_name = "-".join(sorted(c for c in system_components[:4]))

        if system_name:
            q2 = random.choice(COMPOSITION_Q_TEMPLATES).format(
                component=major[0], value=f"{major[1]:.1f}", system=system_name
            )
        else:
            q2 = f"What is known about the composition of glass #{glass_id}?"

        a2 = f"Glass #{glass_id} from the SciGlass database has the following composition: {comp_str}."
        if prop_str:
            a2 += f" Its measured properties include: {prop_str}."

        pairs.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": q2},
                {"role": "assistant", "content": a2},
            ]
        })

    return pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-rows", type=int, default=10680,
                        help="Skip first N rows (already in DB)")
    parser.add_argument("--limit", type=int, default=0, help="Max rows to process")
    args = parser.parse_args()

    logger.info(f"Reading {CSV_PATH}...")
    prop_names = load_property_names()
    logger.info(f"Loaded {len(prop_names)} property name mappings")

    output_path = OUTPUT_DIR / "sciglass_train.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    generated = 0
    skipped = 0
    row_count = 0

    with open(output_path, "w", encoding="utf-8") as f:
        chunk_iter = pd.read_csv(
            str(CSV_PATH), encoding="utf-8", on_bad_lines="skip",
            chunksize=10000, low_memory=False
        )

        for df_chunk in chunk_iter:
            for _, row in df_chunk.iterrows():
                row_count += 1

                if row_count <= args.skip_rows:
                    continue

                if args.limit > 0 and (row_count - args.skip_rows) > args.limit:
                    break

                glass_id, comps, props = row_to_text(row, prop_names)

                if not comps and not props:
                    skipped += 1
                    continue

                qa_pairs = generate_qa_for_row(glass_id, comps, props)
                for qa in qa_pairs:
                    f.write(json.dumps(qa, ensure_ascii=False) + "\n")
                    generated += 1

                if generated % 50000 == 0 and generated > 0:
                    logger.info(f"  Generated {generated:,} Q&A pairs from {row_count:,} rows...")

            if args.limit > 0 and (row_count - args.skip_rows) > args.limit:
                break

    logger.info(f"\nSciGlass training data generation complete!")
    logger.info(f"  Rows processed: {row_count:,}")
    logger.info(f"  Skipped (empty): {skipped:,}")
    logger.info(f"  Q&A pairs generated: {generated:,}")
    logger.info(f"  Output: {output_path}")

    # Now merge everything: combined_train + chunks_train + sciglass_train
    merged_path = OUTPUT_DIR / "merged_train.jsonl"
    files_to_merge = [
        OUTPUT_DIR / "combined_train.jsonl",
        OUTPUT_DIR / "chunks_train.jsonl",
        output_path,
    ]

    total = 0
    with open(merged_path, "w", encoding="utf-8") as out:
        for src in files_to_merge:
            if src.exists():
                count = 0
                with open(src, "r", encoding="utf-8") as inp:
                    for line in inp:
                        line = line.strip()
                        if line:
                            out.write(line + "\n")
                            count += 1
                            total += 1
                logger.info(f"  Merged {count:,} from {src.name}")
            else:
                logger.warning(f"  Missing: {src}")

    logger.info(f"\nFinal merged training file: {merged_path}")
    logger.info(f"  Total training examples: {total:,}")


if __name__ == "__main__":
    main()
