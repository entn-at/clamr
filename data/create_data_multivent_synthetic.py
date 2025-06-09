#!/usr/bin/env python3
import os
import json
import glob
import random
from collections import defaultdict
from tqdm import tqdm
from datasets import Dataset, DatasetDict
import sys
import pandas as pd

class DataFileNotFoundError(Exception):
    """Exception raised when required data files are not found."""
    pass

def validate_example(example):
    """Validate if an example has required fields."""
    if not isinstance(example, dict):
        return False
    return "query" in example and "query_type" in example

def convert_video_id_to_frame_path(video_id, frame_num):
    """Convert video ID and frame number to frame file path."""
    # Expected input: "001085/lryNvnMvHwY"
    # Expected output: "train_001085_lryNvnMvHwY_frame_{frame_num}.jpg"
    return f"{video_id}_frame_{frame_num}.jpg"


frame_dir = "/nas-hdd/davidwan/frames/multivent"

def main():
    data_dir = sys.argv[1]
    ocr_dir = f"{data_dir}/features/train/exp/scale24/features/train/ocr/icdar_ocr_max"
    asr_dir = f"{data_dir}/features/train/exp/scale24/data/video/baseline/train/speech_to_text/speech_to_text"
    description_dir = f"{data_dir}/train"

    # Load video to images mapping
    mapping_file = sys.argv[2]
    video_images_mapping = json.load(open(mapping_file))
    # change the key into the correct video_id format
    video_images_mapping_ = {}
    for key, value in video_images_mapping.items():
        # 000001/edhOSNX8Qg0 -> 000001_edhOSNX8Qg0.mp4
        video_id = "train_" + key.split("/")[0] + "_" + key.split("/")[1]
        video_images_mapping_[video_id[:-4]] = value
    video_images_mapping = video_images_mapping_

    # Load new train/val query files
    train_queries_file = sys.argv[3]
    val_queries_file = sys.argv[4]
    with open(train_queries_file, "r") as f:
        train_queries = json.load(f)
    with open(val_queries_file, "r") as f:
        val_queries = json.load(f)

    def build_examples(queries_dict, split_name):
        examples = []
        missing_videos = 0
        for video_id, query_list in tqdm(queries_dict.items(), desc=f"Processing {split_name} queries"):
            if video_id not in video_images_mapping:
                print(f"Video ID not found in mapping: {video_id}")
                missing_videos += 1
                continue
            frame_nums = video_images_mapping[video_id]
            frame_paths = [convert_video_id_to_frame_path(video_id, frame_num) for frame_num in frame_nums]
            valid_frame_paths = [fp for fp in frame_paths if os.path.exists(os.path.join(frame_dir, fp))]
            
            #  Parse video_id to extract components
            try:
                split = "train"
                video_id = video_id.lstrip("train_")
                dir_name, video_name = video_id.split('_', 1)
            except ValueError:
                raise ValueError(f"Could not parse video_id {video_id}")
            
            ocr_data = []
            if ocr_dir:
                # Find all OCR files for this video
                ocr_frame_prefix = f"{dir_name}/{video_name}.mp4_frames/{video_name}_frame_"
                ocr_pattern = os.path.join(ocr_dir, ocr_frame_prefix + "*.png.csv")
                ocr_files = glob.glob(ocr_pattern)
                
                if not ocr_files:
                    print(f"Warning: No OCR files found for video {video_id} with pattern {ocr_pattern}")
                
                for ocr_file in ocr_files:
                    try:
                        if os.path.getsize(ocr_file) > 0:  # Check if file is not empty
                            df = pd.read_csv(ocr_file)
                            for _, row in df.iterrows():
                                # Only extract and store the text from OCR
                                ocr_text = row.get("text", "")
                                # Skip None, nan or empty values
                                if ocr_text and pd.notna(ocr_text) and str(ocr_text).strip():
                                    ocr_data.append(str(ocr_text).strip())
                    except Exception as e:
                        raise DataFileNotFoundError(f"Error reading OCR file {ocr_file}: {e}")
            
            # Get description if available
            description = ""
            title = ""
            if description_dir:
                desc_file = os.path.join(description_dir, dir_name, f"{video_name}.json")
                try:
                    if os.path.exists(desc_file):
                        with open(desc_file, 'r') as f:
                            desc_data = json.load(f)
                            description = desc_data.get("yt_meta_dict", {}).get("info", {}).get("description", "")
                            title = desc_data.get("yt_meta_dict", {}).get("info", {}).get("title", "")
                            # Ensure description and title are not None or nan
                            if description is None or pd.isna(description):
                                description = ""
                            if title is None or pd.isna(title):
                                title = ""
                    else:
                        raise DataFileNotFoundError(f"Description file not found: {desc_file}")
                except Exception as e:
                    raise DataFileNotFoundError(f"Error reading description file {desc_file}: {e}")
            
            # Get ASR data if available
            asr_text = ""
            if asr_dir:
                asr_file = os.path.join(asr_dir, dir_name, f"{video_name}.m4a.csv")
                try:
                    if os.path.exists(asr_file):
                        if os.path.getsize(asr_file) > 0:
                            try:
                                df = pd.read_csv(asr_file)
                                if "text" in df.columns:
                                    # Convert all values to strings to avoid type errors
                                    try:
                                        text_values = [str(text) for text in df["text"] if pd.notna(text)]
                                        asr_text = " ".join(text_values)
                                    except (TypeError, ValueError) as type_err:
                                        print(f"Warning: Type error in ASR file {asr_file}: {type_err}. Using empty string.")
                                        asr_text = ""
                                else:
                                    asr_text = ""
                            except Exception as csv_err:
                                print(f"Warning: Failed to process ASR file {asr_file}: {csv_err}. Using empty string.")
                                asr_text = ""
                        else:
                            print(f"Warning: ASR file is empty: {asr_file}")
                    else:
                        raise DataFileNotFoundError(f"ASR file not found: {asr_file}")
                except DataFileNotFoundError as e:
                    raise e
                except Exception as e:
                    print(f"Warning: Error reading ASR file {asr_file}: {e}. Using empty string.")
                    asr_text = ""
            
            example = {
                "video_id": video_id,
                "query": query_list.get("query", ""),
                "query_type": query_list.get("query_type", ""),
                "images": valid_frame_paths,
                "ocr": " ".join(ocr_data),
                "description": description,
                "asr": asr_text
            }
            examples.append(example)
        if missing_videos > 0:
            print(f"{missing_videos} videos in {split_name} not found in mapping.")
        return examples

    train_examples = build_examples(train_queries, "train")
    val_examples = build_examples(val_queries, "validation")

    print(f"\nTrain examples: {len(train_examples)}")
    print(f"Validation examples: {len(val_examples)}")

    # Count examples with images
    train_with_images = sum(1 for ex in train_examples if ex["images"])
    val_with_images = sum(1 for ex in val_examples if ex["images"])
    print(f"Train examples with images: {train_with_images} ({train_with_images/len(train_examples)*100:.2f}%)")
    print(f"Validation examples with images: {val_with_images} ({val_with_images/len(val_examples)*100:.2f}%)")

    # Query type distribution
    from collections import defaultdict
    train_query_types = defaultdict(int)
    val_query_types = defaultdict(int)
    for ex in train_examples:
        train_query_types[ex["query_type"]] += 1
    for ex in val_examples:
        val_query_types[ex["query_type"]] += 1
    print("\nTrain query types:")
    for qt, count in sorted(train_query_types.items()):
        print(f"  {qt}: {count}")
    print("Validation query types:")
    for qt, count in sorted(val_query_types.items()):
        print(f"  {qt}: {count}")

    # Create output directories
    output_dir = os.path.join(base_dir, "multivent_synthetic_hf")

    # Ensure all examples have the same keys
    all_keys = set()
    for example in train_examples + val_examples:
        all_keys.update(example.keys())
    for example in train_examples + val_examples:
        for key in all_keys:
            if key not in example:
                if key == "images":
                    example[key] = []
                elif key in ["ocr", "description", "asr"]:
                    example[key] = ""
                else:
                    example[key] = None

    train_dataset = Dataset.from_list(train_examples)
    val_dataset = Dataset.from_list(val_examples)

    print("Train dataset features:", train_dataset.features)
    print("Train dataset column names:", train_dataset.column_names)

    dataset = DatasetDict({"train": train_dataset, "validation": val_dataset})

    print(f"Saving training dataset to {output_dir}...")
    dataset.save_to_disk(output_dir)

    print("\nDataset creation completed!")

if __name__ == "__main__":
    main() 