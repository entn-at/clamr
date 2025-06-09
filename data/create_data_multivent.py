import os
from tqdm import tqdm
import json
import pandas as pd
import glob
from datasets import Dataset, DatasetDict
import sys

class DataFileNotFoundError(Exception):
    """Exception raised when required data files are not found."""
    pass

def create_dataset_entries(split_data, ocr_dir=None, description_dir=None, asr_dir=None, frame_mapping=None):
    """
    Create dataset entries without symbolic links
    
    Parameters:
    - ocr_dir: Directory containing OCR data
    - description_dir: Directory containing video metadata (for descriptions)
    - asr_dir: Directory containing ASR data
    - frame_mapping: Mapping of video paths to frame numbers
    """
    data = []
    
    for entry in tqdm(split_data, desc="Processing entries"):
        video_id = entry["video_id"]
        query = entry["query"]
        query_id = entry["query_id"]
        query_type = entry["query_type"]
        video_path = entry.get("video_path", "")  # Get video path if available
        
        # Parse video_id to extract components
        try:
            split = "train"
            dir_name, video_name = video_id.split('_', 1)
        except ValueError:
            raise ValueError(f"Could not parse video_id {video_id}")
            
        # Get OCR data if available
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
        
        # Get frame numbers from mapping if available
        frame_numbers = []
        if "frame_numbers" in entry:
            # Use pre-validated frame numbers from entry
            frame_numbers = entry["frame_numbers"]
        elif frame_mapping and video_path in frame_mapping:
            # Try to get from mapping, but don't validate here
            frame_numbers = frame_mapping[video_path]
            # Limit to num_frames if needed
            if len(frame_numbers) > 10:  # Default to 10 frames
                frame_numbers = frame_numbers[:10]
        else:
            # Fallback to uniform frames if mapping not available
            frame_numbers = list(range(10))
            
        # Skip entries with no frames
        if not frame_numbers:
            print(f"Warning: No frames available for {video_id}, skipping")
            continue
            
        # Create file_names list based on frame numbers
        file_names = [f"train_{video_id}_frame_{i}.jpg" for i in frame_numbers]
            
        data.append({
            "video_id": video_id,
            "images": file_names,
            "query": query,
            "query_id": query_id,
            "query_type": query_type,
            "ocr": " ".join(ocr_data),
            "description": description,
            "title": title,
            "asr": asr_text
        })
    
    return data

def main():
    data_dir = sys.argv[1]
    ocr_dir = f"{data_dir}/features/train/exp/scale24/features/train/ocr/icdar_ocr_max"
    asr_dir = f"{data_dir}/features/train/exp/scale24/data/video/baseline/train/speech_to_text/speech_to_text"
    description_dir = f"{data_dir}/train"

    judgments = [json.loads(line) for line in open(os.path.join(data_dir, "multivent_2_train_judgments.jsonl")).readlines()] # {"query_id": "internvid_english_sports_pov_base_0", "doc_id": "PPQVve-Jeb4", "relevance": 1}
    vidid2id = {line.strip():i for i, line in enumerate(open(os.path.join(data_dir,"multivent_2_ids.csv")).readlines()[1:])}
    queryid2query = {line.strip().split(",")[0]:line.strip().split(",")[1] for line in open(os.path.join(data_dir,"multivent_2_train_queries.csv")).readlines()[1:]}

    # Load the mapping file for path information
    mapping_file = json.load(open(sys.argv[2]))
    
    # Create doc_id to video path mapping
    doc_id_to_path = {}
    for video_path, frame_nums in mapping_file.items():
        # Extract doc_id from the path (last part of the path without extension)
        doc_id = os.path.splitext(os.path.basename(video_path))[0]
        doc_id_to_path[doc_id] = video_path
    
    # Create output directories
    hf_base_dir = "multivent_hf"
    
    # Process judgments to create entries with query information
    dataset_entries = []
    missing_frames = []
    
    for judgment in tqdm(judgments, desc="Processing judgments"):
        query_id = judgment["query_id"]
        doc_id = judgment["doc_id"]
        relevance = judgment["relevance"]
        
        # Skip irrelevant judgments
        if relevance == 0:
            continue
        
        # Get query text
        query = queryid2query.get(query_id)
        if not query:
            print(f"Warning: No query found for query_id {query_id}")
            continue
        
        # Get video path from mapping
        video_path = doc_id_to_path.get(doc_id)
        if not video_path:
            print(f"Warning: No video path found for doc_id {doc_id}")
            continue
        
        # Extract components from path
        path_parts = os.path.normpath(video_path).split(os.sep)
        last_parts = path_parts[-3:]  # Get last three components
        path_as_name = '_'.join(last_parts)
        path_as_name = os.path.splitext(path_as_name)[0]  # Remove extension
        
        video_id = path_as_name
        
        # Check if this video has frame mapping
        if video_path not in mapping_file:
            print(f"Warning: No frame mapping found for video {video_path}")
            continue
            
        # Check if frames exist for the specific frame numbers
        frame_numbers = mapping_file[video_path]
        if not frame_numbers:
            print(f"Warning: Empty frame list for video {video_path}")
            continue
        
        # Determine query type from query_id
        if "description" in query_id:
            query_type = "description"
        elif "ocr" in query_id:
            query_type = "ocr"
        elif "speech" in query_id:
            query_type = "speech"
        elif "base" in query_id:
            query_type = "video"
        else:
            print(f"Error: Unknown query type for query_id: {query_id}")
            query_type = "unknown"
        
        # Create a separate entry for each query-video pair
        dataset_entries.append({
            "video_id": video_id,
            "video_path": video_path,
            "query": query,
            "query_id": query_id,
            "query_type": query_type,
            "frame_numbers": frame_numbers[:10]  # Limit to 10 frames
        })
    
    print(f"Found {len(dataset_entries)} valid query-video pairs")
    
    # Count query types
    query_type_counts = {}
    for entry in dataset_entries:
        query_type = entry["query_type"]
        query_type_counts[query_type] = query_type_counts.get(query_type, 0) + 1
    
    print("\nQuery Type Statistics:")
    for query_type, count in query_type_counts.items():
        print(f"  {query_type}: {count} ({count/len(dataset_entries)*100:.2f}%)")
    
    try:
        # Create main dataset
        train_data = create_dataset_entries(
            dataset_entries,
            ocr_dir=ocr_dir,
            description_dir=description_dir,
            asr_dir=asr_dir,
            frame_mapping=mapping_file
        )
        
        # Create HuggingFace dataset and save
        train_dataset = Dataset.from_list(train_data)
        train_dataset = DatasetDict({"train": train_dataset})
        train_dataset.save_to_disk(hf_base_dir)
        
        print(f"Saved main dataset with {len(train_data)} examples to {hf_base_dir}")
    except DataFileNotFoundError as e:
        print(f"Error: {e}")
        exit(1)

if __name__ == "__main__":
    main() 