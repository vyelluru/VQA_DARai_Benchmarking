# import configparser
# from dataset import VideoFrameDataset
# from models import load_model_and_processor, generate_answer
# from utils import set_seed
# import csv
# import os
# import json
#
#
# def LLaVa_NeXT_Video_generator(config_filename):
#
#     with open(config_filename, "r") as f:
#         cfg = json.load(f)
#
#     dataset_root = cfg.get("dataset_root", "/mnt/Data1/RGB_sd")
#     sequence_length = cfg.get("sequence_length", 16)
#     csv_filename = cfg.get("csv_filename", "output_answers.csv")
#     question = cfg.get("question", "What is the person doing in this video?")
#     max_new_tokens = cfg.get("max_new_tokens", 100)
#     seed = cfg.get("seed", 42)
#     set_seed(seed)
#
#     # Load the dataset
#     dataset = VideoFrameDataset(root_dir=dataset_root, sequence_length=sequence_length)
#     print(f"Loaded dataset from {dataset_root} with {len(dataset)} samples.")
#
#     # Load the model and processor
#     model, processor = load_model_and_processor()
#     print("Loaded model and processor.")
#
#     # Prepare the CSV file: write header if file does not exist
#     csv_exists = os.path.exists(csv_filename)
#     if not csv_exists:
#         with open(csv_filename, mode="w", newline="", encoding="utf-8") as f:
#             writer = csv.DictWriter(f,
#                                     fieldnames=["activity", "camera", "subject_id", "session_id", "question", "answer"])
#             writer.writeheader()
#
#     # Loop through the dataset and process each sample
#     for idx in range(len(dataset)):
#         try:
#             instance = dataset[idx]
#             answer = generate_answer(instance, question, processor, model, max_new_tokens=max_new_tokens)
#             # Extract metadata (assuming instance returns: frames, activity, camera, (subject_id, session_id))
#             _, activity, camera, (subject_id, session_id) = instance
#
#             # Create the output dictionary. Here we clean the answer by splitting on "ASSISTANT:".
#             try:
#                 cleaned_answer = answer.split("ASSISTANT:")[1].strip()
#             except IndexError:
#                 cleaned_answer = answer.strip()
#
#             output_entry = {
#                 "activity": activity,
#                 "camera": camera,
#                 "subject_id": subject_id,
#                 "session_id": session_id,
#                 "question": question,
#                 "answer": cleaned_answer
#             }
#
#             # Append the current sample's answer to the CSV file
#             with open(csv_filename, mode="a", newline="", encoding="utf-8") as f:
#                 writer = csv.DictWriter(f, fieldnames=["activity", "camera", "subject_id", "session_id", "question",
#                                                        "answer"])
#                 writer.writerow(output_entry)
#
#             print(f"Processed sample {idx}: {output_entry}")
#         except Exception as e:
#             print(f"Error processing sample {idx}: {e}")
#             # Optionally, log the error or break here if needed
#
#     print("Saved answers to", csv_filename)
#     return csv_filename

import csv
import os
import json
from dataset import VideoFrameDataset
from models import load_model_and_processor, generate_answer
from utils import set_seed


def LLaVa_NeXT_Video_generator(config_filename):
    """
    Generates answers for multiple questions from a dataset using LLaVA-NeXT-Video.
    Saves results incrementally to avoid data loss.
    """

    # Load configuration
    with open(config_filename, "r") as f:
        cfg = json.load(f)

    dataset_root = cfg.get("dataset_root", "/mnt/Data1/RGB_sd")
    sequence_length = cfg.get("sequence_length", 16)
    csv_filename = cfg.get("csv_filename", "output_answers.csv")
    questions = cfg.get("questions", ["What is the person doing in this video?"])  # List of questions
    max_new_tokens = cfg.get("max_new_tokens", 100)


    # Load the dataset
    dataset = VideoFrameDataset(root_dir=dataset_root, sequence_length=sequence_length)
    print(f"Loaded dataset from {dataset_root} with {len(dataset)} samples.")

    # Load the model and processor
    model, processor = load_model_and_processor()
    print("Loaded model and processor.")

    # Prepare the CSV file: Write header if file does not exist
    csv_exists = os.path.exists(csv_filename)
    if not csv_exists:
        with open(csv_filename, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f,
                                    fieldnames=["activity", "camera", "subject_id", "session_id", "question", "answer"])
            writer.writeheader()

    # Loop through the dataset
    for idx in range(len(dataset)):
        try:
            instance = dataset[idx]
            _, activity, camera, (subject_id, session_id) = instance

            for question in questions:  # Loop through all questions
                try:
                    answer = generate_answer(instance, question, processor, model, max_new_tokens=max_new_tokens)
                    cleaned_answer = answer.split("ASSISTANT:")[1].strip() if "ASSISTANT:" in answer else answer.strip()

                    output_entry = {
                        "activity": activity,
                        "camera": camera,
                        "subject_id": subject_id,
                        "session_id": session_id,
                        "question": question,
                        "answer": cleaned_answer
                    }

                    # Append the answer to the CSV file
                    with open(csv_filename, mode="a", newline="", encoding="utf-8") as f:
                        writer = csv.DictWriter(f, fieldnames=["activity", "camera", "subject_id", "session_id",
                                                               "question", "answer"])
                        writer.writerow(output_entry)

                    print(f"Processed sample {idx}, Question: {question}")

                except Exception as qe:
                    print(f"Error processing question '{question}' for sample {idx}: {qe}")
                    continue  # Continue to the next question

        except Exception as e:
            print(f"Error processing sample {idx}: {e}")
            continue  # Skip to the next sample

    print("Saved answers to", csv_filename)
    return csv_filename
