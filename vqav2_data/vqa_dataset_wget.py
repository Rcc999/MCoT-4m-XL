# Copyright 2020 The HuggingFace Datasets Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""VQA v2 loading script."""


import csv
import json
from multiprocessing.sharedctypes import Value
import os
from pathlib import Path
import datasets
import subprocess
import shutil


_CITATION = """
@InProceedings{VQA,
author = {Stanislaw Antol and Aishwarya Agrawal and Jiasen Lu and Margaret Mitchell and Dhruv Batra and C. Lawrence Zitnick and Devi Parikh},
title = {VQA: Visual Question Answering},
booktitle = {International Conference on Computer Vision (ICCV)},
year = {2015},
}
"""

_DESCRIPTION = """
VQA is a new dataset containing open-ended questions about images. These questions require an understanding of vision, language and commonsense knowledge to answer.
"""

_HOMEPAGE = "https://visualqa.org"

_LICENSE = "CC BY 4.0"

_URLS = {
    "questions": {
        "train": "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Train_mscoco.zip",
        "val": "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Val_mscoco.zip",
        "test-dev": "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Test_mscoco.zip",
        "test": "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Test_mscoco.zip",
    },
    "annotations": {
        "train": "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Train_mscoco.zip",
        "val": "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Val_mscoco.zip",
    },
    "images": {
        "train": "http://images.cocodataset.org/zips/train2014.zip",
        "val": "http://images.cocodataset.org/zips/val2014.zip",
        "test-dev": "http://images.cocodataset.org/zips/test2015.zip",
        "test": "http://images.cocodataset.org/zips/test2015.zip",
    },
}
_SUB_FOLDER_OR_FILE_NAME = {
    "questions": {
        "train": "v2_OpenEnded_mscoco_train2014_questions.json",
        "val": "v2_OpenEnded_mscoco_val2014_questions.json",
        "test-dev": "v2_OpenEnded_mscoco_test-dev2015_questions.json",
        "test": "v2_OpenEnded_mscoco_test2015_questions.json",
    },
    "annotations": {
        "train": "v2_mscoco_train2014_annotations.json",
        "val": "v2_mscoco_val2014_annotations.json",
    },
    "images": {
        "train": "train2014",
        "val": "val2014",
        "test-dev": "test2015",
        "test": "test2015",
    },
}


class VQAv2Dataset(datasets.GeneratorBasedBuilder):

    VERSION = datasets.Version("1.0.0")

    def _info(self):
        features = datasets.Features(
            {
                "question_type": datasets.Value("string"),
                "multiple_choice_answer": datasets.Value("string"),
                "answers": [
                    {
                        "answer": datasets.Value("string"),
                        "answer_confidence": datasets.Value("string"),
                        "answer_id": datasets.Value("int64"),
                    }
                ],
                "image_id": datasets.Value("int64"),
                "answer_type": datasets.Value("string"),
                "question_id": datasets.Value("int64"),
                "question": datasets.Value("string"),
                "image": datasets.Image(),
            }
        )
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        # Get the manual directory path provided by the user (if any) via load_dataset(..., data_dir=...)
        user_manual_dir = dl_manager.manual_dir

        if user_manual_dir is None:
            # If no manual_dir is specified, default to "vqa_v2_manual_downloads" in the current working directory.
            # This is based on the user's command `cache_dir='.'` and the original script's implied name.
            current_working_dir = Path(os.getcwd())
            data_dir = current_working_dir / "vqa_v2_manual_downloads"
            print(
                f"INFO: 'data_dir' was not specified in load_dataset. "
                f"Defaulting to '{data_dir}' for downloads and extractions. "
                f"It is recommended to specify 'data_dir' explicitly for manual downloads."
            )
        else:
            data_dir = Path(user_manual_dir)

        # Ensure the target data_dir exists. This directory will store downloaded zips and extracted contents.
        # The original script created it, so we maintain that behavior.
        # Using os.makedirs with exist_ok=True is safer.
        if not data_dir.exists():
            os.makedirs(data_dir, exist_ok=True)

        split_paths = {}

        for split_name in ["train", "val", "test-dev", "test"]:
            split_paths[split_name] = {}
            for dir_name in _URLS.keys():
                url = _URLS[dir_name].get(split_name)
                if url:
                    # data_dir is now a Path object, so direct / operator can be used.
                    downloaded_file = data_dir / Path(url).name
                    extracted_path = data_dir / f"{dir_name}_{split_name}"

                    if not extracted_path.exists():
                        if not downloaded_file.exists():
                            print(f"Downloading {url} to {downloaded_file}")
                            subprocess.run(["wget", url, "-O", str(downloaded_file)], check=True)

                        if downloaded_file.suffix == ".zip":
                            print(f"Extracting {downloaded_file} to {extracted_path}")
                            os.makedirs(extracted_path, exist_ok=True) # Ensure extraction target dir exists
                            subprocess.run(["unzip", str(downloaded_file), "-d", str(extracted_path)], check=True)
                        else:
                            print(f"Skipping extraction for non-zip file: {downloaded_file}")
                            # If the file is not a zip, the original script pointed extracted_path to the file itself.
                            # This might be problematic if _SUB_FOLDER_OR_FILE_NAME expects extracted_path to be a directory.
                            # For now, maintaining original logic. If this causes issues, it might need to be
                            # shutil.copy(downloaded_file, extracted_path) if extracted_path is meant to be a dir.
                            extracted_path = downloaded_file 

                    expected_final_path = extracted_path / _SUB_FOLDER_OR_FILE_NAME[dir_name][split_name]
                    split_paths[split_name][f"{dir_name}_path"] = expected_final_path
                else:
                    split_paths[split_name][f"{dir_name}_path"] = None


        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs=split_paths["train"],
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs=split_paths["val"],
            ),
            datasets.SplitGenerator(
                name="testdev",
                gen_kwargs=split_paths["test-dev"],
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs=split_paths["test"],
            ),
        ]

    def _generate_examples(self, questions_path, annotations_path, images_path):
        # Ensure paths are Path objects if they are not None
        questions_path = Path(questions_path) if questions_path else None
        annotations_path = Path(annotations_path) if annotations_path else None
        images_path = Path(images_path) if images_path else None

        if not questions_path or not questions_path.exists():
             raise FileNotFoundError(f"Questions file not found or path is None: {questions_path}")

        with open(questions_path, "r", encoding='utf-8') as f:
            questions = json.load(f)


        if annotations_path is not None:
            if not annotations_path.exists():
                raise FileNotFoundError(f"Annotations file not found: {annotations_path}")
            with open(annotations_path, "r", encoding='utf-8') as f:
                 dataset = json.load(f)

            qa = {ann["question_id"]: [] for ann in dataset["annotations"]} # Initialize with empty list
            for ann in dataset["annotations"]:
                qa[ann["question_id"]] = ann

            for question in questions["questions"]:
                annotation = qa[question["question_id"]]
                # Original asserts:
                assert len(set(question.keys()) ^ set(["image_id", "question", "question_id"])) == 0
                assert (
                    len(
                        set(annotation.keys())
                        ^ set(
                            [
                                "question_type",
                                "multiple_choice_answer",
                                "answers",
                                "image_id",
                                "answer_type",
                                "question_id",
                            ]
                        )
                    )
                    == 0
                )
                record = question.copy() # Use copy to avoid modifying the original question dict from questions list
                record.update(annotation)
                if not images_path or not images_path.exists(): # images_path is a directory
                    raise FileNotFoundError(f"Images directory not found or path is None: {images_path}")
                record["image"] = str(images_path / f"COCO_{images_path.name}_{record['image_id']:0>12}.jpg")
                yield question["question_id"], record
        else:
            # This branch is for test sets that don't have annotations
            for question in questions["questions"]:
                record = question.copy() # Use copy
                record.update(
                    {
                        "question_type": None,
                        "multiple_choice_answer": None,
                        "answers": None, # Or [] if an empty list is more appropriate for the schema
                        "answer_type": None,
                    }
                )
                if not images_path or not images_path.exists(): # images_path is a directory
                    raise FileNotFoundError(f"Images directory not found or path is None: {images_path}")
                record["image"] = str(images_path / f"COCO_{images_path.name}_{question['image_id']:0>12}.jpg")
                yield question["question_id"], record
