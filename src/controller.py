"""
controller.py

Main controller for MetaScribe pipeline.
"""

import yaml
import time
import os
import json
import tempfile
import fitz
from datetime import datetime
from PIL import Image

from src.core.generate_metadata import generate_single_metadata
from src.core.ocr import generate_single_ocr
from src.core.preprocessing_document import resize_image, binarize_directory
from src.core.aggregation import generate_single_aggregated_metadata


class MetaScribeController:
    """
    Main controller for the Metascribe pipeline that orchestrates document processing.
    
    This controller handles the complete workflow from preprocessing documents
    through OCR to metadata generation and aggregation, using configuration from a YAML file.
    """
    
    def __init__(self, config_path=None):
        """
        Initialize the controller with configuration.

        Args:
            config_path (str, optional): Path to the YAML configuration file.
                If None, looks in environment and current working directory.
        """
        self.config = None

        config_locations = [
            config_path,
            os.environ.get("METASCRIBE_CONFIG"),
            os.path.join(os.getcwd(), "metascribe_config.yaml")
        ]

        for loc in config_locations:
            if loc and os.path.exists(loc):
                try:
                    with open(loc, "r") as f:
                        self.config = yaml.safe_load(f)
                        print(f"\nSuccessfully loaded configuration from {loc}\n")
                        break
                except Exception as e:
                    print(f"\nError loading configuration from {loc}: {e}\n")
                    continue

        if not self.config:
            raise ValueError("""
                No valid configuration found.
                Please provide a configuration file with any of the following options:
                    - Setting the METASCRIBE_CONFIG environment variable
                    - Placing a metascribe_config.yaml file in the current working directory
                    - Providing a path to a custom configuration file as an argument
            """)

    def process_documents(self, input_dir, output_dir=None):
        """
        Process all documents in the input directory and save the results in the output directory.
            - Documents must be in PDF, JPG, or PNG format
            - Will use document's basename as ID for naming purposes

        Args:
            input_dir (str): The directory containing user's documents to process.
            output_dir (str): The directory to save the processed documents.

        Result:
            - Output will be saved in a single JSONL per document

            output_dir/
            |---- metadata/
            |    |--- document1. json
            |    |--- document2. json
            |    |--- ...
            |---- manifest. json
        """

        print(f"\nStarting MetaScribe pipeline for {input_dir}...\n")

        # Set up output directory.
        if output_dir is None:
            output_dir = os.path.join(os.getcwd(), "metascribe_output")
            print(f"\nNo output directory provided, using default: {output_dir}\n")

        if os.path.exists(output_dir):
            while True:
                user_response = input(f"\nWARNING: Output directory {output_dir} already exists. Continue? (y/n): ")
                if user_response.lower() in ["y", "yes"]:
                    break
                elif user_response.lower() in ["n", "no"]:
                    print("\nMetaScribe pipeline aborted by user (existing output directory).")
                    return
                else:
                    print("\nInvalid input. Please enter 'y' or 'n'.\n")

        # Check that json schema is valid.
        metadata_schema = json.load(open(self.config["metadata_generation"]["json_schema_path"]))
        fields_to_aggregate = self.config["aggregation"]["included_fields"]
        for field in fields_to_aggregate:
            if field not in metadata_schema["properties"]:
                print(f"Stopping MetaScribe: requested field '{field}' to aggregate not found in metadata schema. Ensure that this field is present in the JSON schema under 'properties'.") 
                return

        # Check run_config and schema. Ensure that current run matches previous run's config.
        skip_resize = self.config.get("preprocessing", {}).get("resize", {}).get("skip", False)
        skip_binarization = self.config.get("preprocessing", {}).get("binarization", {}).get("skip", False)
        skip_ocr = self.config.get("ocr", {}).get("skip", False)
        skip_aggregation = self.config.get("aggregation", {}).get("skip", False)

        manifest_path = os.path.join(output_dir, "manifest.json")
        if os.path.exists(manifest_path):
            with open(manifest_path, "r") as f:
                manifest = json.load(f)
                if "run_config" not in manifest:
                    print("ERROR: Manifest found but no run_config... manifest is corrupted. Please address this issue before running again.")
                    return
                else:
                    if manifest["run_config"] != {
                        "resize": not skip_resize,
                        "binarization": not skip_binarization,
                        "ocr": not skip_ocr,
                        "aggregation": not skip_aggregation
                    }:
                        print("ERROR: run_config mismatch from previous run. Ensure that the run_config (ie. which procedures are skipped in your current yaml file) is the same as the manifest (from previous run) to continue.")
                        return
                    
                current_schema = os.path.basename(self.config["metadata_generation"]["json_schema_path"])
                if manifest["schema_used"] != current_schema:
                    print(f"ERROR: schema mismatch from previous run. Ensure that the JSON schema used in your current yaml file ({current_schema}) is the same as the manifest ({manifest['schema_used']}) to continue.")
                    return
                

        # Get list of files in input directory
        ACCEPTABLE_EXTENSIONS = (".png", ".jpeg", ".jpg", ".pdf")
        files_to_process = [
            f for f in os.listdir(input_dir) 
            if os.path.isfile(os.path.join(input_dir, f)) and f.lower().endswith(ACCEPTABLE_EXTENSIONS)
        ]

        if len(files_to_process) == 0:
            print("Exiting... no files given to process.")
            return
        else:
            print(f"\nGiven {len(files_to_process)} valid file(s) to process.\n")

        unaccepted_files = [
            f for f in os.listdir(input_dir)
            if os.path.isfile(os.path.join(input_dir, f)) and not f.lower().endswith(ACCEPTABLE_EXTENSIONS)
        ]
        
        if unaccepted_files:
            while True:
                example_files = ", ".join(unaccepted_files[:3]) + ("..." if len(unaccepted_files) > 3 else "")
                user_response = input(f"WARNING: Detected {len(unaccepted_files)} file(s) with unsupported extensions in input directory ({example_files}). Must be one of {ACCEPTABLE_EXTENSIONS}. Continue? (y/n): ")
                if user_response.lower() in ["y", "yes"]:
                    break
                elif user_response.lower() in ["n", "no"]:
                    print("MetaScribe pipeline aborted by user (unsupported file extensions).")
                    return
                else:
                    print("\nInvalid input. Please enter 'y' or 'n'.\n")
            
        
        # Check for already successfully processed files.
        files_to_process = self._check_for_already_processed_files(files_to_process, manifest_path)     # includes extension
        if files_to_process is None:    # stop pipeline
            return
        elif not files_to_process:
            print("No new files to process. MetaScribe pipeline complete.")    # everything already processed
            return

        # All checks passed, create new directories.
        os.makedirs(output_dir, exist_ok=True)
        metadata_dir = os.path.join(output_dir, "metadata")
        os.makedirs(metadata_dir, exist_ok=True)

        # Preprocess files serially.
        current_success_processed_file_count = 0
        current_failed_processed_file_count = 0

        for file_name in files_to_process:
            previous_error_count = 0
            if os.path.exists(manifest_path):
                with open(manifest_path, "r") as f:
                    manifest = json.load(f)
                    if file_name in manifest["files"]:
                        previous_error_count = manifest["files"][file_name]["processing_error_count"]

            # File-level data.
            work_id = os.path.splitext(file_name)[0]    # IMPORTANT: uses basename (without extension) as ID for naming
            doc_metadata_path = os.path.join(metadata_dir, f"{work_id}.jsonl")
            file_data = {
                "processing_date": None,    # None means will update at end of file processing
                "metadata_file": os.path.basename(doc_metadata_path),
                "status": None,   
                "total_processing_time": None,
                "total_cost": None,
                "timing_breakdown": {
                    "preprocessing": {
                        "resize": 0,
                        "binarization": 0
                    },
                    "metadata_generation": 0,
                    "ocr": 0,
                    "aggregation":0
                },
                "cost_breakdown": {
                    "metadata_generation": 0,
                    "ocr": 0,
                    "aggregation": 0
                },
                "processing_error_count": previous_error_count
            }
            current_error_count = 0     # >0 if any error occurs: small (API call failure, etc. but will continue processing) and big (halting processing a file entirely).


            print(f"\nProcessing {file_name}\n")
            file_path = os.path.join(input_dir, file_name)
            file_extension = os.path.splitext(file_name)[1].lower()

            # Temporary directory to hold temp files while processing.
            with tempfile.TemporaryDirectory() as temp_dir:
                try:
                    # Subdirectories for images after pre-processing.
                    # STEP 1: Preprocessing -- resizing & binarization.
                    resized_dir = os.path.join(temp_dir, "resized")
                    binarized_dir = os.path.join(temp_dir, "resized_binarized")
                    os.makedirs(resized_dir)
                    os.makedirs(binarized_dir)

                    # Resolve ZigZag.jar path.
                    project_core = os.path.dirname(os.path.abspath(__file__))
                    zigzag_jar_path = os.path.join(project_core, self.config["preprocessing"]["binarization"]["jar_path"])
                    
                    # PDF files.
                    preprocessing_start_time = time.time()
                    if file_extension == ".pdf":
                        
                        with fitz.open(file_path) as pdf_doc:
                            total_pages = len(pdf_doc)

                            for page_num in range(total_pages):
                                page_idx = page_num + 1
                                page_id = f"{page_idx}_{work_id}"

                                page = pdf_doc[page_num]
                                pix = page.get_pixmap(matrix=fitz.Matrix(self.config["preprocessing"]["pdf_image_dpi"] / 72, self.config["preprocessing"]["pdf_image_dpi"] / 72), alpha=False)
                                image_pil = self._pix_to_PIL(pix)

                                # Resizing.
                                resized_path = os.path.join(resized_dir, f"{page_id}.jpg")

                                if skip_resize:
                                    # Save original image to resized, "skipping" resize.
                                    image_pil.save(resized_path)
                                    resized_image = image_pil
                                else:
                                    resized_start_time = time.time()
                                    resized_image = resize_image(image_pil,
                                                                max_width=self.config["preprocessing"]["resize"]["max_image_width"],
                                                                max_height=self.config["preprocessing"]["resize"]["max_image_height"])
                                    resized_image.save(resized_path)
                                    resized_end_time = time.time()
                                    file_data["timing_breakdown"]["preprocessing"]["resize"] += resized_end_time - resized_start_time

                                # Binarization.
                                if skip_binarization:
                                    binarized_path = os.path.join(binarized_dir, f"{page_id}.jpg")
                                    resized_image.save(binarized_path)
                            
                            if not skip_binarization:
                                binarize_start_time = time.time()
                                binarize_directory(resized_dir, binarized_dir, zigzag_jar_path)
                                binarize_end_time = time.time()
                                file_data["timing_breakdown"]["preprocessing"]["binarization"] += binarize_end_time - binarize_start_time

                        
                    else:           # only jpg, png beyond this point

                        image_pil = Image.open(file_path)

                        # Resizing.
                        resized_path = os.path.join(resized_dir, f"{work_id}.jpg")  # use work_id for single images
                        if skip_resize:
                            image_pil.save(resized_path)
                            resized_image = image_pil
                        else:
                            resized_start_time = time.time()
                            resized_image = resize_image(image_pil,
                                                        max_width=self.config["preprocessing"]["resize"]["max_image_width"],
                                                        max_height=self.config["preprocessing"]["resize"]["max_image_height"])
                            resized_image.save(resized_path)
                            resized_end_time = time.time()
                            file_data["timing_breakdown"]["preprocessing"]["resize"] += resized_end_time - resized_start_time

                       
                        # Binarization.
                        if skip_binarization:
                            binarized_path = os.path.join(binarized_dir, f"{work_id}.jpg")
                            resized_image.save(binarized_path)
                        else:
                            binarize_start_time = time.time()
                            binarize_directory(resized_dir, binarized_dir, zigzag_jar_path)
                            binarize_end_time = time.time()
                            file_data["timing_breakdown"]["preprocessing"]["binarization"] += binarize_end_time - binarize_start_time


                    # STEP 2 & 3: Metadata generation via resized images; OCR via resized & binarized images.
                    
                    if not skip_ocr:
                        if skip_resize and skip_binarization:
                            print(f"\nGenerating metadata and OCRing {file_name}, skipping preprocessing per config.\n")
                        else:
                            print(f"\nPreprocessing {file_name} complete. Generating metadata and OCRing...\n")
                    else:
                        if skip_resize and skip_binarization:
                            print(f"\nSkipping preprocessing and generating metadata, skipping OCR per config.\n")
                        else:
                            print(f"\nPreprocessing {file_name} complete. Generating metadata, skipping OCR per config...\n")

                    

                     # Check past work to recover any already processed pages.
                    if not skip_aggregation:
                        aggregated_metadata = {}
                        aggregation_cache = {}

                        for field in fields_to_aggregate:
                            aggregated_metadata[field] = {}     # dict of strings for concatenation at end

                    successfully_processed_pages = set()
                    metadata_cache = {}    # store already successfully created metadata (avoiding rerunning)
                    ocr_cache = {} 
                    if os.path.exists(doc_metadata_path):
                        print(f"\nFound existing metadata file for {file_name}. Resuming from where it left off...\n")
                        with open(doc_metadata_path, "r") as f:
                            for line in f:
                                entry = json.loads(line)

                                # Aggregation entry.
                                if "image_id" not in entry:     # if more than one, only the last (most recent) one is used
                                    if self._no_none_values(entry):     # None values signify error
                                        aggregation_cache = entry   
                                        continue

                                image_id = entry["image_id"]

                                # Cache successful metadata and OCR from previous runs.
                                if "error" not in entry and self._no_none_values(entry):
                                    metadata_cache[image_id] = entry    # may include "ocr" as field but will be overwritten by new OCR
                                
                                    # Add previously generated metadata for aggregation later.
                                    if not skip_aggregation:
                                        order = self._get_page_order(image_id)

                                        for field in fields_to_aggregate:
                                            if entry["metadata"][field] is not None and entry["metadata"][field].strip():
                                                aggregated_metadata[field][order] = entry["metadata"][field]

                                if "ocr" in entry and "error" not in entry["ocr"] and self._no_none_values(entry["ocr"]):
                                    ocr_cache[image_id] = entry["ocr"]

                                # Fully successful: can safely skip.
                                if "error" not in entry:
                                    if "ocr" in entry:
                                        if "error" in entry["ocr"]:
                                            continue
                                    if self._no_none_values(entry):
                                        successfully_processed_pages.add(image_id)

                                        file_data["timing_breakdown"]["ocr"] += entry["ocr"]["elapsed_time"]
                                        file_data["cost_breakdown"]["ocr"] += entry["ocr"]["cost"]
                                        file_data["timing_breakdown"]["metadata_generation"] += entry["elapsed_time"]
                                        file_data["cost_breakdown"]["metadata_generation"] += entry["cost"]


                    # Remaining pages to process.

                    # Assumes only pdf (potentially multiple pages), jpg, png files given. 
                    # Multiple-page files (via pdfs) have naming: {page_number}_{work_id}.jpg (guaranteed by internal counting process); jpg, png files have naming: {work_id}.jpg where {work_id}s are the basename of the file.
                    sorted_order = sorted(os.listdir(resized_dir), key=lambda x: int(x.split("_")[0])) if len(os.listdir(resized_dir)) > 1 else os.listdir(resized_dir)
                    remaining_pages_to_process = [page for page in sorted_order if os.path.splitext(page)[0] not in successfully_processed_pages]   # either erroneous or not yet processed

                    if not remaining_pages_to_process:
                        print(f"\nAll pages for {work_id} have already been processed.\n")
                        if not skip_aggregation:
                            print("\nSkipping to aggregation.\n")
                    else:
                        print(f"\n{len(remaining_pages_to_process)} page(s) remaining to process.\n")


                    # Generate metadata and OCR.
                    with open(doc_metadata_path, "a") as f:
                       
                        for page in remaining_pages_to_process:      
                            page_id = os.path.splitext(page)[0]    # page filename is already id
                            resized_page_path = os.path.join(resized_dir, page)
                            binarized_page_path = os.path.join(binarized_dir, page)

                            if page_id in metadata_cache:
                                page_metadata = metadata_cache[page_id]
                                file_data["cost_breakdown"]["metadata_generation"] += page_metadata["cost"]
                                file_data["timing_breakdown"]["metadata_generation"] += page_metadata["elapsed_time"]
                            else:
                                page_metadata = generate_single_metadata(
                                    model_name=self.config["metadata_generation"]["llm_model"],
                                    image_path=resized_page_path,
                                    json_schema=metadata_schema,
                                    system_prompt=self.config["metadata_generation"]["system_prompt"],
                                    user_prompt=self.config["metadata_generation"]["user_prompt"]
                                )

                                if "error" in page_metadata:
                                    current_error_count += 1

                                if "cost" in page_metadata:
                                    file_data["cost_breakdown"]["metadata_generation"] += page_metadata["cost"]
                                if "elapsed_time" in page_metadata:
                                    file_data["timing_breakdown"]["metadata_generation"] += page_metadata["elapsed_time"]

                            if not skip_ocr:

                                if page_id in ocr_cache:
                                    page_ocr = ocr_cache[page_id]
                                    file_data["cost_breakdown"]["ocr"] += page_ocr["cost"]
                                    file_data["timing_breakdown"]["ocr"] += page_ocr["elapsed_time"]
                                else:
                                    page_ocr = generate_single_ocr(
                                        model_name=self.config["ocr"]["llm_model"],
                                        image_path=binarized_page_path,
                                        system_prompt=self.config["ocr"]["system_prompt"],
                                        user_prompt=self.config["ocr"]["user_prompt"]
                                    )

                                    if "error" in page_ocr:
                                        current_error_count += 1

                                    if "cost" in page_ocr:
                                        file_data["cost_breakdown"]["ocr"] += page_ocr["cost"]
                                    if "elapsed_time" in page_ocr:
                                        file_data["timing_breakdown"]["ocr"] += page_ocr["elapsed_time"]

                                    if "id" in page_ocr:
                                        del page_ocr["id"]

                                page_metadata["ocr"] = page_ocr
                           
                            f.write(json.dumps(page_metadata, ensure_ascii=False) + "\n")
                            f.flush()

                            if not skip_aggregation:
                                for field in fields_to_aggregate:
                                    order = self._get_page_order(page)

                                    # "field" is a valid key in page_metadata["metadata"] as guaranteed by JSON Schema.
                                    if page_metadata["metadata"][field] is not None and page_metadata["metadata"][field].strip():
                                        aggregated_metadata[field][order] = page_metadata["metadata"][field]
    

                    # STEP 4: Metadata aggregation (represented at work-level).
                    if skip_aggregation:
                        if not skip_ocr:
                            print(f"\nMetadata generation and OCR complete for {file_name}, skipping metadata aggregation per config.\n")
                        else:
                            print(f"\nMetadata generation complete for {file_name}, skipping metadata aggregation per config.\n")
                    else:
                        if not skip_ocr:
                            print(f"\nMetadata generation and OCR complete for {file_name}. Aggregating requested metadata...\n")
                        else:
                            print(f"\nMetadata generation complete for {file_name}. Aggregating requested metadata...\n")
                            
                        if current_error_count > 0:    # previous error with LLM responses; no aggregation will be done
                            aggregation_to_save = {f"{work_id}_aggregated": False, "error": "Error occurred during metadata generation or OCR in one or more pages. Please run again to fix those error(s)."}
                        else:
                            aggregation_to_save = {f"{work_id}_aggregated": True} 
                            for field in fields_to_aggregate:

                                if field in aggregation_cache and "error" not in aggregation_cache[field]:
                                    summary_result = aggregation_cache[field]
                                    file_data["cost_breakdown"]["aggregation"] += summary_result.get("cost", 0)
                                    file_data["timing_breakdown"]["aggregation"] += summary_result.get("elapsed_time", 0)
                                else:
                                    # Skip aggregation if dict is empty.
                                    if not aggregated_metadata[field]:
                                        
                                        print(f"\n{work_id} - Skipping aggregation for field '{field}' (no data).\n")
                                        aggregation_to_save[field] = {"result": "No data available for aggregation."}
                                        continue

                                    # Sort aggregated data (may be out of order from correcting errors inprevious runs).
                                    sorted_keys = sorted(aggregated_metadata[field].keys(), key=int)
                                    concatenated_field_data = "\n\n".join([aggregated_metadata[field][key] for key in sorted_keys])

                                    summary_result = generate_single_aggregated_metadata(
                                        model_name=self.config["aggregation"]["llm_model"],
                                        concatenated_metadata=concatenated_field_data,
                                        system_prompt=self.config["aggregation"]["system_prompt"],
                                        user_prompt=self.config["aggregation"]["user_prompt"]
                                    )

                                    if "error" in summary_result:
                                        current_error_count += 1

                                    if "cost" in summary_result:
                                        file_data["cost_breakdown"]["aggregation"] += summary_result["cost"]
                                    if "elapsed_time" in summary_result:
                                        file_data["timing_breakdown"]["aggregation"] += summary_result["elapsed_time"]

                                aggregation_to_save[field] = summary_result
                            
                            if current_error_count > 0:     # somewhere in aggregation itself, error occurred
                                aggregation_to_save[f"{work_id}_aggregated"] = False   

                        with open(doc_metadata_path, "a") as f:
                            f.write(json.dumps(aggregation_to_save, ensure_ascii=False) + "\n")
                            f.flush()


                    # Add to manifest tally.
                    current_success_processed_file_count += 1 if current_error_count == 0 else 0
                    current_failed_processed_file_count += 1 if current_error_count > 0 else 0


                    file_data["processing_date"] = datetime.now().isoformat()
                    file_data["status"] = "success" if current_error_count == 0 else "failed"
                    file_data["total_processing_time"] = file_data["timing_breakdown"]["metadata_generation"] + file_data["timing_breakdown"]["ocr"] + file_data["timing_breakdown"]["aggregation"] + file_data["timing_breakdown"]["preprocessing"]["resize"] + file_data["timing_breakdown"]["preprocessing"]["binarization"]
                    file_data["total_cost"] = sum(file_data["cost_breakdown"].values())
                    file_data["processing_error_count"] += current_error_count

                    if current_error_count == 0:
                        print(f"\nSuccessfully processed {file_name}.\n")
                    else:
                        print(f"\nFully processed {file_name} but incomplete due to {current_error_count} non-client-side errors (e.g., server failure from LLM providers). Please rerun.\n")

                except Exception as e:      # log file processing as failed (something major went wrong; ie. setup error such that no files can be processed)
                    current_error_count += 1    # accounting for what caused the exception
                    current_failed_processed_file_count += 1

                    file_data["error"] = str(e)

                    file_data["processing_date"] = datetime.now().isoformat()
                    file_data["status"] = "failed"
                    file_data["total_processing_time"] = file_data["timing_breakdown"]["metadata_generation"] + file_data["timing_breakdown"]["ocr"] + file_data["timing_breakdown"]["aggregation"] + file_data["timing_breakdown"]["preprocessing"]["resize"] + file_data["timing_breakdown"]["preprocessing"]["binarization"]
                    file_data["total_cost"] = sum(file_data["cost_breakdown"].values())
                    file_data["processing_error_count"] += current_error_count     


                    print(f"\nFailed to process {file_name} fully.\n")


            # STEP 5: Finished a file, updating or creating manifest to log file processing completion.
            if os.path.exists(manifest_path):
                with open(manifest_path, "r") as f:
                    manifest = json.load(f)

                # Update.
                manifest["latest_processing_date"] = datetime.now().isoformat()
                manifest["total_files"] += 1 if file_name not in manifest["files"] else 0
                manifest["files"][file_name] = file_data

                failed_count = 0
                successful_count = 0
                for file_name, file_data in manifest["files"].items():
                    if file_data["status"] == "failed":
                        failed_count += 1
                    else:
                        successful_count += 1

                manifest["successful"] = successful_count
                manifest["failed"] = failed_count


                with open(manifest_path, "w") as f:     # overwriting existing to update
                    f.write(json.dumps(manifest, indent=4, ensure_ascii=False))
            else:       # no manifest found, create new one
                manifest = {
                    "latest_processing_date": datetime.now().isoformat(),
                    "schema_used": os.path.basename(self.config["metadata_generation"]["json_schema_path"]),
                    "run_config": {
                        "resize": not skip_resize,
                        "binarization": not skip_binarization,
                        "ocr": not skip_ocr,
                        "aggregation": not skip_aggregation
                    },
                    "total_files": 1,
                    "successful": 1 if current_error_count == 0 else 0,
                    "failed": 0 if current_error_count == 0 else 1,
                    "files": {file_name: file_data}
                }

                with open(manifest_path, "w") as f:
                    f.write(json.dumps(manifest, indent=4, ensure_ascii=False))

        print(f"\nMetaScribe pipeline complete. {current_success_processed_file_count} file(s) processed successfully, {current_failed_processed_file_count} file(s) failed to process.\n\nSee {manifest_path} for details.")

    def _get_page_order(self, image_id: str):
        """
        Get the order of a page from the image ID (internally named: {page_number}_{work_id}). 
        
        Returns 0 if the image ID is not in the correct format (happens for single jpg/png files).
        """
        try:
            return int(image_id.split("_")[0])
        except Exception as _:
            return 0

    def _pix_to_PIL(self, pix):
        """
        Convert a fitz.Pixmap to a PIL Image.

        Args:
            pix (fitz.Pixmap): The pixmap to convert.

        Returns:
            PIL.Image.Image: The converted image.
        """
        
        img_data = pix.samples
        img_mode = "RGB" if pix.n == 3 else "RGBA" if pix.n == 4 else "L"
        image_pil = Image.frombytes(img_mode, (pix.width, pix.height), img_data)

        return image_pil
        

    def _check_for_already_processed_files(self, files_to_process, manifest_path):
        """
        Check for already successfully processed files via manifest.json and return a list of files to process.

        Args:
            files_to_process (list): file list that are candidates for processing.
            manifest_path (str): path to manifest.json.
        Returns:
            list: files that still need processing
        """
        
        already_processed = set()

        if os.path.exists(manifest_path):
            try:
                with open(manifest_path, "r") as f:
                    manifest = json.load(f)
                    if "files" in manifest and isinstance(manifest["files"], dict):

                        # Extract already successfully processed files.
                        for file_name, file_entry in manifest["files"].items():
                            if isinstance(file_entry, dict) and file_entry["status"] == "success":
                                already_processed.add(file_name)

                        print(f"\nFound {len(already_processed)} file(s) that have already been successfully processed.\n")

            except Exception as e:
                print(f"\nError loading manifest from {manifest_path}: {e}\n\nLikely corrupted: please address this issue before running again.\n")
                return None


        filtered_files = [f for f in files_to_process if not f in already_processed]

         # Resume?
        if already_processed:
            while True:
                user_response = input(f"Resume processing {len(filtered_files)} more file(s) (y/n)? ")
                if user_response.lower() in ["y", "yes", "n", "no"]:
                    break
                else:
                    print("\nInvalid input. Please enter 'y' or 'n'.\n")

            if user_response.lower() in ("n", "no"):
                print(f"\nStopped MetaScribe pipeline.\n")
                return None
            else:
                if len(filtered_files) != 0:
                    print(f"\nResuming processing, skipping already processed files...\n")
            
        return filtered_files
        

    def _no_none_values(self, d):
        """
        Recursively checks if a dictionary contains any None values.
        
        Args:
            d: Dictionary to check, which may contain nested dictionaries
            
        Returns:
            bool: True if no None values exist, False if any None is found
        """
        if not isinstance(d, dict):
            return d is not None
            
        for key, value in d.items():
            if value is None:
                return False
            elif isinstance(value, dict):
                if not self._no_none_values(value):
                    return False
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        if not self._no_none_values(item):
                            return False
                    elif item is None:
                        return False
                        
        return True
    
        
        
        
