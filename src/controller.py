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
            
        os.makedirs(output_dir, exist_ok=True)
        metadata_dir = os.path.join(output_dir, "metadata")
        os.makedirs(metadata_dir, exist_ok=True)

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
            print(f"\nGiven {len(files_to_process)} valid files to process.\n")

        unaccepted_files = [
            f for f in os.listdir(input_dir)
            if os.path.isfile(os.path.join(input_dir, f)) and not f.lower().endswith(ACCEPTABLE_EXTENSIONS)
        ]
        
        if unaccepted_files:
            while True:
                example_files = ", ".join(unaccepted_files[:3]) + ("..." if len(unaccepted_files) > 3 else "")
                user_response = input(f"WARNING: Detected {len(unaccepted_files)} file(s) with unsupported extensions in input directory ({example_files}). Must be one of {ACCEPTABLE_EXTENSIONS}.\nContinue? (y/n): ")
                if user_response.lower() in ["y", "yes"]:
                    break
                elif user_response.lower() in ["n", "no"]:
                    print("MetaScribe pipeline aborted by user (unsupported file extensions).")
                    return
                else:
                    print("\nInvalid input. Please enter 'y' or 'n'.\n")
            
        
        # Check for exisiting processed files.
        files_to_process = self._check_for_already_processed_files(output_dir, files_to_process)
        if files_to_process is None:    # user stopped pipeline
            return
        elif not files_to_process:
            print("No new files to process. MetaScribe pipeline complete.")    # everything already processed
            return
        
        
        # Preprocess files serially.
        files_data = []
        success_processed_count = 0
        failed_processed_count = 0
        for file_name in files_to_process:
            total_cost = 0
            start_time = time.time()
            successfully_processed = False
            small_error_flag = False     # True if any small error occurs (API call failure, etc.), but will continue processing.

            print(f"\nProcessing {file_name}\n")
            file_path = os.path.join(input_dir, file_name)
            file_extension = os.path.splitext(file_name)[1].lower()
            work_id = os.path.splitext(file_name)[0]    # use basename as ID for naming

            # Temporary directory to hold temp files while processing.
            with tempfile.TemporaryDirectory() as temp_dir:
                try:
                    # Subdirectories for images after pre-processing.
                    # STEP 1: Preprocessing -- resizing & binarization.
                    resized_dir = os.path.join(temp_dir, "resized")
                    binarized_dir = os.path.join(temp_dir, "resized_binarized")
                    os.makedirs(resized_dir)
                    os.makedirs(binarized_dir)

                    # Check for skips.
                    skip_resize = self.config.get("preprocessing", {}).get("resize", {}).get("skip", False)
                    skip_binarization = self.config.get("preprocessing", {}).get("binarization", {}).get("skip", False)

                    # Resolve ZigZag.jar path.
                    project_core = os.path.dirname(os.path.abspath(__file__))
                    zigzag_jar_path = os.path.join(project_core, self.config["preprocessing"]["binarization"]["jar_path"])
                    
                    # PDF files.
                    if file_extension == ".pdf":
                        
                        pdf_doc = fitz.open(file_path)
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
                                resized_image = resize_image(image_pil,
                                                            max_width=self.config["preprocessing"]["resize"]["max_image_width"],
                                                            max_height=self.config["preprocessing"]["resize"]["max_image_height"])
                                resized_image.save(resized_path)
                            
                            # Binarization.
                            if skip_binarization:
                                binarized_path = os.path.join(binarized_dir, f"{page_id}.jpg")
                                resized_image.save(binarized_path)
                        
                        if not skip_binarization:
                            binarize_directory(resized_dir, binarized_dir, zigzag_jar_path)

                        pdf_doc.close()
                        
                    else:           # only jpg, png beyond this point

                        image_pil = Image.open(file_path)

                        # Resizing.
                        resized_path = os.path.join(resized_dir, f"{work_id}.jpg")  # use work_id for single images
                        if skip_resize:
                            image_pil.save(resized_path)
                            resized_image = image_pil
                        else:
                            resized_image = resize_image(image_pil,
                                                        max_width=self.config["preprocessing"]["resize"]["max_image_width"],
                                                        max_height=self.config["preprocessing"]["resize"]["max_image_height"])
                            resized_image.save(resized_path)

                       
                        # Binarization.
                        if skip_binarization:
                            binarized_path = os.path.join(binarized_dir, f"{work_id}.jpg")
                            resized_image.save(binarized_path)
                        else:
                            binarize_directory(resized_dir, binarized_dir, zigzag_jar_path)


                    # STEP 2 & 3: Metadata generation via resized images; OCR via resized & binarized images.
                    skip_ocr = self.config.get("ocr", {}).get("skip", False)
                    
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

                    
                    doc_metadata_path = os.path.join(metadata_dir, f"{work_id}.jsonl")
                    metadata_schema = json.load(open(self.config["metadata_generation"]["json_schema_path"]))

                    skip_aggregation = self.config.get("aggregation", {}).get("skip", False)
                    if not skip_aggregation:
                        aggregated_metadata = {}
                        fields_to_aggregate = self.config["aggregation"]["included_fields"]

                        for field in fields_to_aggregate:
                            if field not in metadata_schema["properties"]:
                                print(f"Stopping MetaScribe: requested field '{field}' to aggregate not found in metadata schema. Ensure that this field is present in the JSON schema under 'properties'.") 
                                return
                            aggregated_metadata[field] = {}     # dict of strings for concatenation at end

                    
                    processed_pages = set()
                    if os.path.exists(doc_metadata_path):
                        print(f"\nFound existing metadata file for {file_name}. Resuming from where it left off...\n")
                        with open(doc_metadata_path, "r") as f:
                            for line in f:
                                entry = json.loads(line)

                                if "image_id" in entry and "error" not in entry:
                                    if "ocr" in entry:
                                        if "error" in entry["ocr"]:
                                            continue
                                    processed_pages.add(entry["image_id"])

                                    # Add previously generated metadata for aggregation later.
                                    if not skip_aggregation:
                                        try:
                                            order = int(entry["image_id"].split("_")[0])
                                        except (ValueError, IndexError):    # likely from single jpg/png file without page number id at filename start
                                            order = 0

                                        for field in fields_to_aggregate:
                                            if entry["metadata"][field] is not None and entry["metadata"][field].strip():
                                                aggregated_metadata[field][order] = entry["metadata"][field]

                    
                    # Remaining pages to process.
                    sorted_order = sorted(os.listdir(resized_dir), key=lambda x: int(x.split("_")[0]))
                    remaining_pages_to_process = [page for page in sorted_order if os.path.splitext(page)[0] not in processed_pages]

                    if not remaining_pages_to_process:
                        print(f"\nAll pages for {work_id} have already been processed. ")
                        if not skip_aggregation:
                            print("Skipping to aggregation.")
                        sorted_order =[]
                    else:
                        print(f"\n{len(remaining_pages_to_process)} page(s) remaining to process.\n")
                        sorted_order = remaining_pages_to_process

    
                    # Generate metadata and OCR.
                    with open(doc_metadata_path, "a") as f:
                       
                        for page in sorted_order:       # page filename is already id
                            
                            resized_page_path = os.path.join(resized_dir, page)
                            binarized_page_path = os.path.join(binarized_dir, page)

                            page_metadata = generate_single_metadata(
                                model_name=self.config["metadata_generation"]["llm_model"],
                                image_path=resized_page_path,
                                json_schema=metadata_schema,
                                system_prompt=self.config["metadata_generation"]["system_prompt"],
                                user_prompt=self.config["metadata_generation"]["user_prompt"]
                            )

                            if "error" in page_metadata:
                                small_error_flag = True

                            if "cost" in page_metadata:
                                total_cost += page_metadata["cost"]

                            if not skip_ocr:
                                page_ocr = generate_single_ocr(
                                    model_name=self.config["ocr"]["model"],
                                    image_path=binarized_page_path,
                                    system_prompt=self.config["ocr"]["system_prompt"],
                                    user_prompt=self.config["ocr"]["user_prompt"]
                                )

                                if "error" in page_ocr:
                                    small_error_flag = True

                                if "cost" in page_ocr:
                                    total_cost += page_ocr["cost"]

                                if "id" in page_ocr:
                                    del page_ocr["id"]

                                page_metadata["ocr"] = page_ocr
                           
                            f.write(json.dumps(page_metadata) + "\n")
                            f.flush()

                            if not skip_aggregation:
                                order = int(page.split("_")[0])

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
                            
                        if small_error_flag:    # previous error with LLM responses; no aggregation will be done
                            aggregation_to_save = {f"{work_id}_aggregated": False, "error": "Error occurred during metadata generation or OCR in one or more pages. Please try again after fixing the errors."}
                        else:
                            aggregation_to_save = {f"{work_id}_aggregated": True} 
                            for field in fields_to_aggregate:

                                # Skip aggregation if dict is empty.
                                if not aggregated_metadata[field]:
                                    print(f"\n{work_id} - Skipping aggregation for field '{field}' (no data).\n")
                                    aggregation_to_save[field] = {"result": "No data available for aggregation."}
                                    continue

                                # Sort aggregated data (may be out of order from errors inprevious runs).
                                sorted_keys = sorted(aggregated_metadata[field].keys(), key=int)
                                concatenated_field_data = "\n\n".join([aggregated_metadata[field][key] for key in sorted_keys])

                                summary_result = generate_single_aggregated_metadata(
                                    model_name=self.config["aggregation"]["llm_model"],
                                    concatenated_metadata=concatenated_field_data,
                                    system_prompt=self.config["aggregation"]["system_prompt"],
                                    user_prompt=self.config["aggregation"]["user_prompt"]
                                )

                                if "error" in summary_result:
                                    small_error_flag = True

                                if "cost" in summary_result:
                                    total_cost += summary_result["cost"]

                                aggregation_to_save[field] = summary_result
                            
                            if small_error_flag:
                                aggregation_to_save[f"{work_id}_aggregated"] = False    # somewhere in aggregation, error occurred

                        with open(doc_metadata_path, "a") as f:
                            f.write(json.dumps(aggregation_to_save) + "\n")
                            f.flush()


                    # Add to manifest tally.
                    successfully_processed = True
                    success_processed_count += 1

                    end_time = time.time()
                    total_elapsed_time = end_time - start_time

                    files_data = [{
                        "processing_date": datetime.now().isoformat(),
                        "original_file": file_name,
                        "metadata_file": os.path.basename(doc_metadata_path),
                        "status": "success" if not small_error_flag else "failed",
                        "total_processing_time": total_elapsed_time,
                        "total_cost": total_cost
                    }]
                    print(f"\nSuccessfully processed {file_name}.\n")

                except Exception as e:      # log file processing as failed (something major went wrong; ie. setup error such that no files can be processed)
                    successfully_processed = False
                    failed_processed_count += 1

                    files_data = [{
                        "processing_date": datetime.now().isoformat(),
                        "original_file": file_name,
                        "status": "failed",
                        "error": str(e),
                        "time_spent_before_failure": total_elapsed_time,
                        "cost_before_failure": total_cost
                    }]
                    print(f"\nFailed to process {file_name}.\n")
                    continue

            # STEP 5: Finish, updating or creating manifest to log file processing completion.
            manifest_path = os.path.join(output_dir, "manifest.json")

            if os.path.exists(manifest_path):
                with open(manifest_path, "r") as f:
                    manifest = json.load(f)

                # Update.
                manifest["latest_processing_date"] = datetime.now().isoformat()
                manifest["total_files"] += 1
                if successfully_processed:
                    manifest["successful"] += 1
                else:
                    manifest["failed"] += 1
                manifest["files"].extend(files_data)

                with open(manifest_path, "w") as f:     # overwriting existing to update
                    f.write(json.dumps(manifest, indent=4))
            else:
                manifest = {
                    "latest_processing_date": datetime.now().isoformat(),
                    "total_files": 1,
                    "successful": 1 if successfully_processed else 0,
                    "failed": 0 if successfully_processed else 1,
                    "files": files_data
                }

                with open(manifest_path, "w") as f:
                    f.write(json.dumps(manifest, indent=4))

        print(f"\nMetaScribe pipeline complete. {success_processed_count} files processed successfully, {failed_processed_count} files failed to process.\n\nSee {manifest_path} for details.")


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
        

    def _check_for_already_processed_files(self, output_dir, files_to_process):
        """
        Check for already successfully processed files via manifest.json and return a list of files to process.

        Args:
            output_dir (str): The directory to save the processed documents.
            files_to_process (list): file list that are candidates for processing.

        Returns:
            list: files that still need processing
        """
        
        manifest_path = os.path.join(output_dir, "manifest.json")
        already_processed = set()

        if os.path.exists(manifest_path):
            try:
                with open(manifest_path, "r") as f:
                    manifest = json.load(f)
                    if "files" in manifest and isinstance(manifest["files"], list):

                        # Extract already successfully processed files.
                        for file_entry in manifest["files"]:
                            if isinstance(file_entry, dict) and "original_file" in file_entry and file_entry["status"] == "success":
                                already_processed.add(file_entry["original_file"])

                        print(f"\nFound {len(already_processed)} files that have already been successfully processed.\n")

            except Exception as e:
                print(f"\nError loading manifest from {manifest_path}: {e}\n\nWill process all {len(files_to_process)} files.\n")
                already_processed = set()


        filtered_files = [f for f in files_to_process if not f in already_processed]

         # Resume or start fresh?
        if already_processed:
            while True:
                user_response = input(f"Resume processing (y/n)? ")
                if user_response.lower() in ["y", "yes", "n", "no"]:
                    break
                else:
                    print("\nInvalid input. Please enter 'y' or 'n'.\n")

            if user_response.lower() in ("n", "no"):
                print(f"\nStopped MetaScribe pipeline.\n")
                return None
            else:
                if len(filtered_files) != 0:
                    print(f"\nResuming processing ({len(filtered_files)} files), skipping already processed files...\n")
            
        return filtered_files
        
        
        
        
        
