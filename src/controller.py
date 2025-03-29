"""
controller.py

Main controller for MetaScribe pipeline.
"""

import yaml
import os
import json
import tempfile
import fitz
from datetime import datetime
from PIL import Image

from core.generate_metadata import generate_single_metadata
from core.ocr import generate_single_ocr
from core.preprocessing_document import resize_image, binarize_image
from core.aggregation import generate_single_aggregated_metadata


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
                        print(f"Successfully loaded configuration from {loc}\n\n")
                        break
                except Exception as e:
                    print(f"Error loading configuration from {loc}: {e}\n\n")
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

        print(f"Starting MetaScribe pipeline for {input_dir}...\n\n")

        # Set up output directory.
        if output_dir is None:
            output_dir = os.path.join(os.getcwd(), "metascribe_output")
            print(f"No output directory provided, using default: {output_dir}\n\n")

        if os.path.exists(output_dir):
            user_response = input(f"WARNING: Output directory {output_dir} already exists. Continue? (y/n): ")
            if user_response.lower() not in ["y", "yes"]:
                print("MetaScribe pipeline aborted by user (existing output directory).")
                return
            
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
            print(f"Given {len(files_to_process)} files to process.\n\n")

        unaccepted_files = [
            f for f in os.listdir(input_dir)
            if os.path.isfile(os.path.join(input_dir, f)) and not f.lower().endswith(ACCEPTABLE_EXTENSIONS)
        ]
        
        if unaccepted_files:
            user_response = input(f"WARNING: Found {len(unaccepted_files)} files with unsupported extensions. Must be one of {ACCEPTABLE_EXTENSIONS}.\nContinue? (y/n): ")
            if user_response.lower() not in ["y", "yes"]:
                print("MetaScribe pipeline aborted by user (unsupported file extensions).")
                return
            
        
        # Check for exisiting processed files.
        files_to_process = self._check_for_already_processed_files(output_dir, files_to_process)
        if not files_to_process:
            print("No new files to process. MetaScribe pipeline complete.")
            return
        
        
        # Preprocess files serially.
        success_processed_count = 0
        failed_processed_count = 0
        files_data = []

        for file_name in files_to_process:
            total_cost = 0
            total_elapsed_time = 0

            print(f"Processing {file_name}\n\n")
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

                    os.makedirs(resized_dir, exist_ok=True)
                    os.makedirs(binarized_dir, exist_ok=True)

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
                            resized_image = resize_image(image_pil,
                                                         max_width=self.config["preprocessing"]["resize"]["max_image_width"],
                                                         max_height=self.config["preprocessing"]["resize"]["max_image_height"])
                            resized_image.save(resized_path)
                            
                            # Binarization.
                            binarized_path = os.path.join(binarized_dir, f"{page_id}.jpg")
                            binarized_image = binarize_image(
                                resized_image,
                                jar_path=zigzag_jar_path
                            )
                      
                            binarized_image.save(binarized_path)
         
                        pdf_doc.close()
                        
                    else:           # only jpg, png beyond this point

                        image_pil = Image.open(file_path)

                        # Resizing.
                        resized_path = os.path.join(resized_dir, f"{work_id}.jpg")  # use work_id for single images
                        resized_image = resize_image(image_pil,
                                                     max_width=self.config["preprocessing"]["resize"]["max_image_width"],
                                                     max_height=self.config["preprocessing"]["resize"]["max_image_height"])
                        resized_image.save(resized_path)

                        # Binarization.
                        binarized_path = os.path.join(binarized_dir, f"{work_id}.jpg")
                        binarized_image = binarize_image(
                            resized_image,
                            jar_path=zigzag_jar_path
                        )
                        binarized_image.save(binarized_path)


                    # STEP 2 & 3: Metadata generation via resized images; OCR via resized & binarized images.
                    print(f"Preprocessing {file_name} complete. Generating metadata and OCRing...\n\n")
                    doc_metadata_path = os.path.join(metadata_dir, f"{work_id}.jsonl")
                    # TODO: will need to check if this already exists, if so, see WHERE to pick up and start appending (make a note of this).
                    # If exists, get pages that had been processed --> what more to process? Modify line 228

                    # Aggregation setup.
                    aggregated_metadata = {}
                    fields_to_aggregate = self.config["aggregation"]["included_fields"]

                    for field in fields_to_aggregate:
                        aggregated_metadata[field] = []

                    # Generate metadata and OCR.
                    with open(doc_metadata_path, "a") as f:
                        sorted_order = sorted(os.listdir(resized_dir), key=lambda x: int(x.split("_")[0]))
                       
                        for page in sorted_order:       # page filename is already id
                            
                            resized_page_path = os.path.join(resized_dir, page)
                            binarized_page_path = os.path.join(binarized_dir, page)

                            metadata_schema = json.load(open(self.config["metadata_generation"]["json_schema_path"]))

                            page_metadata = generate_single_metadata(
                                model_name=self.config["metadata_generation"]["llm_model"],
                                image_path=resized_page_path,
                                json_schema=metadata_schema,
                                system_prompt=self.config["metadata_generation"]["system_prompt"],
                                user_prompt=self.config["metadata_generation"]["user_prompt"]
                            )

                            total_cost += page_metadata["cost"]
                            total_elapsed_time += page_metadata["elapsed_time"]

                            page_ocr = generate_single_ocr(
                                model_name=self.config["ocr"]["model"],
                                image_path=binarized_page_path,
                                system_prompt=self.config["ocr"]["system_prompt"],
                                user_prompt=self.config["ocr"]["user_prompt"]
                            )

                            total_cost += page_ocr["cost"]
                            total_elapsed_time += page_ocr["elapsed_time"]

                            if "id" in page_ocr:
                                del page_ocr["id"]

                            page_metadata["ocr"] = page_ocr
                            f.write(json.dumps(page_metadata) + "\n")
                            f.flush()

                            for field in fields_to_aggregate:
                                if page_metadata["metadata"][field] is not None and page_metadata["metadata"][field].strip():
                                    aggregated_metadata[field].append(page_metadata["metadata"][field])
    
                    # STEP 4: Metadata aggregation (represented at work-level).
                    print(f"Metadata generation and OCR complete for {file_name}. Aggregating metadata...\n\n")
                    aggregation_to_save = {f"{work_id}_aggregated": True}   # to flag aggregated metadata in jsonl
                    for field in fields_to_aggregate:

                        # Skip aggregation if list is empty.
                        if not aggregated_metadata[field]:
                            print(f"{work_id} - Skipping aggregation for {field} (no data).")
                            aggregation_to_save[field] = {"result": "No data available for aggregation."}
                            continue

                        concatenated_field_data = "\n\n".join(aggregated_metadata[field])

                        summary_result = generate_single_aggregated_metadata(
                            model_name=self.config["aggregation"]["llm_model"],
                            concatenated_metadata=concatenated_field_data,
                            system_prompt=self.config["aggregation"]["system_prompt"],
                            user_prompt=self.config["aggregation"]["user_prompt"]
                        )

                        total_cost += summary_result["cost"]
                        total_elapsed_time += summary_result["elapsed_time"]

                        aggregation_to_save[field] = summary_result

                    with open(doc_metadata_path, "a") as f:
                        f.write(json.dumps(aggregation_to_save) + "\n")
                        f.flush()

                    # Add to manifest tally.
                    success_processed_count += 1    

                    files_data.append({
                        "processing_date": datetime.now().isoformat(),
                        "original_file": file_name,
                        "metadata_file": os.path.basename(doc_metadata_path),
                        "status": "success",
                        "total_processing_time": total_elapsed_time,
                        "total_cost": total_cost
                    })
                    print(f"Successfully processed {file_name}.\n\n")

                except Exception as e:      # log file processing as failed
                    failed_processed_count += 1

                    files_data.append({
                        "processing_date": datetime.now().isoformat(),
                        "original_file": file_name,
                        "status": "failed",
                        "error": str(e)
                    })
                    continue

        # STEP 5: Finish, updating or creating full run's manifest.
        # TODO: add to manifest after each file is processed (not all at end here)... need to move this up into loop above.
        manifest_path = os.path.join(output_dir, "manifest.json")

        if os.path.exists(manifest_path):
            with open(manifest_path, "r") as f:
                manifest = json.load(f)

            # Update.
            manifest["latest_processing_date"] = datetime.now().isoformat()
            manifest["total_files"] += len(files_to_process)
            manifest["successful"] += success_processed_count
            manifest["failed"] += failed_processed_count
            manifest["files"].extend(files_data)

            with open(manifest_path, "w") as f:
                f.write(json.dumps(manifest, indent=4))
        else:
            manifest = {
                "latest_processing_date": datetime.now().isoformat(),
                "total_files": len(files_to_process),
                "successful": success_processed_count,
                "failed": failed_processed_count,
                "files": files_data
            }

            with open(manifest_path, "w") as f:
                f.write(json.dumps(manifest, indent=4))

        print(f"MetaScribe pipeline complete. {success_processed_count} files processed successfully, {failed_processed_count} files failed to process.\n\nSee {manifest_path} for details.")


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
        Check for already processed files and return a list of files that have not been processed.

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

                        # Extract already processed files.
                        for file_entry in manifest["files"]:
                            if isinstance(file_entry, dict) and "original_file" in file_entry:
                                already_processed.add(file_entry["original_file"])

                        print(f"Found {len(already_processed)} files that have already been processed.\n\n")

            except Exception as e:
                print(f"Error loading manifest from {manifest_path}: {e}\n\nWill process all {len(files_to_process)} files.\n\n")
                already_processed = set()

         # Resume or start fresh?
        if already_processed:
            user_response = input(f"Resume processing (y) or start fresh (n)? ")
            if user_response.lower() in ("n", "no"):
                already_processed = set()
                print(f"Starting fresh... reprocessing all {len(files_to_process)} files.\n\n")
            else:
                print(f"Resuming processing ({len(filtered_files)} files), skipping already processed files.\n\n")

        filtered_files = [f for f in files_to_process if not f in already_processed]

        return filtered_files
        
        
        
        
        
