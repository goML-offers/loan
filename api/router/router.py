import sys
import aiofiles
import pandas as pd
from fastapi import FastAPI, File,  UploadFile
from typing import List
from fastapi import APIRouter
from fastapi import APIRouter, HTTPException, Response

from services.data_generation import read_csv_header
from services.accuracy import train_loan_approval_model, predict_loan_approval
import os
from fastapi.responses import JSONResponse

import json
from dotenv import load_dotenv
load_dotenv()



from logger import setup_logger
logger = setup_logger(f"log_data.log")


router = APIRouter()
@router.post('/goml/LLM marketplace/finance_data_generator/upload_file', status_code=201)
def data_generator(file: UploadFile):
    try:
        UPLOAD_DIR = "/api/uploads"

        if not os.path.exists(UPLOAD_DIR):
            os.makedirs(UPLOAD_DIR)
        # Generate a unique file name to avoid overwriting existing files
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        
        with open(file_path, "wb") as f:
            f.write(file.file.read())
            f.close()
        
        csv_data = read_csv_header(file_path)
    

        # Set the response headers for CSV file download
        response = Response(content=csv_data)
        response.headers["Content-Disposition"] = 'attachment; filename="data.csv"'
        response.headers["Content-Type"] = "text/csv"
        os.remove(file_path)
        return response
    except Exception as e:
            logger.error(str(e))
            raise HTTPException(status_code=400, detail=str(e))
   
@router.post('/goml/LLM marketplace/finance_data_generator/accuracy_generator', status_code=201)
async def accuracy_generator(original_file: UploadFile, synthesized_file: UploadFile):
    try:
        upload_dir = "api/uploads/"
        original_file_path = os.path.join(upload_dir, original_file.filename)
        synthesized_file_path = os.path.join(upload_dir, synthesized_file.filename)

        # Use 'await' when reading the file contents
        async with aiofiles.open(original_file_path, "wb") as f:
            await f.write(await original_file.read())



        async with aiofiles.open(synthesized_file_path, "wb") as f:
            await f.write(await synthesized_file.read())

        # Print the synthesized file path
        print(f"Synthesized file path: {synthesized_file_path}")

        # Read the CSV files into DataFrames
        df_original = pd.read_csv(original_file_path)
        df_synthesized = pd.read_csv(synthesized_file_path)

        # Merge the DataFrames using an appropriate merging strategy
        merged_df = pd.concat([df_original, df_synthesized], axis=0, ignore_index=True)

        # Define the file path where you want to save the merged CSV
        merged_file_path = 'api/uploads/synthesized_file.csv'

        # Write the merged DataFrame to a new CSV file
        merged_df.to_csv(merged_file_path, index=False)
        without_synthesized_accuracy = train_loan_approval_model(original_file_path)
        synthesized_accuracy = train_loan_approval_model(merged_file_path)
        os.remove(original_file_path)
        os.remove(merged_file_path)
        # push_to_s3(merged_file_path)
        os.remove(synthesized_file_path)
        return {
            "synthesized_accuracy": synthesized_accuracy,
            "without_synthesized_accuracy": without_synthesized_accuracy
        }
    except Exception as e:
            logger.error(str(e))
            raise HTTPException(status_code=400, detail=str(e))
            return 


@router.post('/goml/LLM marketplace/finance_data_generator/validating_test/', status_code=201)
def validating_test(file: UploadFile = File(...)):
   
    
    UPLOAD_DIR = "/api/uploads/"

    if not os.path.exists(UPLOAD_DIR):
        os.makedirs(UPLOAD_DIR)
    
    # Generate a unique file name to avoid overwriting existing files
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    
    with open(file_path, "wb") as f:
        f.write(file.file.read())
    print(file_path)
    predicted_data = predict_loan_approval(file_path)
    
    # Perform any additional processing or response logic here
    
    return {"predicted_data": predicted_data}
    # except Exception as e:
    #     logger.error(str(e))
    #     # raise HTTPException(status_code=500, detail="Internal server error")
    #     return e