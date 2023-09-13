from typing import Optional, Dict
from fastapi import FastAPI, File, Form, UploadFile
from pydantic import BaseModel, Field

# class FileUpload(BaseModel):
#     email: str
#     file: UploadFile

class FileUpload(BaseModel):
    company_name : str
    emails: list[str] = Field(..., example=["user1@example.com", "user2@example.com"])
    files: list[str] = Field(..., example=["path/to/file1.txt", "path/to/file2.txt"])

class EmailFile(BaseModel):
    emails: list[str] = Field(..., example=["user1@example.com", "user2@example.com"])
    files: list[str] = Field(..., example=["path/to/file1.txt", "path/to/file2.txt"])
 
class SendEmail(BaseModel):
    emails: list[str] = Field(..., example=["user1@example.com", "user2@example.com"])

class FilePath(BaseModel):
    file_path: list

class Prompt(BaseModel):
    prompt:list


class Send_Email(BaseModel):
    file: list[str]
    email: list[str]
    subject: str
    body: str
