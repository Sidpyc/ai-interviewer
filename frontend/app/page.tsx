'use client'
import { useState, useEffect } from "react"
import React from 'react'

const page = () => {
//Connectivity Testing
const [backendMessage , setBackendMessage] = useState<string>('Loading Message')
const [errorMessage, setErrorMessage] = useState<string | null>(null);
//File Related
const [selectedFile, setSelectedFile] = useState<File | null>(null); //To set the file
const [uploadStatus, setUploadStatus] = useState<string>('') //To upload via post api
const [resumeText, setResumeText] = useState<String>('')// Return response of Resume Text 

// Testing Connection 
useEffect(()=> {
  const fetchMessage = async () => {
    try {
      const response = await fetch('http://localhost:8000/api/data/') //fetch the actual response from backend
      if (!response.ok){
        throw new Error(`HTTP error! status ${response.status}`)  //if response not okay , return the response status as new Error
      }
      const data = await response.json(); //if response recieved, wait until its converted to json
      setBackendMessage(data.message) // fetch the message object 
    }
    catch (e:any){
      console.error("failed")
      setErrorMessage(`${e.message}`)
      setBackendMessage("Error Fetching Data"); // set error in case of faults
    }
  }
  fetchMessage();
}, []);

// File handling Code
const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) =>
{

  if (event.target.files && event.target.files.length>0){ // check if file name is non empty and file exists
    setSelectedFile(event.target.files[0]) // set the first selected file
    setUploadStatus('')//clear prev status
    setResumeText('')// clear prev text

  } else {
    setSelectedFile(null) // default back to null
  }
}

//File Upload Code
const handleFileUpload = async () =>{
  if (!selectedFile) {
    setUploadStatus('Please select a file first')
    return;
  }
  setUploadStatus('Uploading...');
  setErrorMessage(null);

  const formData = new FormData();
  formData.append('file',selectedFile) // set the resume file as new formdata object

  try{
    const response = await fetch('http://localhost:8000/upload-resume/',{
      method : 'POST',
      body : formData
    }) //send the resume over post api
    if (!response.ok){
      const errorData = await response.json();
      throw new Error(`${response.status} : ${errorData.detail}`)
    }
    const result = await response.json();
    setUploadStatus('Upload Successful');//fetch the return response
    setResumeText(result.extracted_text || 'No Text Extracted')//fetch return text
    console.log('Upload Result', result)
  
  }catch (e:any){
    console.error(e)
    setUploadStatus(`Failed ${e.message} `)
    setErrorMessage(e.message)
  }
  
}


  return (
    <>
    {/* Backend Connection Test */}
    <h1><center>AI Interviewer</center></h1>
    <h2><center>Backend Test</center></h2>
    {
      errorMessage?(
        <p>Error: {errorMessage}</p>
      ):
      <p>Message:{backendMessage}</p>
    }
    {/* Resume Upload */}
    <section>
      <h2>Upload Resume</h2>
      <input
       type="file"
       accept=".pdf , .docx"
       onChange={handleFileChange} 
       />
       <button
       onClick={handleFileUpload}
       disabled = {!selectedFile}
       >
        Upload Resume
       </button>

       {uploadStatus && (
        <p>
        {uploadStatus}
        </p>
       )}

       {resumeText && (
        <p>
          {resumeText}
        </p>
       )}
    </section>

    </>
  )
}

export default page