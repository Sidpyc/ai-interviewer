'use client'
import { useState, useEffect } from "react"
import React from 'react'

const page = () => {

const [backendMessage , setBackendMessage] = useState<string>('Loading Message')
const [errorMessage, setErrorMessage] = useState<string | null>(null);

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

  return (
    <>
    <h1>AI Interviewer</h1>
    <h2>Backend Test</h2>
    {
      errorMessage?(
        <p>Error: {errorMessage}</p>
      ):
      <p>Message:{backendMessage}</p>
    }
    </>
  )
}

export default page