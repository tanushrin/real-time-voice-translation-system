import './App.css';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faMicrophone } from '@fortawesome/free-solid-svg-icons';
import { useState, useRef } from 'react';
import axios from "axios";

export default function App() {
    const [isRecording, setIsRecording] = useState(false);
    const mediaRecorderRef = useRef(null);
    const audioChunksRef = useRef([]);
    const [originalLanguage, setOriginalLanguage] = useState('english');
    const [toTranslateLanguage, setToTranslateLanguage] = useState('bengali');
    const audioRef = useRef(null);

    const handleOriginalLanguageChange = (event) => {
        setOriginalLanguage(event.target.value);
    };


    const handleToTranslateLanguageChange = (event) => {
        setToTranslateLanguage(event.target.value);
    };
    const handleToggleRecording = async () => {
        try {
            if (!isRecording) {
                const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                mediaRecorderRef.current = new MediaRecorder(stream);
                mediaRecorderRef.current.ondataavailable = (event) => {
                    audioChunksRef.current.push(event.data);
                };
                mediaRecorderRef.current.onstop = async () => {
                    const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/wav' });

                    try {
                        await callAPI(originalLanguage, toTranslateLanguage, audioBlob);
                    } catch (error) {
                        console.error('Error calling API:', error);
                    }
                    audioChunksRef.current = [];
                };

                mediaRecorderRef.current.start();
                setIsRecording(true);
            } else {
                mediaRecorderRef.current.stop();
                setIsRecording(false);
            }
        } catch (error) {
            console.error('Error accessing the microphone:', error);
        }
    };

    async function callAPI(originalLanguage, toTranslateLanguage, audioFile) {
        const formData = new FormData();
        formData.append('audio_file', audioFile);

        const queryParams = new URLSearchParams();
        queryParams.append('original_language', originalLanguage);
        queryParams.append('to_translate_language', toTranslateLanguage);

        await axios.post(`http://localhost:8000/predict?${queryParams.toString()}`, formData, {
            headers: {
                'Content-Type': 'multipart/form-data',
            },
        });
    }


    return (
        <>
            <div className="hidden">
                <audio ref={audioRef} autoPlay controls>.
                </audio>
            </div>
            <div className="container">
                <h2>Speech to Speech Converter</h2>
                <br/>
                <p>A tool that converts a bengali / english speech to an english / bengali speech in real time</p>

                <div className="main-contents">
                    <select
                        id="selector1"
                        className="mb-6 ml-32"
                        onChange={handleOriginalLanguageChange}
                        value={originalLanguage}
                    >
                        <option value="english">English</option>
                        <option value="bengali">Bengali</option>
                    </select>

                    <select
                        id="selector2"
                        className="mb-6 mr-32"
                        onChange={handleToTranslateLanguageChange}
                        value={toTranslateLanguage}
                    >
                        <option value="english">English</option>
                        <option value="bengali">Bengali</option>
                    </select>
                </div>

                <div className="main-contents">
                    <textarea disabled className="main-content">
                    </textarea>
                    <div className="m-10 cursor-pointer" onClick={handleToggleRecording}>
                        <FontAwesomeIcon
                            icon={faMicrophone}
                            className={`text-${isRecording ? 'red' : 'blue'}-500 text-4xl`}
                        />
                    </div>
                    <textarea disabled className="main-content">
                    </textarea>
                </div>
            </div>
        </>
    );
}
