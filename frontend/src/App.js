import './App.css';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faMicrophone } from '@fortawesome/free-solid-svg-icons';
import { useState, useRef } from 'react';

export default function App() {
    const [isRecording, setIsRecording] = useState(false);
    const mediaRecorderRef = useRef(null);
    const audioChunksRef = useRef([]);

    const handleToggleRecording = async () => {
        try {
            if (!isRecording) {
                const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                mediaRecorderRef.current = new MediaRecorder(stream);
                mediaRecorderRef.current.ondataavailable = (event) => {
                    audioChunksRef.current.push(event.data);
                };
                mediaRecorderRef.current.onstop = () => {
                    const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/wav' });

                    const audioUrl = URL.createObjectURL(audioBlob);

                    const downloadLink = document.createElement('a');
                    downloadLink.href = audioUrl;
                    downloadLink.download = 'recorded_audio.wav';
                    document.body.appendChild(downloadLink);
                    downloadLink.click();

                    document.body.removeChild(downloadLink);
                    URL.revokeObjectURL(audioUrl);
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


    return (
        <>
            <div className="container">
                <h2>Speech to Speech Converter</h2>
                <br/>
                <p>A tool that converts a bengali / english speech to an english / bengali speech in real time</p>

                <div className="main-contents">
                    <select id="selector1" className="mb-6 ml-32">
                        <option value="english-option1">English</option>
                        <option value="english-option2">Bengali</option>
                    </select>

                    <select id="selector2" className="mb-6 mr-32">
                        <option value="bengali-option1">English</option>
                        <option value="bengali-option2">Bengali</option>

                    </select>

                </div>
                <div className="main-contents">
                    <textarea disabled className="main-content">
                    </textarea>
                    <div className="m-10 cursor-pointer" onClick={handleToggleRecording}>
                        <FontAwesomeIcon icon={faMicrophone} className="text-blue-500 text-4xl" />
                    </div>
                    <textarea disabled className="main-content">
                    </textarea>
                </div>
            </div>
        </>
    );
}
