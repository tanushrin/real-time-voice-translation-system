import './App.css';
import {FontAwesomeIcon} from '@fortawesome/react-fontawesome';
import {faMicrophone} from '@fortawesome/free-solid-svg-icons';
import {useRef, useState} from 'react';
import axios from "axios";
import { Puff } from 'react-loader-spinner';

export default function App() {
    const [isRecording, setIsRecording] = useState(false);
    const mediaRecorderRef = useRef(null);
    const audioChunksRef = useRef([]);
    const [originalLanguage, setOriginalLanguage] = useState('english');
    const [toTranslateLanguage, setToTranslateLanguage] = useState('bengali');
    const audioRef = useRef(null);
    const [firstStepText, setFirstStepText] = useState('');
    const [secondStepText, setSecondStepText] = useState('');
    const [isLoading, setIsLoading] = useState(false);

    const simulateTyping = (text, setText) => {
        const chars = text.split('');
        let currentIndex = 0;

        const intervalId = setInterval(() => {
            if (currentIndex <= chars.length) {
                setText(chars.slice(0, currentIndex).join(''));
                currentIndex += 1;
            } else {
                clearInterval(intervalId);
            }
        }, 70);
    };

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
                    setIsRecording(false);
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
        setIsLoading(true)
        const formData = new FormData();
        formData.append('audio_file', audioFile);

        const queryParams = new URLSearchParams();
        queryParams.append('original_language', originalLanguage);
        queryParams.append('to_translate_language', toTranslateLanguage);


        try {
            const response = await axios.post(`http://localhost:8000/predict?${queryParams.toString()}`, formData, {
                headers: {
                    'Content-Type': 'multipart/form-data',
                },
                responseType: 'json'
            });
            setIsLoading(false)
            const translated_audio = response.data.third_step;
            simulateTyping(response.data.first_step || '', setFirstStepText);
            simulateTyping(response.data.second_step || '', setSecondStepText);


            if (translated_audio) {
                const decodedAudio = atob(translated_audio);

                const byteArray = new Uint8Array(decodedAudio.length);

                for (let i = 0; i < decodedAudio.length; i++) {
                    byteArray[i] = decodedAudio.charCodeAt(i);
                }

                const audioBlob = new Blob([byteArray], { type: 'audio/mpeg' });

                audioRef.current.src = URL.createObjectURL(audioBlob);
                audioRef.current.play();
            } else {
                console.error('Translated audio not found in the response.');
            }
        } catch (error) {
            console.error('Error calling the API', error);
        }

    }


    return (
        <>
            <div className="hidden">
                <audio ref={audioRef} autoPlay controls>.
                </audio>
            </div>
            <div className="container">
                <h2 className="green">BhashaBridge</h2>
                <br/>
                <p>A tool that translates English to Bengali or Bengali to English in real time</p>
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
                    <textarea disabled className="main-content" value={firstStepText}>
                    </textarea>
                    <div className="m-10 cursor-pointer" onClick={handleToggleRecording}>
                        {isLoading ? (
                            <Puff
                                height={40}
                                width={40}
                                radius={7}
                                color="green"
                                ariaLabel="loading"
                            />
                        ) : (
                            <FontAwesomeIcon
                                icon={faMicrophone}
                                size="2x"
                                className={`${isRecording ? 'red' : 'green'} text-4xl`}
                            />
                        )}
                    </div>

                    <textarea disabled className="main-content" value={secondStepText}>
                    </textarea>
                </div>
            </div>
        </>
    );
}
