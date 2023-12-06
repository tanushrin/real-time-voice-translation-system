# BhashsaBridge (a Real-Time Audio Translator)
## Project Description
BhashaBridge is a cutting-edge solution designed to translate audio in real time while preserving the gender identity of the speaker. This innovative tool is essential in today’s globalized world, ensuring effective communication across different languages without losing the essence of the speaker’s identity.
## Features
- **Real-Time Translation**: Instant translation of audio across multiple languages.
- **Gender Identity Preservation**: Maintains the gender characteristics of the speaker’s voice during translation.
- **Multi-Language Support**: Right now we have Bengali and English only.The scope can be widened to add more languages.
- **User-Friendly Interface**: Easy-to-use interface for seamless operation.
## Installation
In order to install the project and start the servers, please follow these steps:
- **Frontend**. Install necessary packages and run the server:
    ```
    cd frontend
    npm install
    npm start
    ```
- **Backend**. Install necessary packages and run the server:
  ```
  cd backend
  make reinstall_package
  make run_api
  ```

## Usage
  Once the installation steps are done, you should go to http://localhost:3000/. There you would see the main UI.
  You should be able to select which language you want to translate and which is your desired target language.
  Once you set this up, you can click on the microphone icon to start recording your message, after a few seconds, 
  your browser will play the transcription of your original message.
## Dependencies
  You should have Python installed in your computer, as well as node and npm for the frontend.  
  Moreover you should enable the 2 Google APIs used in the project, Speech-to-Text and Text-to-Speech.
  Be careful about this as this is a paid service.
## Contributing
We welcome contributions! If you would like to help improve or extend the project, please follow these steps:
1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes.
4. Commit your changes (`git commit -am ‘Add some feature’`).
5. Push to the branch (`git push origin feature-branch`).
6. Create a new Pull Request.
