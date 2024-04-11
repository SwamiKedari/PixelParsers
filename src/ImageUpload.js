import React, { useState } from 'react';
import axios from 'axios';
import './ImageUpload.css';

const PNG_FILE_URL = 'http://localhost:3000/table.docx';
const ImageUpload = () => {
  const [image, setImage] = useState(null);
  const [processedImage, setProcessedImage] = useState(null);
  const [textInfo, setTextInfo] = useState('');

  const handleImageUpload = (event) => {
    const file = event.target.files[0];
    setImage(file);
  };

  const downloadFileAtURL = (url) => {
    fetch(url)
      .then((response) => response.blob())
      .then((blob) => {
        const blobURL = window.URL.createObjectURL(new Blob([blob]));
        const fileName = url.split('/').pop();
        const aTag = document.createElement('a');
        aTag.href = url;
        aTag.setAttribute('download', fileName);
        document.body.appendChild(aTag);
        aTag.click();
        aTag.remove();
      });
  };

  const handleSubmit = async () => {
    const formData = new FormData();
    formData.append('image', image);

    try {
      const response = await axios.post('http://127.0.0.1:8000/api/process_image/', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      setProcessedImage(response.data.processed_image);
      setTextInfo(response.data.text_info);
    } catch (error) {
      console.error('Error processing image:', error);
    }
  };

  return (
    <div className="image-upload-container">
      <h2 className="upload-header">Upload Image</h2>
      <div className="upload-form">
        <input type="file" onChange={handleImageUpload} className="file-input" />
        <button onClick={handleSubmit} className="process-button">
          Process Image
        </button>
      </div>
      {processedImage && (
         <div className="download-container">
         <button onClick={() => downloadFileAtURL(PNG_FILE_URL)} className="download-button">
           Download .docx
         </button>
       </div>
      )}
     

      {processedImage && (
        <div className="result">
          <div className="image-container">
            <img src={`data:image/png;base64,${processedImage}`} alt="Processed" className="processed-image" />
          </div>
          <div className="text-info">
            <pre className="text-info-content">{textInfo}</pre>
          </div>
        </div>
      )}
    </div>
  );
};

export default ImageUpload;
